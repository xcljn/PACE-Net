# -*- coding: utf-8 -*-
"""
train.py - PACE-Net training entry point
=========================================
Reproduces the exact training recipe of the paper (combination used for
the final model):

  Recipe A: PFE v2 -- 3x3 receptive field + value-aware dual-channel
            condition [point mask, measurement/256];
  Recipe B: two-phase training --
            phase 1 freezes the first-level U-Net and trains only the
            second level (default 80 epochs, Adam 1e-4);
            phase 2 unfreezes everything and fine-tunes jointly with a
            small learning rate (default 20 epochs, Adam 2e-5);
  Recipe C: hard-example weighting on measured points,
            w = 1 + beta * min(|e|/tau, 1);
  Recipe D: EMA (exponential moving average) + cosine LR schedule;
            validation/model selection use the EMA weights.

Data protocol (identical for all baselines, see data.py):
  RadioMapSeer/DPM, 501/100/99 map split, numTx=10, thresh=0.2,
  10~300 random sparse points; loss L_total = L_global + l1*L_point
  + l2*L_spatial.

Inference is a pure forward pass: the model derives the point mask
internally from the measurement channel and directly outputs the
predicted radio map -- zero post-processing.

Starting weights:
  The paper's run was initialized from our best phase-1 (v1) checkpoint
  (`init/init_stage1.pt`, download from the GitHub Release). Non-PFE keys
  are loaded strictly; old v1 PFE keys are ignored and the v2 modules
  start from their zero-initialized identity mapping. Alternatively,
  `--baseline-weights` accepts the official RadioWNet_s second-level
  checkpoint; with an empty --init-weights the official baseline is used.

Usage:
  python train.py                                  # full 80+20 epochs
  python train.py --epochs 2 --stage2-epochs 1 \
      --max-train-batches 20 --max-val-batches 10  # smoke test
"""
import argparse
import copy
import json
import os
import sys
import time
from collections import defaultdict

import torch
from torch.optim import lr_scheduler

import data as cb
import pace_net as mdl
from point_losses import point_aware_loss, MetricAccumulator

_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHT_DIR = os.path.join(_HERE, "weights")

# Starting checkpoint of the paper's actual run (v1 best second level,
# old PFE keys ignored on load)
DEFAULT_INIT_W = os.path.join(_HERE, "init", "init_stage1.pt")
# Fallback: official RadioWNet_s_DPM_Thr2 second-level checkpoint
# (available from the official RadioUNet repository)
DEFAULT_BASELINE_W = os.path.join(_HERE, "init",
                                  "Trained_Model_SecondU_s.pt")


# ============================================================
# EMA: exponential moving average of parameters
# (validation / selection / saving all use the shadow weights)
# ============================================================
class ModelEMA:
    """Decay closer to 1 = stronger smoothing; non-float tensors
    (e.g. BatchNorm num_batches) are copied directly."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.shadow.items():
            cur = msd[k].detach()
            if torch.is_floating_point(v):
                v.mul_(self.decay).add_(cur, alpha=1.0 - self.decay)
            else:
                v.copy_(cur)

    def state(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state(self, sd):
        for k in self.shadow:
            self.shadow[k].copy_(sd[k])


def _load_init_weights(model, init_weights):
    """
    Load the starting checkpoint (e.g. the v1 best weights): all non-
    pfe_layers.* keys must exist with matching shapes; old PFE keys in the
    checkpoint are ignored (v1 and v2 PFE parameters are incompatible) and
    the v2 modules start from their zero-initialized identity mapping.
    """
    sd_model = model.state_dict()
    state = torch.load(init_weights, map_location=cb.DEVICE, weights_only=False)
    bad = [k for k in sd_model
           if not k.startswith("pfe_layers.")
           and (k not in state or state[k].shape != sd_model[k].shape)]
    if bad:
        raise RuntimeError(
            f"init-weights: missing or shape-mismatched non-PFE keys: {bad[:5]}")
    loadable = {k: v for k, v in state.items()
                if k in sd_model and not k.startswith("pfe_layers.")}
    sd_model.update(loadable)
    model.load_state_dict(sd_model)
    print(f"[INFO] Loaded starting weights {init_weights} "
          f"({len(loadable)} non-PFE keys; PFE v2 starts as identity)",
          flush=True)


@torch.no_grad()
def evaluate(model, loader, max_batches=None):
    """Compute point / whole-map RMSE, NMSE and latency on val/test
    (normalized gain space)."""
    model.eval()
    if cb.DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0, n_seen = time.time(), 0
    acc = MetricAccumulator()
    for bi, (inputs, target, pmask, hmask) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        inputs = inputs.to(cb.DEVICE)
        pred = model(inputs)[1]                        # second-level output
        acc.update(pred.cpu(), target, pmask, hmask)
        n_seen += inputs.size(0)
    if cb.DEVICE.type == "cuda":
        torch.cuda.synchronize()
    res = acc.result()
    res["ms_per_sample"] = (time.time() - t0) / max(n_seen, 1) * 1000.0
    model.train()
    return res


def main():
    p = argparse.ArgumentParser(description="PACE-Net training")
    p.add_argument("--epochs", type=int, default=80, help="phase-1 epochs")
    p.add_argument("--stage2-epochs", type=int, default=20,
                   help="phase-2 fine-tuning epochs")
    p.add_argument("--batch-size", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4, help="phase-1 LR")
    p.add_argument("--stage2-lr", type=float, default=2e-5, help="phase-2 LR")
    p.add_argument("--lambda1", type=float, default=1.5)
    p.add_argument("--lambda2", type=float, default=2.0)
    p.add_argument("--init-alpha", type=float, default=0.1)
    p.add_argument("--hard-beta", type=float, default=1.0,
                   help="hard-example weighting coefficient on points")
    p.add_argument("--hard-tau", type=float, default=0.02)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--init-weights", default=DEFAULT_INIT_W,
                   help="starting weights; empty string falls back to "
                        "--baseline-weights (official second level)")
    p.add_argument("--baseline-weights", default=DEFAULT_BASELINE_W,
                   help="official RadioWNet_s second-level checkpoint")
    p.add_argument("--data-dir", default=None,
                   help="path to RadioMapSeer (default: ./RadioMapSeer or "
                        "$RADIOMAPSEER_DIR)")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    args = p.parse_args()

    cb.set_seed()
    bundle = cb.make_bundle(batch_size=args.batch_size, data_dir=args.data_dir)
    loaders_ = bundle["loaders"]

    # ---- Build model (PFE v2) ----
    model = mdl.PointAwareWNetV2(
        inputs=3, phase="secondU",
        init_alpha=args.init_alpha).to(cb.DEVICE)
    if args.init_weights and os.path.exists(args.init_weights):
        _load_init_weights(model, args.init_weights)
    elif args.baseline_weights and os.path.exists(args.baseline_weights):
        base_state = torch.load(args.baseline_weights, map_location=cb.DEVICE,
                                weights_only=False)
        pfe_keys = mdl.load_baseline_to_point_aware(model, base_state)
        print(f"[INFO] Loaded official baseline weights; {len(pfe_keys)} new "
              f"PFE v2 parameter keys (zero-init identity)", flush=True)
    else:
        raise FileNotFoundError(
            "No starting weights found. Download init/init_stage1.pt from "
            "the GitHub Release page, or pass --init-weights / "
            "--baseline-weights explicitly.")

    mdl.freeze_first_u(model)                      # phase 1: freeze level 1
    total, trainable, pfe_n = mdl.count_parameters(model)
    print(f"[INFO] Total params {total/1e6:.4f}M | trainable "
          f"{trainable/1e6:.4f}M | PFE v2 added {pfe_n} "
          f"({pfe_n/total*100:.3f}%)", flush=True)

    ema = ModelEMA(model, args.ema_decay)
    total_epochs = args.epochs + args.stage2_epochs
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    def make_opt_sch(stage):
        """Optimizer + cosine schedule per phase; phase 2 also unfreezes
        all parameters and disables detach on the first level."""
        if stage == 1:
            opt = torch.optim.Adam(
                filter(lambda x: x.requires_grad, model.parameters()),
                lr=args.lr)
            sch = lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
        else:
            for x in model.parameters():
                x.requires_grad = True
            model.detach_first = False             # enable level-1 gradients
            opt = torch.optim.Adam(model.parameters(), lr=args.stage2_lr)
            sch = lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(args.stage2_epochs, 1))
        return opt, sch

    # ---- Persist training configuration ----
    with open(os.path.join(WEIGHT_DIR, "config.json"), "w",
              encoding="utf-8") as f:
        json.dump({"group": "D", "pfe_v2": True,
                   "lambda1": args.lambda1, "lambda2": args.lambda2,
                   "init_alpha": args.init_alpha,
                   "epochs": args.epochs,
                   "stage2_epochs": args.stage2_epochs,
                   "stage2_lr": args.stage2_lr,
                   "batch_size": args.batch_size, "lr": args.lr,
                   "scheduler": "cosine",
                   "hard_beta": args.hard_beta, "hard_tau": args.hard_tau,
                   "ema_decay": args.ema_decay,
                   "init_weights": args.init_weights}, f,
                  ensure_ascii=False, indent=2)

    best, log, cur_stage = 1e18, [], 1
    opt, sch = make_opt_sch(1)
    print("=" * 60 + f"\n[PACE-Net] Training epochs={args.epochs}+"
          f"{args.stage2_epochs}, lambda1={args.lambda1}, "
          f"lambda2={args.lambda2}\n" + "=" * 60, flush=True)
    for epoch in range(total_epochs):
        stage = 1 if epoch < args.epochs else 2
        if stage != cur_stage:
            cur_stage = stage
            print("[phase2] Unfreezing first-level U-Net; full-model "
                  "fine-tuning with a small LR", flush=True)
            opt, sch = make_opt_sch(2)
        model.train()
        t_ep, agg, nb = time.time(), defaultdict(float), 0
        for inputs, target, pmask, _ in loaders_["train"]:
            if args.max_train_batches is not None and nb >= args.max_train_batches:
                break
            inputs = inputs.to(cb.DEVICE)
            target = target.to(cb.DEVICE)
            pmask = pmask.to(cb.DEVICE)
            opt.zero_grad()
            pred2 = model(inputs, pmask)[1]
            loss, parts = point_aware_loss(
                pred2, target, pmask,
                args.lambda1, args.lambda2,
                hard_beta=args.hard_beta, hard_tau=args.hard_tau)
            loss.backward()
            opt.step()
            ema.update(model)
            for k, v in parts.items():
                agg[k] += v
            nb += 1
        sch.step()

        # ---- Validation: temporarily swap in the EMA shadow weights ----
        cur_sd = copy.deepcopy(model.state_dict())
        model.load_state_dict(ema.shadow)
        val_res = evaluate(model, loaders_["val"], args.max_val_batches)
        is_best = val_res["point_rmse"] < best
        if is_best:
            best = val_res["point_rmse"]
            torch.save(model.state_dict(),            # i.e. the EMA weights
                       os.path.join(WEIGHT_DIR, "model_best.pt"))
        model.load_state_dict(cur_sd)
        del cur_sd

        msg = " ".join(f"{k}={v/max(nb,1):.5f}" for k, v in agg.items())
        row = {"epoch": epoch, "stage": cur_stage,
               "lr": opt.param_groups[0]["lr"], **{k: v/max(nb,1)
                                                   for k, v in agg.items()},
               "val_point_rmse": val_res["point_rmse"],
               "val_map_rmse": val_res["map_rmse"]}
        log.append(row)
        print(f"[pace_net] ep{epoch:03d}/{total_epochs-1} (phase{cur_stage}) "
              f"lr={opt.param_groups[0]['lr']:.2e} | {msg} || "
              f"val point RMSE={val_res['point_rmse']:.5f} "
              f"val map RMSE={val_res['map_rmse']:.5f}"
              + (" | *new best*" if is_best else "")
              + f" | {time.time()-t_ep:.0f}s", flush=True)
        with open(os.path.join(WEIGHT_DIR, "train_log.json"), "w",
                  encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)

    # ---- Finalize: the best (EMA) validation weights are the final model ----
    best_path = os.path.join(WEIGHT_DIR, "model_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=cb.DEVICE,
                                         weights_only=False))
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "model_final.pt"))

    # Record the final learnable alphas (did the modules learn real gains?)
    alpha_log = {name: float(p.detach().cpu())
                 for name, p in model.named_parameters()
                 if name.endswith(".alpha")}
    with open(os.path.join(WEIGHT_DIR, "alpha_final.json"), "w",
              encoding="utf-8") as f:
        json.dump(alpha_log, f, ensure_ascii=False, indent=2)
    print("[pace_net] Final alphas: " + ", ".join(
        f"{k.split('.')[1]}={v:.3f}" for k, v in alpha_log.items()),
        flush=True)

    # ---- Test-set evaluation ----
    test_res = evaluate(model, loaders_["test"])
    base_n = sum(p.numel()
                 for p in mdl.modules.RadioWNet(inputs=3,
                                                phase="secondU").parameters())
    test_res.update({"pfe_v2": True, "total_params": total,
                     "pfe_params": pfe_n,
                     "param_ratio_vs_baseline": pfe_n / base_n})
    with open(os.path.join(WEIGHT_DIR, "test_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(test_res, f, ensure_ascii=False, indent=2)
    print(f"[pace_net] Done. Best val point RMSE={best:.5f}; test: "
          f"point RMSE={test_res['point_rmse']:.5f} "
          f"map RMSE={test_res['map_rmse']:.5f} "
          f"NMSE={test_res['map_nmse']:.5f}", flush=True)


if __name__ == "__main__":
    main()
