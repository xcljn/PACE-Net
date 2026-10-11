# -*- coding: utf-8 -*-
"""
evaluate.py - Reproduce the paper's test-set numbers
=====================================================
Loads a trained PACE-Net checkpoint and evaluates it on the RadioMapSeer
test split (99 maps x 10 Tx = 990 samples) with the exact paper protocol
(DPM, thresh=0.2, 10~300 random sparse points per sample, seed 20260922).

Reports (normalized [0,1] gain space, paper Table 1):
  point_rmse    RMSE at the measured input points     (paper: 0.0126)
  heldout_rmse  RMSE at 300 held-out points per map   (paper: 0.0318)
  map_rmse      whole-map RMSE                        (paper: 0.0316)
  map_nmse      whole-map NMSE                        (paper: 0.0183)

The seeding order below reproduces the paper's numbers bit-exactly.

Usage:
  python evaluate.py                                # weights/model_best.pt
  python evaluate.py --weights path/to/model_best.pt
  python evaluate.py --max-batches 10               # quick check
"""
import argparse
import json
import os
import time

import torch

import data as cb
from pace_net import PointAwareWNetV2
from point_losses import MetricAccumulator

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(_HERE, "weights", "model_best.pt")


@torch.no_grad()
def run_eval(model, loader, max_batches=None, verbose=True):
    model.eval()
    if cb.DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0, n_seen = time.time(), 0
    acc = MetricAccumulator()
    for bi, (inputs, target, pmask, hmask) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        inputs = inputs.to(cb.DEVICE)
        pred = model(inputs)[1]                     # second-level output
        acc.update(pred.cpu(), target, pmask, hmask)
        n_seen += inputs.size(0)
        if verbose and (bi + 1) % 10 == 0:
            print(f"  ...batch {bi + 1}", flush=True)
    if cb.DEVICE.type == "cuda":
        torch.cuda.synchronize()
    res = acc.result()
    res["ms_per_sample"] = (time.time() - t0) / max(n_seen, 1) * 1000.0
    return res


def main():
    ap = argparse.ArgumentParser(description="PACE-Net test-set evaluation")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--data-dir", default=None,
                    help="path to RadioMapSeer (default: ./RadioMapSeer or "
                         "$RADIOMAPSEER_DIR)")
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="limit number of test batches (quick check)")
    ap.add_argument("--out", default=None,
                    help="optional JSON path to save the metrics")
    args = ap.parse_args()

    print(f"[device] {cb.DEVICE}")
    print(f"[weights] {args.weights}")
    model = PointAwareWNetV2(inputs=3, phase="secondU")
    state = torch.load(args.weights, map_location=cb.DEVICE,
                       weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(cb.DEVICE)

    total = sum(p.numel() for p in model.parameters())
    pfe = sum(p.numel() for n, p in model.named_parameters()
              if n.startswith("pfe_layers."))
    print(f"[model] total params {total/1e6:.4f}M | PFE added {pfe} "
          f"({pfe/total*100:.3f}%)")

    # IMPORTANT seeding order (reproduces the paper's numbers exactly):
    # the loaders are built FIRST (loader construction internally reseeds
    # numpy to 42 for the deterministic 501/100/99 map split), and the
    # sparse-point sampling seed is set AFTERWARDS, matching the paper's
    # evaluation script.
    bundle = cb.make_bundle(batch_size=args.batch_size, data_dir=args.data_dir)
    cb.set_seed()
    res = run_eval(model, bundle["loaders"]["test"], args.max_batches)

    print("\n===== Test results (normalized gain space) =====")
    print(f"  point RMSE    : {res['point_rmse']:.5f}   (paper: 0.0126)")
    print(f"  held-out RMSE : {res.get('heldout_rmse', float('nan')):.5f}"
          f"   (paper: 0.0318)")
    print(f"  map RMSE      : {res['map_rmse']:.5f}   (paper: 0.0316)")
    print(f"  map NMSE      : {res['map_nmse']:.5f}   (paper: 0.0183)")
    print(f"  latency       : {res['ms_per_sample']:.2f} ms/sample")
    print(f"  samples       : {res['num_samples']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
