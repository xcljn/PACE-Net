# -*- coding: utf-8 -*-
"""
point_losses.py - Point-aware weighted losses and evaluation metrics.

Total loss: L_global (whole-map MSE) + lambda1 * L_point (MSE on the
sparse point mask) + lambda2 * L_spatial (gradient L1). All terms are
computed in the [0,1] gain space (GAIN_SCALE=256).
"""
import torch
import torch.nn.functional as F

GAIN_SCALE = 256.0  # RadioUNet_s target gain scale (errors also /256)


def point_aware_loss(pred, target, point_mask,
                     lambda1=0.5, lambda2=2.0, gain_scale=GAIN_SCALE,
                     hard_beta=0.0, hard_tau=0.02):
    """
    Args:
        pred:       model prediction [B, 1, H, W] (0~256 scale)
        target:     ground truth      [B, 1, H, W]
        point_mask: binary point mask [B, 1, H, W], 1 at measured points
        lambda1:    weight of the point loss
        lambda2:    weight of the spatial-gradient loss
    Returns:
        total: scalar loss (averaged over the batch)
        parts: dict with the three individual terms (normalized gain space,
               convenient for logging / ablation analysis)
    """
    pred_n = pred / gain_scale
    target_n = target / gain_scale

    # 1) whole-map MSE
    l_global = F.mse_loss(pred_n, target_n)

    # 2) point MSE: select by mask, normalize per sample by its point count,
    #    then average over the batch. Each sample has a different number of
    #    points (10~300); per-sample normalization prevents samples with
    #    many points from dominating the loss.
    B = pred_n.size(0)
    err = pred_n - target_n
    sqerr = (err ** 2) * point_mask
    if hard_beta > 0:
        # Hard-example weighting on points (focal-style): points with
        # larger errors get higher weights, w = 1 + beta*min(|e|/tau, 1);
        # the weights themselves carry no gradient (no_grad). beta=0
        # reduces strictly to the plain point MSE.
        if hard_tau <= 0:
            raise ValueError("hard_beta>0 requires hard_tau>0")
        with torch.no_grad():
            w_hard = 1.0 + hard_beta * torch.clamp(err.abs() / hard_tau,
                                                   max=1.0)
        sqerr = sqerr * w_hard
    point_sse = sqerr.reshape(B, -1).sum(dim=1)
    point_cnt = point_mask.reshape(B, -1).sum(dim=1).clamp_min(1.0)
    l_point = (point_sse / point_cnt).mean()

    # 3) spatial gradient L1 (along x and y, aligned with the GT gradients)
    gx_p = pred_n[..., :, 1:] - pred_n[..., :, :-1]
    gx_t = target_n[..., :, 1:] - target_n[..., :, :-1]
    gy_p = pred_n[..., 1:, :] - pred_n[..., :-1, :]
    gy_t = target_n[..., 1:, :] - target_n[..., :-1, :]
    l_spatial = F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)

    total = l_global + lambda1 * l_point + lambda2 * l_spatial
    parts = {
        "L_global": float(l_global.detach()),
        "L_point": float(l_point.detach()),
        "L_spatial": float(l_spatial.detach()),
        "L_total": float(total.detach()),
    }
    return total, parts


# ============================================================
# Evaluation metrics (normalized [0,1] gain space; fixed test protocol)
# ============================================================
class MetricAccumulator:
    """
    Accumulates MSE numerators/denominators weighted by pixel/sample counts
    and finally reports:
      - point_rmse   : RMSE at the measured input points (core metric)
      - heldout_rmse : RMSE at 300 held-out points never shown to the model
                       (generalization control: shows the network truly
                       learned point-conditioned correction rather than
                       copying the input channel)
      - map_rmse     : whole-map RMSE
      - map_nmse     : whole-map NMSE = sum(err^2)/sum(target^2)
    """

    def __init__(self, gain_scale=GAIN_SCALE):
        self.s = gain_scale
        self.reset()

    def reset(self):
        self.p_sse = 0.0     # squared error at measured points
        self.p_cnt = 0       # number of measured pixels
        self.h_sse = 0.0     # held-out points
        self.h_cnt = 0
        self.m_sse = 0.0     # whole map
        self.m_pow = 0.0     # ground-truth energy
        self.n = 0           # number of samples

    @torch.no_grad()
    def update(self, pred, target, point_mask, heldout_mask=None):
        pred_n = pred / self.s
        target_n = target / self.s
        sq = (pred_n - target_n) ** 2

        self.p_sse += float((sq * point_mask).sum())
        self.p_cnt += int(point_mask.sum())
        if heldout_mask is not None:
            self.h_sse += float((sq * heldout_mask).sum())
            self.h_cnt += int(heldout_mask.sum())
        self.m_sse += float(sq.sum())
        self.m_pow += float((target_n ** 2).sum())
        self.n += pred.size(0)

    def result(self):
        out = {
            "point_rmse": (self.p_sse / max(self.p_cnt, 1)) ** 0.5,
            "map_rmse": (self.m_sse / max(self.n * 256 * 256, 1)) ** 0.5,
            "map_nmse": self.m_sse / max(self.m_pow, 1e-12),
            "num_samples": self.n,
            "num_point_pixels": self.p_cnt,
        }
        if self.h_cnt > 0:
            out["heldout_rmse"] = (self.h_sse / self.h_cnt) ** 0.5
        return out
