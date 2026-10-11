# -*- coding: utf-8 -*-
"""
pace_net.py - PACE-Net: Point-Aware Conditional Enhancement Network
====================================================================
This file implements PACE-Net on top of the original RadioWNet backbone
(same-directory modules.py, from the official RadioUNet implementation):

1. Component A: build_point_mask -- internal point-mask generation
   - Derives the binary point mask M internally from the non-zero locations
     of the sparse-measurement input channel (channel 3);
   - Training and inference share the same source: at inference time no
     external coordinate list and no post-processing are required.

2. Component B: PointFeatureEnhanceV2 -- point feature enhancement module
   (3x3 receptive field + value-aware dual-channel condition)
   - Condition C = [M, measurement/256]: the module perceives both WHERE the
     measured points are and HOW STRONG each measured signal is;
   - 3x3 gated convolution (padding=1): enhancement at a measured point can
     reference neighborhood context, spreading isolated point information
     into surrounding features;
   - Feature enhancement: F_enh = F * (1 + alpha * W_gate * M_down)
   - Degeneracy principle: conv weights/bias are zero-initialized so the
     initial W_gate = 0 and the module is an identity mapping; loading the
     baseline weights therefore reproduces the original W-Net exactly, and
     the per-location enhancement/suppression magnitudes are learned
     adaptively during training. alpha is an independent learnable scalar
     per layer (initial 0.1).

3. PointAwareWNetV2 -- the PACE-Net model
   - Inherits the original RadioWNet and inserts PointFeatureEnhanceV2
     after every skip concatenation in the second-level (correction)
     U-Net decoder (9 locations in total); point locations and measured
     values are embedded as conditions into the intermediate feature flow;

Parameter count: nine 3x3 convolutions (channels 220/180/140/120/100/80/
60/44/24, each taking 2 input maps) + 9 alphas, ~18.4k new parameters,
i.e. ~0.14% of the original 13.28M total, far below the 15% budget.
"""
import importlib.util
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Load the same-directory modules.py (official RadioWNet from RadioUNet)
# explicitly by file path. We avoid `import modules` so that same-named
# modules under other sys.path entries cannot shadow it; this keeps the
# folder self-contained with a well-defined backbone source.
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "pace_net_modules", os.path.join(_HERE, "modules.py"))
modules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(modules)


# ============================================================
# Component A: internal point-mask generation (train/inference identical)
# ============================================================
def build_point_mask(inputs, meas_channel=2):
    """
    Generate the binary point mask M internally from the sparse-measurement
    input channel.

    Args:
        inputs: model input [B, C, 256, 256]; for C=3 the channel order is
                [building map, transmitter map, sparse measurements].
        meas_channel: index of the sparse-measurement channel
                (RadioUNet_s convention: 2).
    Returns:
        M: [B, 1, 256, 256] float binary mask; 1 at measured points, 0 else.
    Note:
        At inference time no external coordinate list is available, so a
        measured point is simply "non-zero in the measurement channel".
        During training the loader produces the exact mask (including
        points whose measured value happens to be 0) and passes it in
        explicitly; both paths are handled identically downstream.
    """
    meas = inputs[:, meas_channel:meas_channel + 1, :, :]
    return (meas.abs() > 0).to(inputs.dtype)


# ============================================================
# Component B: point feature enhancement module v2 (3x3 + value-aware)
# ============================================================
class PointFeatureEnhanceV2(nn.Module):
    """
    Point-conditioned feature enhancement:
        C_down = nearest_downsample([M, measurement/256])   # match feat size
        W_gate = 2 * sigmoid( Conv3x3(C_down) ) - 1         # per-channel gate
        F_enh  = F * (1 + alpha * W_gate * M_down)          # only at points

    - Modulation happens only where M_down = 1; non-measured regions are
      carried over unchanged (no global leakage);
    - Conv3x3 zero-initialized => initial W_gate = 0 => identity mapping,
      so training starts smoothly from the baseline.
    """

    def __init__(self, channels, init_alpha=0.1):
        super().__init__()
        # 3x3 conv: dual-channel condition map -> per-channel gate
        self.attn_conv = nn.Conv2d(2, channels, kernel_size=3, padding=1,
                                   bias=True)
        # Zero init guarantees the module starts as an identity mapping
        nn.init.zeros_(self.attn_conv.weight)
        nn.init.zeros_(self.attn_conv.bias)
        # Learnable guidance scalar alpha (per layer), small positive init
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, feat, cond_full):
        """
        Args:
            feat:      current decoder feature [B, C, h, w]
            cond_full: full-resolution dual-channel condition
                       [B, 2, 256, 256] = [point mask, measurement/256]
        Returns:
            Enhanced feature (same shape)
        """
        # Nearest-neighbor downsampling: points stay on their original
        # pixel grid, no bilinear smoothing.
        cond_down = F.interpolate(cond_full, size=feat.shape[-2:],
                                  mode="nearest")
        m_down = cond_down[:, 0:1]  # mask channel: modulate only at points
        gate = 2.0 * torch.sigmoid(self.attn_conv(cond_down)) - 1.0  # (-1, 1)
        return feat * (1.0 + self.alpha * gate * m_down)


# ============================================================
# PACE-Net (point-aware W-UNet v2)
# ============================================================
class PointAwareWNetV2(modules.RadioWNet):
    """
    PACE-Net: identical architecture to the original RadioWNet except that
    PointFeatureEnhanceV2 (3x3 receptive field + [point mask, measurement/256]
    dual-channel condition) is inserted after each of the 9 skip
    concatenations in the second-level decoder.

    detach_first (two-phase training switch):
      True (default): the entire first-level output is detached; only the
                      second level is trained (phase 1);
      False          : gradients flow into the first level as well, used for
                       full-model fine-tuning in phase 2.
    Inference path: when point_mask=None the mask is generated internally
    from the measurement channel; the model directly outputs the predicted
    radio map with zero post-processing.
    """

    def __init__(self, inputs=3, phase="secondU", init_alpha=0.1,
                 detach_first=True):
        super().__init__(inputs=inputs, phase=phase)
        if phase != "secondU":
            raise ValueError("PointAwareWNetV2 only supports phase='secondU'")
        self.detach_first = bool(detach_first)

        # Channel counts at the 9 insertion points (must match the
        # second-level decoder in modules.py exactly)
        pfe_channels = {
            "pfe4": 110 + 110,                 # Wconv_up5 out cat Wlayer4
            "pfe3": 90 + 90,                   # Wconv_up4 out cat Wlayer3
            "pfe20": 70 + 70,                  # Wconv_up3 out cat Wlayer20
            "pfe2": 60 + 60,                   # Wconv_up20 out cat Wlayer2
            "pfe10": 50 + 50,                  # Wconv_up2 out cat Wlayer10
            "pfe1": 40 + 40,                   # Wconv_up10 out cat Wlayer1
            "pfe0": 30 + 30,                   # Wconv_up1 out cat Wlayer0
            "pfe00": 20 + 20 + (inputs + 1),   # Wconv_up0 out cat Winput
            "pfe000": 20 + (inputs + 1),       # Wconv_up00 out cat Winput
        }
        self.pfe_layers = nn.ModuleDict({
            name: PointFeatureEnhanceV2(ch, init_alpha=init_alpha)
            for name, ch in pfe_channels.items()})

    def forward(self, input, point_mask=None):
        """
        Args:
            input: [B, inputs, 256, 256]; channel 2 is the sparse
                   measurement map (0~256 scale)
            point_mask: optional [B, 1, 256, 256]; generated internally
                   from the input when None
        Returns:
            [output1, output2]: first-level and second-level predictions,
            same interface as the original RadioWNet
        """
        if point_mask is None:
            point_mask = build_point_mask(input)
        # Dual-channel condition: [point mask, measurement/256]
        # (measurements normalized to the [0, 1] gain space)
        cond_full = torch.cat([point_mask, input[:, 2:3] / 256.0], dim=1)

        det = self.detach_first  # phase1=True (frozen first level); phase2=False

        def d(x):
            return x.detach() if det else x

        input0 = input[:, 0:self.inputs, :, :]

        # ------------------------------------------------------------
        # First-level U-Net (unchanged; gradients gated by detach_first)
        # ------------------------------------------------------------
        layer00 = d(self.layer00(input0))
        layer0 = d(self.layer0(layer00))
        layer1 = d(self.layer1(layer0))
        layer10 = d(self.layer10(layer1))
        layer2 = d(self.layer2(layer10))
        layer20 = d(self.layer20(layer2))
        layer3 = d(self.layer3(layer20))
        layer4 = d(self.layer4(layer3))
        layer5 = d(self.layer5(layer4))

        layer4u = d(self.conv_up5(layer5))
        layer4u = d(torch.cat([layer4u, layer4], dim=1))
        layer3u = d(self.conv_up4(layer4u))
        layer3u = d(torch.cat([layer3u, layer3], dim=1))
        layer20u = d(self.conv_up3(layer3u))
        layer20u = d(torch.cat([layer20u, layer20], dim=1))
        layer2u = d(self.conv_up20(layer20u))
        layer2u = d(torch.cat([layer2u, layer2], dim=1))
        layer10u = d(self.conv_up2(layer2u))
        layer10u = d(torch.cat([layer10u, layer10], dim=1))
        layer1u = d(self.conv_up10(layer10u))
        layer1u = d(torch.cat([layer1u, layer1], dim=1))
        layer0u = d(self.conv_up1(layer1u))
        layer0u = d(torch.cat([layer0u, layer0], dim=1))
        layer00u = d(self.conv_up0(layer0u))
        layer00u = d(torch.cat([layer00u, layer00], dim=1))
        layer00u = d(torch.cat([layer00u, input0], dim=1))
        layer000u = d(self.conv_up00(layer00u))
        layer000u = d(torch.cat([layer000u, input0], dim=1))
        output1 = d(self.conv_up000(layer000u))

        # ------------------------------------------------------------
        # Second-level correction U-Net (encoder unchanged)
        # ------------------------------------------------------------
        Winput = torch.cat([output1, input], dim=1)
        Wlayer00 = self.Wlayer00(Winput)
        Wlayer0 = self.Wlayer0(Wlayer00)
        Wlayer1 = self.Wlayer1(Wlayer0)
        Wlayer10 = self.Wlayer10(Wlayer1)
        Wlayer2 = self.Wlayer2(Wlayer10)
        Wlayer20 = self.Wlayer20(Wlayer2)
        Wlayer3 = self.Wlayer3(Wlayer20)
        Wlayer4 = self.Wlayer4(Wlayer3)
        Wlayer5 = self.Wlayer5(Wlayer4)

        # Decoder: insert PFE v2 after each skip concatenation
        # (condition = point locations + measured values)
        def enhance(name, feat):
            return self.pfe_layers[name](feat, cond_full)

        Wlayer4u = self.Wconv_up5(Wlayer5)
        Wlayer4u = torch.cat([Wlayer4u, Wlayer4], dim=1)
        Wlayer4u = enhance("pfe4", Wlayer4u)

        Wlayer3u = self.Wconv_up4(Wlayer4u)
        Wlayer3u = torch.cat([Wlayer3u, Wlayer3], dim=1)
        Wlayer3u = enhance("pfe3", Wlayer3u)

        Wlayer20u = self.Wconv_up3(Wlayer3u)
        Wlayer20u = torch.cat([Wlayer20u, Wlayer20], dim=1)
        Wlayer20u = enhance("pfe20", Wlayer20u)

        Wlayer2u = self.Wconv_up20(Wlayer20u)
        Wlayer2u = torch.cat([Wlayer2u, Wlayer2], dim=1)
        Wlayer2u = enhance("pfe2", Wlayer2u)

        Wlayer10u = self.Wconv_up2(Wlayer2u)
        Wlayer10u = torch.cat([Wlayer10u, Wlayer10], dim=1)
        Wlayer10u = enhance("pfe10", Wlayer10u)

        Wlayer1u = self.Wconv_up10(Wlayer10u)
        Wlayer1u = torch.cat([Wlayer1u, Wlayer1], dim=1)
        Wlayer1u = enhance("pfe1", Wlayer1u)

        Wlayer0u = self.Wconv_up1(Wlayer1u)
        Wlayer0u = torch.cat([Wlayer0u, Wlayer0], dim=1)
        Wlayer0u = enhance("pfe0", Wlayer0u)

        Wlayer00u = self.Wconv_up0(Wlayer0u)
        Wlayer00u = torch.cat([Wlayer00u, Wlayer00], dim=1)
        Wlayer00u = torch.cat([Wlayer00u, Winput], dim=1)
        Wlayer00u = enhance("pfe00", Wlayer00u)

        Wlayer000u = self.Wconv_up00(Wlayer00u)
        Wlayer000u = torch.cat([Wlayer000u, Winput], dim=1)
        Wlayer000u = enhance("pfe000", Wlayer000u)

        output2 = self.Wconv_up000(Wlayer000u)
        return [output1, output2]


# ============================================================
# Training helpers
# ============================================================
def freeze_first_u(model):
    """
    Freeze the first-level U-Net parameters (layer names starting with
    `layer` / `conv_up`); the second-level W* layers and pfe_* modules
    stay trainable. Together with the detach inside the phase-1 forward,
    this guarantees the first level receives no updates.
    """
    n_frozen = 0
    for name, p in model.named_parameters():
        if name.startswith("layer") or name.startswith("conv_up"):
            p.requires_grad = False
            n_frozen += p.numel()
    return n_frozen


def count_parameters(model):
    """Return (total, trainable, PFE-added) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pfe = sum(p.numel() for name, p in model.named_parameters()
              if name.startswith("pfe_layers."))
    return total, trainable, pfe


def load_baseline_to_point_aware(model_pa, baseline_state):
    """
    Load original RadioWNet (same `inputs`) baseline weights into
    PointAwareWNetV2. All keys except pfe_layers.* must match exactly
    (names and shapes) to avoid silent misloads; the PFE parameters keep
    their zero-initialized identity mapping.
    """
    model_state = model_pa.state_dict()
    missing = [k for k in model_state.keys()
               if k not in baseline_state and not k.startswith("pfe_layers.")]
    unexpected = [k for k in baseline_state.keys() if k not in model_state]
    if missing or unexpected:
        raise RuntimeError(
            f"Baseline weights do not match PointAwareWNetV2: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}")
    loadable = {k: v for k, v in baseline_state.items()
                if k in model_state and not k.startswith("pfe_layers.")}
    model_state.update(loadable)
    model_pa.load_state_dict(model_state)
    pfe_keys = [k for k in model_state if k.startswith("pfe_layers.")]
    return pfe_keys
