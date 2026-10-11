# PACE-Net: Point-Aware Conditional Enhancement Network for Radio Map Reconstruction from Sparse Observations

Official PyTorch implementation of PACE-Net.

High-resolution radio maps must be reconstructed from extremely sparse
field measurements — on a 256×256 grid the measured pixels cover barely
0.5% of the image. These are the *only* pixels with ground-truth
anchoring, yet their contribution is diluted in global losses and smoothed
away by convolution.

**PACE-Net** is a point-aware conditional enhancement network that anchors
on measured points and radiates their information across the feature
stream:

- **PFE (Point Feature Enhancement) module** — uses the point mask and the
  measured values as a two-channel condition to drive *zero-initialized*
  gated convolutions at **nine** decoder positions. Adds only **+0.139%**
  parameters (18.4k on a 13.28M backbone) and starts as an exact identity
  mapping, so training always improves on the baseline.
- **Point-aware loss** — point weighting (with hard-example up-weighting)
  + a spatial gradient constraint on the whole-map reconstruction.
- **Two-phase training** — freezes the coarse U-Net for 80 epochs, then
  briefly fine-tunes the whole network for 20 epochs.

<p align="center">
  <img src="assets/architecture.png" alt="PACE-Net architecture" width="820"/>
</p>

## Results

On 990 RadioMapSeer test samples (99 maps × 10 transmitters, 10–300
randomly drawn sparse measurements per sample), PACE-Net is compared with
nine methods under identical conditions (Kriging, RBF, DeepAE, UNet, cGAN,
RadioUNet, RME-GAN, ACT-GAN, TAN) and ranks first on global RMSE,
held-out RMSE and NMSE:

| Model | Point RMSE ↓ | Held-out RMSE ↓ | Global RMSE ↓ | Global NMSE ↓ |
|---|---|---|---|---|
| RadioUNet (backbone) | 0.0271 | 0.0397 | 0.0394 | 0.0286 |
| **PACE-Net (ours)** | **0.0126** | **0.0318** | **0.0316** | **0.0183** |

Relative to RadioUNet: global RMSE **−19.8%**, NMSE **−36.0%**,
measurement-point RMSE **−53.5%**, with a negligible **+0.139%** model-size
increase. On a real campus with at most 300 field measurements, the
predictions reach **0.901 correlation** and **3.42 dBm** RMSE at the
measurement points (within the 5 dBm engineering margin).

## Repository layout

```
PACE-Net/
├── pace_net.py          # PACE-Net model (PointAwareWNetV2 + PFE v2)
├── modules.py           # RadioWNet backbone (from the RadioUNet repository)
├── point_losses.py      # Point-aware loss + evaluation metrics
├── loaders.py           # RadioUNet_s_points dataset (derived from RadioUNet)
├── data.py              # Data pipeline: 501/100/99 split, 10~300 points
├── train.py             # Two-phase training recipe
├── evaluate.py          # Verify the trained model on the test split
├── requirements.txt
├── LICENSE              # CC BY-NC 4.0
└── assets/              # Architecture figure used in this README
```

No pre-trained checkpoints are distributed; `train.py` writes the trained
weights to `weights/` (created automatically).

## 1. Setup

```bash
git clone https://github.com/xcljn/PACE-Net.git
cd PACE-Net
pip install -r requirements.txt
```

## 2. Dataset: RadioMapSeer

Download the [RadioMapSeer](https://ieee-dataport.org/open-access/radiomapseer)
dataset (free from IEEE DataPort after registration; DPM simulation
variant). Extract it so that the directory layout is:

```
PACE-Net/RadioMapSeer/
├── png/buildings_complete/   # 700 building maps
├── png/antennas/             # 700×80 transmitter maps
└── gain/DPM/                 # 700×80 gain maps
```

Alternatively, set the environment variable `RADIOMAPSEER_DIR` or pass
`--data-dir` to point to the dataset location.

The 501/100/99 train/val/test map split is the deterministic standard
RadioUNet split (`np.random.seed(42)` inside `loaders.py`), and the random
sparse-point sampling is governed by the seed in `data.set_seed(20260922)`,
applied after loader construction — both are fixed, so `evaluate.py`
is deterministic and bit-for-bit reproducible across runs and machines.

## 3. Train

```bash
python train.py
```

- Phase 1: first-level U-Net frozen, second level trained for 80 epochs
  (Adam 1e-4, cosine schedule).
- Phase 2: full model fine-tuned for 20 epochs (Adam 2e-5).
- Loss: `L = L_global + λ1·L_point + λ2·L_spatial` (λ1=1.5, λ2=2.0) with
  hard-example point weighting (β=1, τ=0.02).
- EMA (decay 0.999); validation/model selection use the EMA weights.
- Batch 15 fits an 8 GB GPU.

Smoke test on a subset:

```bash
python train.py --epochs 2 --stage2-epochs 1 --max-train-batches 20 --max-val-batches 10
```

Starting weights: by default `train.py` expects an optional starting
second-level checkpoint via `--init-weights`; when neither it nor
`--baseline-weights` is supplied, it exits with an explicit message
instead of silently starting from random weights. Two supported options:

- `--baseline-weights` with the official RadioWNet_s second-level
  checkpoint (`Trained_Model_SecondU_s.pt`) from the
  [RadioUNet repository](https://github.com/RonLevie/RadioUNet), placed at
  `init/Trained_Model_SecondU_s.pt`; or
- any compatible second-level checkpoint via `--init-weights init/<file>.pt`.

Non-PFE keys are loaded strictly (missing or shape-mismatched keys abort
with an error); PFE v2 parameters always start from their zero-initialized
identity mapping.

## 4. Evaluate

After training (`weights/model_best.pt` exists):

```bash
python evaluate.py
```

The four metrics are the acceptance criteria for training quality.
Expected output on the full test set (990 samples) when the model is
trained with the full recipe:

```
point RMSE    : 0.01263   (reference: 0.0126)
held-out RMSE : 0.03178   (reference: 0.0318)
map RMSE      : 0.03157   (reference: 0.0316)
map NMSE      : 0.01829   (reference: 0.0183)
```

## Acknowledgments

This implementation builds on the
[RadioUNet](https://github.com/RonLevie/RadioUNet) codebase and the
[RadioMapSeer](https://ieee-dataport.org/open-access/radiomapseer) dataset
by Levie et al. We thank the authors for releasing their code and data.

## License

Released under [CC BY-NC 4.0](LICENSE) for non-commercial research use.
`modules.py` and `loaders.py` are derived from RadioUNet and keep their
original license. The RadioMapSeer dataset is governed by its own license
and is **not** distributed with this repository.
