# Weights

The trained checkpoints are attached to the GitHub **Release** of this
repository (each file is ~50 MB, too large for a plain git push).

| File | Place as | Description |
|------|----------|-------------|
| `model_best.pt` | `weights/model_best.pt` | PACE-Net final model (best EMA weights, selected on val). Used by `evaluate.py` and `demo_campus.py`. |
| `init_stage1.pt` | `init/init_stage1.pt` | Starting checkpoint used by the paper's training run (best phase-1 / v1 second level). Only needed to re-run `train.py` with the exact paper recipe. |

Quick download after installing [GitHub CLI](https://cli.github.com/):

```bash
gh release download v1.0.0 --repo <OWNER>/PACE-Net --output .
```

or manually from the Releases page: download the two `.pt` files and drop
them into `weights/` and `init/` respectively.

Alternative starting weights: the official RadioWNet_s_DPM_Thr2
second-level checkpoint (`Trained_Model_SecondU_s.pt`) from the
[RadioUNet repository](https://github.com/Rice-AML/RadioUNet) can be placed
at `init/Trained_Model_SecondU_s.pt` and used via `--baseline-weights`.
