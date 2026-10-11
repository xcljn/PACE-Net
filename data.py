# -*- coding: utf-8 -*-
"""
data.py - Data pipeline shared by training and evaluation.

Wraps the RadioUNet_s_points loader (loaders.py): RadioMapSeer/DPM,
numTx=10, thresh=0.2, deterministic 501/100/99 map split, 10~300 random
sparse points per sample, plus point_mask / heldout_mask.
"""
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import loaders

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0  # Windows-friendly; keeps sampling RNG single-threaded

GAIN_SCALE = 256.0
NUM_TX = 10
THRESH = 0.2
DATA_DIR = os.environ.get(
    "RADIOMAPSEER_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "RadioMapSeer")) + os.sep


def set_seed(seed=20260922):
    """Seed every RNG. Governs the random sparse-point sampling in the
    loaders (the 501/100/99 map split itself is seed-42 deterministic
    inside loaders.py and unaffected by this call)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_bundle(num_tx=NUM_TX, batch_size=15, samples_low=10,
                samples_high=300, data_dir=None):
    """Build train/val/test datasets and DataLoaders with the standard
    protocol. Returns {"datasets": {...}, "loaders": {...}}."""
    common = dict(numTx=num_tx, thresh=THRESH,
                  simulation="DPM", carsSimul="no", carsInput="no",
                  cityMap="complete",
                  dir_dataset=data_dir if data_dir is not None else DATA_DIR,
                  num_samples_low=samples_low,
                  num_samples_high=samples_high)
    ds_train = loaders.RadioUNet_s_points(phase="train", **common)
    ds_val = loaders.RadioUNet_s_points(phase="val", **common)
    ds_test = loaders.RadioUNet_s_points(phase="test", **common)
    loaders_dict = {
        "train": DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                            num_workers=NUM_WORKERS),
        "val": DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                          num_workers=NUM_WORKERS),
        "test": DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                           num_workers=NUM_WORKERS),
    }
    return {"datasets": {"train": ds_train, "val": ds_val, "test": ds_test},
            "loaders": loaders_dict}
