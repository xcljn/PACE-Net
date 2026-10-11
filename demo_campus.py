# -*- coding: utf-8 -*-
"""
demo_campus.py - PACE-Net real-world campus radio map prediction
=================================================================
Real-scenario demo (3 input channels), mirroring the field-measurement
experiment in the paper. Data handling and encoding are identical to the
training pipeline; the model is simply replaced by PACE-Net
(PointAwareWNetV2 + PFE v2).

Reads [2 images + 1 table] from the input folder:
    image 1: campus building mask 256x256 (white=building / black=road;
             polarity auto-detected)
    image 2: transmitter location map 256x256 (black background + white
             dots; auto-reduced to a single pixel)
    table:   CSV/TXT/XLSX with columns longitude/latitude/SS-RSRP(dBm)
             (column names auto-detected, Chinese headers also accepted);
             mapped to a 256x256 sparse measurement image through the four
             corner coordinates (dBm -> raw gain -> thresh=0.2 transform
             -> x256), fed to the model as the third channel.
Inference uses the trained PACE-Net weights (PFE v2 derives the point mask
internally in the forward pass -- zero post-processing).

Folders (inside this project directory):
    demo_inputs/   put the 2 images + 1 table here (a real campus sample
                   is included: campus_map.png, tx_positions.png,
                   measurements.csv)
    demo_outputs/  results written here:
        measurements.png  sparse measurement image built from the table
                          (check the lon/lat mapping)
        prediction.png    PACE-Net predicted radio map (threshold-inverted)
        overlay.png       semi-transparent buildings + strongest point
        comparison.png    buildings / measurements / prediction triptych
                          (point RMSE and correlation)

Default four corner coordinates (campus in Chengdu, China):
    top-left     103°51'59"E, 30°13'15"N
    top-right    103°52'22"E, 30°13'15"N
    bottom-left  103°51'59"E, 30°12'56"N
    bottom-right 103°52'22"E, 30°12'56"N

Usage:
  python demo_campus.py                     # demo_inputs -> demo_outputs
  python demo_campus.py --input DIR --output DIR --model PT
  python demo_campus.py --corner-tl "103.8664,30.2208"   # decimal degrees
"""
import os
import re
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage import io
from PIL import Image

import pace_net

# ---- Paths (demo_inputs / demo_outputs inside this project directory) ----
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(HERE, "demo_inputs")
DEFAULT_OUT = os.path.join(HERE, "demo_outputs")
DEFAULT_WEIGHTS = os.path.join(HERE, "weights", "model_best.pt")

THRESH = 0.2
IMG_SIZE = (256, 256)
VALID_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
VALID_TBL_EXT = (".csv", ".txt", ".xlsx", ".xls")
DBM_MIN, DBM_MAX = -120.0, -30.0

DEFAULT_CORNERS = {
    "tl": ("103°51'59\"E", "30°13'15\"N"),
    "tr": ("103°52'22\"E", "30°13'15\"N"),
    "bl": ("103°51'59\"E", "30°12'56\"N"),
    "br": ("103°52'22\"E", "30°12'56\"N"),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. Longitude/latitude utilities
# ============================================================
def dms_to_decimal(text):
    """Degree-minute-second or decimal-degree string -> decimal degrees."""
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip()
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
        return float(s)
    m = re.search(r"(\d+)\s*[°º]\s*(\d+)?\s*[′']?\s*(\d+(?:\.\d+)?)?\s*[″\"]?", s)
    if not m:
        raise ValueError(f"Cannot parse coordinate: {text!r}")
    val = (float(m.group(1)) + (float(m.group(2)) if m.group(2) else 0.0) / 60.0
           + (float(m.group(3)) if m.group(3) else 0.0) / 3600.0)
    if re.search(r"[SW]", s, re.IGNORECASE):
        val = -val
    return val


def parse_corner(text):
    """'lon,lat' -> (lon, lat) in decimal degrees."""
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError(f"Corner format must be 'lon,lat', got: {text!r}")
    return dms_to_decimal(parts[0]), dms_to_decimal(parts[1])


def corners_to_bbox(corners):
    """Four corners -> bbox (lon_min/lon_max/lat_min/lat_max)."""
    tl, tr, bl, br = corners["tl"], corners["tr"], corners["bl"], corners["br"]
    return dict(lon_min=min(tl[0], bl[0]), lon_max=max(tr[0], br[0]),
                lat_min=min(bl[1], br[1]), lat_max=max(tl[1], tr[1]))


# ============================================================
# 2. Table reading and automatic column detection
# ============================================================
LON_KEYS = ["longitude", "long", "lon", "lng", "经度"]
LAT_KEYS = ["latitude", "lat", "纬度"]
RSRP_KEYS = ["ss-rsrp", "ss_rsrp", "ssrsrp", "rs-rsrp", "rsrp",
             "signal", "信号强度", "ssb-rsrp"]


def _norm_col(name):
    s = str(name).strip().lower().replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", "", s)


def _find_column(columns, keys):
    normed = {_norm_col(c): c for c in columns}
    for k in keys:
        if k in normed:
            return normed[k]
    for c in columns:
        nc = _norm_col(c)
        for k in keys:
            if k in nc:
                return c
    return None


def read_measurement_table(path):
    """Read the table -> DataFrame(lon, lat, rsrp)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt"):
        df = None
        for enc in ("utf-8-sig", "gbk", "latin-1"):
            try:
                df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError(f"Cannot decode table: {path}")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported table format: {path}")

    df.columns = [str(c).strip() for c in df.columns]
    lon_col = _find_column(df.columns, LON_KEYS)
    lat_col = _find_column(df.columns, LAT_KEYS)
    rsrp_col = _find_column(df.columns, RSRP_KEYS)
    miss = [n for n, c in (("longitude", lon_col), ("latitude", lat_col),
                           ("SS-RSRP", rsrp_col)) if c is None]
    if miss:
        raise ValueError(f"Table is missing columns {miss}; "
                         f"actual header: {list(df.columns)}")

    out = pd.DataFrame({
        "lon": pd.to_numeric(df[lon_col], errors="coerce"),
        "lat": pd.to_numeric(df[lat_col], errors="coerce"),
        "rsrp": pd.to_numeric(df[rsrp_col], errors="coerce"),
    }).dropna()
    print(f"[table] {os.path.basename(path)}: {len(out)} valid records; "
          f"RSRP {out['rsrp'].min():.1f}~{out['rsrp'].max():.1f} dBm")
    return out


def dbm_to_raw_gain(rsrp):
    """dBm -> raw gain [0, 1]."""
    return np.clip((rsrp - DBM_MIN) / (DBM_MAX - DBM_MIN), 0.0, 1.0)


def thresh_transform(g01):
    """thresh=0.2 noise-floor transform and x256 (same as loaders.py)."""
    g = np.clip(g01, THRESH, None)
    return (g - THRESH) / (1.0 - THRESH) * 256.0


def build_measurement_image(table_df, bbox):
    """Table -> 256x256 sparse measurement image (training-compatible)."""
    lon, lat = table_df["lon"].to_numpy(), table_df["lat"].to_numpy()
    rsrp = table_df["rsrp"].to_numpy()
    in_box = ((lon >= bbox["lon_min"]) & (lon <= bbox["lon_max"]) &
              (lat >= bbox["lat_min"]) & (lat <= bbox["lat_max"]))
    lon, lat, rsrp = lon[in_box], lat[in_box], rsrp[in_box]
    n_out = int((~in_box).sum())
    if len(lon) == 0:
        raise ValueError("All measurement points fall outside the corners")

    x = np.round((lon - bbox["lon_min"]) / (bbox["lon_max"] - bbox["lon_min"])
                 * 255).astype(int)
    y = np.round((bbox["lat_max"] - lat) / (bbox["lat_max"] - bbox["lat_min"])
                 * 255).astype(int)
    gain01 = dbm_to_raw_gain(rsrp)
    acc, cnt = np.zeros((256, 256)), np.zeros((256, 256))
    np.add.at(acc, (y, x), gain01)
    np.add.at(cnt, (y, x), 1.0)
    mask = cnt > 0
    mean_g = np.zeros((256, 256))
    mean_g[mask] = acc[mask] / cnt[mask]
    meas = np.round(np.clip(thresh_transform(mean_g), 0, 255)).astype(np.uint8)
    return meas, int(mask.sum()), n_out


def subsample(meas, max_points=300, seed=42):
    """Fixed-seed subsampling when measurement pixels exceed the training
    cap (300)."""
    ys, xs = np.where(meas > 0)
    if max_points <= 0 or len(ys) <= max_points:
        return meas, len(ys)
    rng = np.random.RandomState(seed)
    keep = rng.choice(len(ys), size=max_points, replace=False)
    out = np.zeros_like(meas)
    out[ys[keep], xs[keep]] = meas[ys[keep], xs[keep]]
    return out, max_points


# ============================================================
# 3. Image reading and classification
# ============================================================
def load_gray(path):
    """Read image -> 256x256 grayscale."""
    img = np.asarray(io.imread(path))
    if img.ndim == 3:
        img = img[:, :, 0]
    if img.shape != IMG_SIZE:
        print(f"[WARN] {os.path.basename(path)} has size {img.shape}; resized")
        img = np.asarray(Image.fromarray(img).resize(IMG_SIZE, Image.NEAREST))
    return img


def classify_images(img_files, in_dir):
    """Identify the building map and the transmitter map."""
    imgs = {f: load_gray(os.path.join(in_dir, f)) for f in img_files}
    if len(img_files) < 2:
        raise ValueError("At least 2 images are required (building / Tx)")
    ratio = {f: float((im > 0).sum()) / im.size for f, im in imgs.items()}
    tx_file = min(ratio, key=ratio.get)
    candidates = [f for f in imgs if f != tx_file]
    binary = [f for f in candidates
              if set(np.unique(imgs[f])).issubset({0, 255})]
    if not binary:
        raise ValueError("Cannot identify the building mask "
                         "(expected a binary image with only 0/255)")
    bld_file = max(binary,
                   key=lambda f: min(int((imgs[f] == 0).sum()),
                                     int((imgs[f] == 255).sum())))
    print(f"[images] building <- {bld_file}; transmitter <- {tx_file}")
    return imgs[bld_file], imgs[tx_file]


def reduce_tx(image_Tx):
    """Reduce the antenna map to a single pixel."""
    ys, xs = np.where(image_Tx > 0)
    if len(ys) <= 1:
        return image_Tx, (int(ys[0]), int(xs[0])) if len(ys) else None
    cy, cx = ys.mean(), xs.mean()
    keep = ((ys - cy) ** 2 + (xs - cx) ** 2).argmin()
    out = np.zeros_like(image_Tx)
    out[ys[keep], xs[keep]] = 255
    return out, (int(ys[keep]), int(xs[keep]))


# ============================================================
# 4. Main pipeline
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="PACE-Net campus radio map prediction")
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--output", default=DEFAULT_OUT)
    ap.add_argument("--model", default=DEFAULT_WEIGHTS)
    ap.add_argument("--corner-tl", default=",".join(DEFAULT_CORNERS["tl"]))
    ap.add_argument("--corner-tr", default=",".join(DEFAULT_CORNERS["tr"]))
    ap.add_argument("--corner-bl", default=",".join(DEFAULT_CORNERS["bl"]))
    ap.add_argument("--corner-br", default=",".join(DEFAULT_CORNERS["br"]))
    ap.add_argument("--max-points", type=int, default=300)
    args = ap.parse_args()

    print(f"[device] {device}")
    os.makedirs(args.output, exist_ok=True)

    # 1) Corners + measurement image from the table
    corners = {"tl": parse_corner(args.corner_tl),
               "tr": parse_corner(args.corner_tr),
               "bl": parse_corner(args.corner_bl),
               "br": parse_corner(args.corner_br)}
    bbox = corners_to_bbox(corners)

    tbls = [f for f in os.listdir(args.input)
            if f.lower().endswith(VALID_TBL_EXT)]
    if not tbls:
        raise FileNotFoundError("No measurement table in the input folder")
    table_df = read_measurement_table(os.path.join(args.input, tbls[0]))
    image_meas, n_px, n_out = build_measurement_image(table_df, bbox)
    image_meas, n_pts = subsample(image_meas, args.max_points)
    print(f"[measurements] {n_px} pixels occupied ({n_out} outside bbox), "
          f"{n_pts} points fed to the model")

    # 2) Image classification + polarity
    img_files = [f for f in os.listdir(args.input)
                 if f.lower().endswith(VALID_IMG_EXT)]
    buildings, tx_img = classify_images(img_files, args.input)
    tx_img, tx_pos = reduce_tx(tx_img)
    white_frac = float((buildings == 255).sum()) / buildings.size
    if white_frac > 0.5:
        buildings = 255 - buildings
        print(f"[polarity] building image is {white_frac*100:.1f}% white "
              f"(black=building); inverted")
    else:
        print(f"[polarity] buildings {white_frac*100:.1f}% white "
              f"(white=building); no inversion")

    # 3) Load PACE-Net
    print(f"[model] loading PACE-Net: {args.model}")
    model = pace_net.PointAwareWNetV2(inputs=3, phase="secondU")
    state = torch.load(args.model, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state.get("best_model_state", state["model_state"])
    model.load_state_dict(state)
    model.to(device).eval()

    def run_inference(meas_img):
        """Assemble 3 channels -> PACE-Net inference.
        Returns pred (0~256), raw gain, PNG."""
        inp = np.stack([buildings / 256.0, tx_img / 256.0,
                        meas_img.astype(np.float32)], axis=2).astype(np.float32)
        xt = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            p = model(xt)[1][0, 0].cpu().numpy()   # second level, 0~256
        p_raw = (p / 256.0) * (1 - THRESH) + THRESH  # invert thresh -> raw gain
        p_png = (p_raw * 255).clip(0, 255).astype(np.uint8)
        return p, p_raw, p_png

    # 4) First inference
    pred, pred_raw, pred_png = run_inference(image_meas)
    print(f"[output] range [{pred.min():.2f}, {pred.max():.2f}]")
    peak_y, peak_x = np.unravel_index(np.argmax(pred_png), pred_png.shape)
    peak_val = int(pred_png[peak_y, peak_x])
    coverage = float(np.mean(pred_raw > 0.3)) * 100
    print(f"[strongest point] ({peak_x},{peak_y}) gain {peak_val}/255; "
          f"coverage (>0.3) {coverage:.1f}%")

    # 5) Point accuracy check (on the original predictions + input points,
    #    no modifications of any kind)
    m = image_meas > 0
    meas_raw = image_meas[m].astype(np.float64) / 256.0 * (1 - THRESH) + THRESH
    pred_at = pred_raw[m]
    rmse_g = float(np.sqrt(np.mean((pred_at - meas_raw) ** 2)))
    rmse_dbm = rmse_g * (DBM_MAX - DBM_MIN)
    corr = float(np.corrcoef(pred_at, meas_raw)[0, 1])
    print(f"[point accuracy] n={int(m.sum())}: gain RMSE={rmse_g:.4f} "
          f"(~{rmse_dbm:.2f} dBm), correlation={corr:.4f}")

    # 6) Save outputs
    Image.fromarray(image_meas).save(
        os.path.join(args.output, "measurements.png"))
    Image.fromarray(pred_png).save(
        os.path.join(args.output, "prediction.png"))

    # Overlay
    fig, axo = plt.subplots(1, 1, figsize=(6.2, 6.2))
    axo.imshow(pred_png, cmap="jet", vmin=0, vmax=255)
    ov = np.zeros((256, 256, 4), dtype=np.float32)
    ov[buildings != 0] = [0.2, 0.4, 0.8, 0.35]
    axo.imshow(ov)
    axo.add_patch(Circle((peak_x, peak_y), 8, fill=False,
                         edgecolor="white", lw=2.5))
    axo.add_patch(Circle((peak_x, peak_y), 5, fill=False,
                         edgecolor="#ff4444", lw=2))
    axo.plot(peak_x, peak_y, "o", color="#ff4444", markersize=4)
    axo.set_title(f"PACE-Net prediction overlay\nstrongest point "
                  f"({peak_x},{peak_y}) gain={peak_val}/255", fontsize=12.5)
    axo.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "overlay.png"),
                bbox_inches="tight", dpi=130)
    plt.close()

    # Triptych comparison
    bld_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    bld_rgb[buildings == 0] = [25, 28, 45]
    bld_rgb[buildings != 0] = [70, 120, 190]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 6.3))
    ax[0].imshow(bld_rgb)
    ax[0].set_title("Building map", fontsize=13.5)
    ax[1].imshow(image_meas, cmap="jet", vmin=0, vmax=255)
    ax[1].set_title(f"Sparse measurements ({n_pts} points, from table)",
                    fontsize=13.5)
    ax[2].imshow(pred_png, cmap="jet", vmin=0, vmax=255)
    ax[2].add_patch(Circle((peak_x, peak_y), 7, fill=False,
                           edgecolor="white", lw=2.2))
    ax[2].add_patch(Circle((peak_x, peak_y), 4.5, fill=False,
                           edgecolor="#ff3333", lw=1.8))
    ax[2].set_title(f"PACE-Net prediction\nstrongest ({peak_x},{peak_y})"
                    f"={peak_val}", fontsize=13)
    for a in ax:
        a.axis("off")
    fig.text(0.5, 0.02,
             f"PACE-Net (PFE v2, 3 channels: buildings + Tx + measurements)  |  "
             f"point RMSE ~ {rmse_dbm:.2f} dBm (gain {rmse_g:.4f}), "
             f"correlation {corr:.3f}  |  coverage {coverage:.1f}%",
             ha="center", va="bottom", fontsize=11.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0",
                       edgecolor="#888"))
    plt.subplots_adjust(bottom=0.14)
    plt.savefig(os.path.join(args.output, "comparison.png"),
                bbox_inches="tight", dpi=120)
    plt.close()

    print(f"\n[done] output directory: {args.output}")
    print("  measurements.png / prediction.png / overlay.png / comparison.png")
    # Summary metrics (for the paper's field-experiment table)
    print("\n===== Field experiment summary =====")
    print(f"  Point RMSE ~ {rmse_dbm:.2f} dBm (gain {rmse_g:.4f})")
    print(f"  Correlation = {corr:.4f}")
    print(f"  Prediction max = {pred_raw.max():.3f} (raw gain space)")
    print(f"  Strongest point ({peak_x},{peak_y}), coverage {coverage:.1f}%")


if __name__ == "__main__":
    main()
