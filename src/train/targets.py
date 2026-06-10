"""Ground-truth rasterization for training targets.

From a label's per-spot list (full-resolution x, y, diameter, fluxes) build, at
the model's OUTPUT resolution (stride S):
  - a heatmap target: GT centers splatted as Gaussian bumps (max-merged), ==1 at
    each center cell (CenterNet penalty-reduced-focal target).
  - a positive-weight map: the BOUNDED per-spot SIZE WEIGHT
    ``s_weight = clip(d_ref / d, 1, w_max)`` at each center cell (1 elsewhere).
    INVARIANT: this weight is for the HEATMAP (detection/location) ONLY — it never
    touches the intensity losses (that would be a backdoor curvature prior). See
    docs/decisions/2026-06-10_detector-loss-design.md.
  - per-center arrays for the offset/intensity supervision: integer cell (iy, ix),
    subpixel offset target, and true lipid/protein flux.
"""

import numpy as np


def compute_size_weight(diameter_nm, d_ref=100.0, w_max=5.0):
    """Bounded small-spot detection weight: ``clip(d_ref / d, 1.0, w_max)``.

    The clamp is mandatory — raw ``1/d`` explodes on the 40-80 nm tail. Works on
    scalars or numpy arrays.
    """
    raw = d_ref / np.asarray(diameter_nm, dtype=np.float64)
    return np.clip(raw, 1.0, w_max)


def _draw_gaussian(hmap, cx, cy, sigma):
    """Max-merge a Gaussian bump (peak 1.0) centered at integer cell (cx, cy)."""
    h, w = hmap.shape
    r = int(max(1, round(3.0 * sigma)))
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    if x1 <= x0 or y1 <= y0:
        return
    ys = np.arange(y0, y1)[:, None]
    xs = np.arange(x0, x1)[None, :]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2))
    hmap[y0:y1, x0:x1] = np.maximum(hmap[y0:y1, x0:x1], g)


def build_targets(spots, image_hw, out_stride, heatmap_sigma=1.0,
                  d_ref=100.0, w_max=5.0):
    """Rasterize one image's GT into dense targets + per-center arrays.

    Args:
        spots: list of dicts with 'x','y','diameter_nm','lipid_intensity',
            'protein_intensity' (full-resolution coordinates/fluxes).
        image_hw: (H, W) full-resolution image size.
        out_stride: model output stride S (maps full-res -> H/S x W/S).
        heatmap_sigma: Gaussian-bump sigma in OUTPUT pixels.
        d_ref, w_max: size-weight knobs.

    Returns a dict with:
        'heatmap'        : (1, h, w) float32, GT bumps (==1 at centers).
        'pos_weight_map' : (1, h, w) float32, size weight at centers, 1 elsewhere.
        'iy','ix'        : (N,) int64 center cells.
        'offset'         : (N, 2) float32 subpixel (dx, dy) targets in [0,1).
        'lipid','protein': (N,) float32 true total fluxes.
        'diameter'       : (N,) float32 (carried for diagnostics; NOT used by the
                           intensity loss).
    """
    H, W = image_hw
    h, w = H // out_stride, W // out_stride
    heatmap = np.zeros((h, w), dtype=np.float32)
    pos_weight = np.ones((h, w), dtype=np.float32)

    iy, ix, off, lip, pro, diam = [], [], [], [], [], []
    for s in spots:
        cx_f = s['x'] / out_stride
        cy_f = s['y'] / out_stride
        cx = int(np.clip(np.floor(cx_f), 0, w - 1))
        cy = int(np.clip(np.floor(cy_f), 0, h - 1))
        _draw_gaussian(heatmap, cx, cy, heatmap_sigma)
        sw = float(compute_size_weight(s['diameter_nm'], d_ref, w_max))
        pos_weight[cy, cx] = max(pos_weight[cy, cx], sw)
        iy.append(cy)
        ix.append(cx)
        off.append([cx_f - cx, cy_f - cy])
        lip.append(float(s['lipid_intensity']))
        pro.append(float(s['protein_intensity']))
        diam.append(float(s['diameter_nm']))

    n = len(iy)
    return {
        'heatmap': heatmap[None],
        'pos_weight_map': pos_weight[None],
        'iy': np.array(iy, dtype=np.int64) if n else np.zeros(0, np.int64),
        'ix': np.array(ix, dtype=np.int64) if n else np.zeros(0, np.int64),
        'offset': np.array(off, dtype=np.float32) if n else np.zeros((0, 2), np.float32),
        'lipid': np.array(lip, dtype=np.float32) if n else np.zeros(0, np.float32),
        'protein': np.array(pro, dtype=np.float32) if n else np.zeros(0, np.float32),
        'diameter': np.array(diam, dtype=np.float32) if n else np.zeros(0, np.float32),
    }
