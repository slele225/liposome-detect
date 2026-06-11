"""Recall + protein-intensity error vs liposome diameter, from a trained ckpt.
Gates the full run: is the ~0.63 F1 uniform, or does it crater on small spots?
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import yaml
from src.train.train import build_model
from src.models.decode import decode_image

RUN = Path("runs/hrnet_diagnostic")
VAL = Path("datasets/diag_val")
MATCH_RADIUS = 4.0
# diameter bins (nm): heavy emphasis on the small tail you care about
BIN_EDGES = [40, 55, 70, 90, 120, 160, 220, 300]

def main():
    cfg = yaml.safe_load(open(RUN / "config_snapshot.yaml")) if (RUN/"config_snapshot.yaml").exists() else None
    if cfg is None:
        # fall back to the training config the run used
        cfg = yaml.safe_load(open("configs/train/hrnet_diagnostic.yaml"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(dev).eval()
    ck = torch.load(RUN / "checkpoint.pt", map_location=dev)
    model.load_state_dict(ck["model"])
    dec = cfg["decode"]
    norm_mean = np.array(cfg["data"]["norm_mean"], dtype=np.float32)
    norm_std  = np.array(cfg["data"]["norm_std"], dtype=np.float32)

    img_paths = sorted((VAL / "images").glob("*.npy"))
    nb = len(BIN_EDGES) - 1
    n_gt   = np.zeros(nb)          # GT spots per bin
    n_hit  = np.zeros(nb)          # matched GT spots per bin
    log_pro_err = [[] for _ in range(nb)]   # |log(pred)-log(true)| protein, per bin

    for ip in img_paths:
        arr = np.load(ip).astype(np.float32)          # [2,H,W] (protein,lipid)
        lab = json.load(open(VAL / "labels" / (ip.stem + ".json")))
        spots = lab["spots"]
        gt_xy = np.array([[s["x"], s["y"]] for s in spots], dtype=np.float32)
        gt_d  = np.array([s["diameter_nm"] for s in spots], dtype=np.float32)
        gt_pro= np.array([s["protein_intensity"] for s in spots], dtype=np.float32)

        x = (arr - norm_mean[:, None, None]) / norm_std[:, None, None]
        x = torch.from_numpy(x)[None].to(dev)
        with torch.no_grad():
            out = model(x)
        out = {k: v[0] for k, v in out.items()}
        dets = decode_image(out, model.out_stride,
                            score_threshold=dec["score_threshold"],
                            nms_kernel=dec["nms_kernel"])
        det_xy = np.array([[d["x"], d["y"]] for d in dets], dtype=np.float32) if dets else np.zeros((0,2),np.float32)
        det_pro= np.array([d["protein_intensity"] for d in dets], dtype=np.float32) if dets else np.zeros((0,),np.float32)

        # greedy match GT -> nearest det within radius
        used = np.zeros(len(dets), dtype=bool)
        for i in range(len(spots)):
            b = np.digitize(gt_d[i], BIN_EDGES) - 1
            if b < 0 or b >= nb:
                continue
            n_gt[b] += 1
            if len(dets) == 0:
                continue
            dd = np.hypot(det_xy[:,0]-gt_xy[i,0], det_xy[:,1]-gt_xy[i,1])
            dd[used] = 1e9
            j = int(dd.argmin())
            if dd[j] <= MATCH_RADIUS:
                used[j] = True
                n_hit[b] += 1
                pt, pp = gt_pro[i], det_pro[j]
                if pt > 0 and pp > 0:
                    log_pro_err[b].append(abs(np.log(pp) - np.log(pt)))

    print(f"{'diam bin (nm)':>16} {'n_gt':>7} {'recall':>7} {'med|log protein err|':>22}")
    for b in range(nb):
        rec = n_hit[b]/n_gt[b] if n_gt[b] else float('nan')
        med = float(np.median(log_pro_err[b])) if log_pro_err[b] else float('nan')
        print(f"{BIN_EDGES[b]:>7}-{BIN_EDGES[b+1]:<7} {int(n_gt[b]):>7} {rec:>7.3f} {med:>22.3f}")
    overall = n_hit.sum()/n_gt.sum()
    print(f"\noverall recall (binned spots): {overall:.3f}")

if __name__ == "__main__":
    main()
