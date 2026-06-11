"""Within-size-bin detection bias + slope checks (true vs predicted, with/without smallest bin)."""
import json
from pathlib import Path
import numpy as np, torch, yaml
from src.train.train import build_model
from src.models.decode import decode_image

RUN, VAL, MATCH_RADIUS = Path("runs/hrnet_diagnostic"), Path("datasets/diag_val"), 4.0
BIN_EDGES = [40, 55, 70, 90, 120, 160, 220, 300]

def main():
    cfg = yaml.safe_load(open("configs/train/hrnet_diagnostic.yaml"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(dev).eval()
    model.load_state_dict(torch.load(RUN/"checkpoint.pt", map_location=dev)["model"])
    dec = cfg["decode"]
    nm = np.array(cfg["data"]["norm_mean"], np.float32); ns = np.array(cfg["data"]["norm_std"], np.float32)
    nb = len(BIN_EDGES)-1
    det_pro = [[] for _ in range(nb)]
    mis_pro = [[] for _ in range(nb)]
    all_lip, all_pro, all_a = [], [], []
    all_plip, all_ppro = [], []

    for ip in sorted((VAL/"images").glob("*.npy")):
        arr = np.load(ip).astype(np.float32)
        spots = json.load(open(VAL/"labels"/(ip.stem+".json")))["spots"]
        gxy = np.array([[s["x"],s["y"]] for s in spots], np.float32)
        gd  = np.array([s["diameter_nm"] for s in spots], np.float32)
        glip= np.array([s["lipid_intensity"] for s in spots], np.float32)
        gpro= np.array([s["protein_intensity"] for s in spots], np.float32)
        x = torch.from_numpy((arr-nm[:,None,None])/ns[:,None,None])[None].to(dev)
        with torch.no_grad(): out = model(x)
        out = {k: v[0] for k,v in out.items()}
        dets = decode_image(out, model.out_stride, score_threshold=dec["score_threshold"], nms_kernel=dec["nms_kernel"])
        dxy = np.array([[d["x"],d["y"]] for d in dets], np.float32) if dets else np.zeros((0,2),np.float32)
        used = np.zeros(len(dets), bool)
        for i in range(len(spots)):
            b = np.digitize(gd[i], BIN_EDGES)-1
            if b<0 or b>=nb: continue
            hit = False
            if len(dets):
                dd = np.hypot(dxy[:,0]-gxy[i,0], dxy[:,1]-gxy[i,1]); dd[used]=1e9
                j = int(dd.argmin())
                if dd[j] <= MATCH_RADIUS:
                    used[j]=True; hit=True
                    all_lip.append(glip[i]); all_pro.append(gpro[i]); all_a.append(gd[i])
                    all_plip.append(dets[j]["lipid_intensity"]); all_ppro.append(dets[j]["protein_intensity"])
            (det_pro if hit else mis_pro)[b].append(gpro[i])

    print(f"{'bin(nm)':>12} {'det_med':>9} {'mis_med':>9} {'ratio':>7}  (>1 = catching bright ones = BIAS)")
    for b in range(nb):
        dm = np.median(det_pro[b]) if det_pro[b] else np.nan
        mm = np.median(mis_pro[b]) if mis_pro[b] else np.nan
        r = dm/mm if (mm and not np.isnan(mm)) else np.nan
        print(f"{BIN_EDGES[b]:>5}-{BIN_EDGES[b+1]:<5} {dm:>9.0f} {mm:>9.0f} {r:>7.3f}")

    L, P, D = np.log(np.array(all_lip)), np.log(np.array(all_pro)), np.array(all_a)
    PL, PP = np.log(np.clip(np.array(all_plip),1e-6,None)), np.log(np.clip(np.array(all_ppro),1e-6,None))
    def slope(lx, ly, mask=None):
        if mask is not None: lx, ly = lx[mask], ly[mask]
        return np.polyfit(lx, ly, 1)[0]
    s_true_all  = slope(L, P)
    s_true_no40 = slope(L, P, D >= 55)
    s_pred_all  = slope(PL, PP)
    s_pred_no40 = slope(PL, PP, D >= 55)
    print(f"\n{'':28} slope   alpha(=2*slope)   (per-spot-random mean alpha ~1.35)")
    print(f"  TRUE intens, all bins:    {s_true_all:6.3f}   {2*s_true_all:6.3f}")
    print(f"  TRUE intens, >=55nm:      {s_true_no40:6.3f}   {2*s_true_no40:6.3f}")
    print(f"  PRED intens, all bins:    {s_pred_all:6.3f}   {2*s_pred_all:6.3f}")
    print(f"  PRED intens, >=55nm:      {s_pred_no40:6.3f}   {2*s_pred_no40:6.3f}")

if __name__ == "__main__": main()
