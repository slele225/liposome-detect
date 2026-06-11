"""Calibration: predicted log-space variance exp(logvar) vs actual log-error^2.
The NLL residual is r=log(pred+eps)-log(true+eps) and sigma2=exp(logvar) is the
variance OF THAT LOG RESIDUAL -- so NO delta-method conversion. Compare exp(logvar)
directly to r^2. Calibrated => ratio ~1 and actual_mse rises across deciles."""
import json
from pathlib import Path
import numpy as np, torch, yaml
from src.train.train import build_model
from src.models.decode import decode_image

RUN=Path("runs/hrnet_v1"); MATCH=4.0
SETS=["datasets/alpha_0p50","datasets/alpha_1p00","datasets/alpha_1p50","datasets/alpha_2p00"]
EPS_L=80.15; EPS_P=62.47   # must match the loss eps (same log residual definition)

def main():
    cfg=yaml.safe_load(open("configs/train/hrnet_v1.yaml"))
    dev="cuda" if torch.cuda.is_available() else "cpu"
    model=build_model(cfg).to(dev).eval()
    model.load_state_dict(torch.load(RUN/"best.pt",map_location=dev)["model"])
    dec=cfg["decode"]; nm=np.array(cfg["data"]["norm_mean"],np.float32); ns=np.array(cfg["data"]["norm_std"],np.float32)
    pv_l,err_l,pv_p,err_p=[],[],[],[]
    for vdir in SETS:
        for ip in sorted((Path(vdir)/"images").glob("*.npy")):
            arr=np.load(ip).astype(np.float32)
            spots=json.load(open(Path(vdir)/"labels"/(ip.stem+".json")))["spots"]
            gxy=np.array([[s["x"],s["y"]] for s in spots],np.float32)
            gl=np.array([s["lipid_intensity"] for s in spots],np.float32)
            gp=np.array([s["protein_intensity"] for s in spots],np.float32)
            x=torch.from_numpy((arr-nm[:,None,None])/ns[:,None,None])[None].to(dev)
            with torch.no_grad(): out=model(x)
            out={k:v[0] for k,v in out.items()}
            dets=decode_image(out,model.out_stride,score_threshold=dec["score_threshold"],nms_kernel=dec["nms_kernel"])
            if not dets: continue
            dxy=np.array([[d["x"],d["y"]] for d in dets],np.float32); used=np.zeros(len(dets),bool)
            for i in range(len(spots)):
                dd=np.hypot(dxy[:,0]-gxy[i,0],dxy[:,1]-gxy[i,1]); dd[used]=1e9; j=int(dd.argmin())
                if dd[j]<=MATCH:
                    used[j]=True; d=dets[j]
                    rl=np.log(d["lipid_intensity"]+EPS_L)-np.log(gl[i]+EPS_L)
                    rp=np.log(d["protein_intensity"]+EPS_P)-np.log(gp[i]+EPS_P)
                    pv_l.append(np.exp(d["lipid_intensity_logvar"])); err_l.append(rl*rl)
                    pv_p.append(np.exp(d["protein_intensity_logvar"])); err_p.append(rp*rp)
    for name,pv,err in [("LIPID",pv_l,err_l),("PROTEIN",pv_p,err_p)]:
        pv,err=np.array(pv),np.array(err)
        order=np.argsort(pv); pv,err=pv[order],err[order]
        print(f"\n{name}: predicted var exp(logvar) vs actual log-error^2 (deciles by predicted var)")
        print(f"{'decile':>7} {'pred_var':>10} {'actual_mse':>11} {'ratio':>7}")
        for q in range(10):
            lo,hi=q*len(pv)//10,(q+1)*len(pv)//10
            pm,am=pv[lo:hi].mean(),err[lo:hi].mean()
            print(f"{q+1:>7} {pm:>10.4f} {am:>11.4f} {am/pm if pm>0 else float('nan'):>7.2f}")
        print(f"  overall: mean_pred_var={pv.mean():.4f}  mean_actual_mse={err.mean():.4f}  ratio={err.mean()/pv.mean():.2f}")

if __name__=="__main__": main()
