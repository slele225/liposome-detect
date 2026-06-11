"""Constant-lambda Deming vs per-spot-WEIGHTED total least squares, on the same
recovered intensities. Tests whether per-spot uncertainty weighting (the one thing
existing detectors can't do) actually beats a constant noise ratio."""
import json
from pathlib import Path
import numpy as np, torch, yaml
from src.train.train import build_model
from src.models.decode import decode_image

RUN = Path("runs/hrnet_diagnostic"); MATCH = 4.0
SETS = [(0.50,"datasets/alpha_0p50"),(1.00,"datasets/alpha_1p00"),
        (1.50,"datasets/alpha_1p50"),(2.00,"datasets/alpha_2p00")]

def ols(x,y): return np.polyfit(x,y,1)[0]

def deming_const(x,y,lam):
    mx,my=x.mean(),y.mean()
    sxx=np.mean((x-mx)**2); syy=np.mean((y-my)**2); sxy=np.mean((x-mx)*(y-my))
    return (syy-lam*sxx+np.sqrt((syy-lam*sxx)**2+4*lam*sxy**2))/(2*sxy)

def wtls(x,y,vx,vy,iters=50):
    """Per-spot weighted total least squares. Weight_i = 1/(vy_i + b^2 vx_i),
    iterated because the weight depends on the slope b."""
    b = ols(x,y)  # init
    for _ in range(iters):
        w = 1.0/(vy + (b**2)*vx + 1e-12)
        wm = w.sum()
        mx = (w*x).sum()/wm; my = (w*y).sum()/wm
        sxx = (w*(x-mx)**2).sum()/wm
        syy = (w*(y-my)**2).sum()/wm
        sxy = (w*(x-mx)*(y-my)).sum()/wm
        # effective lambda from the weighted axis variances
        lam = (w*vy).sum()/((w*vx).sum()+1e-12)
        b_new = (syy-lam*sxx+np.sqrt((syy-lam*sxx)**2+4*lam*sxy**2))/(2*sxy)
        if abs(b_new-b) < 1e-8: b=b_new; break
        b = b_new
    return b

def collect(model,cfg,dev,vdir):
    dec=cfg["decode"]; nm=np.array(cfg["data"]["norm_mean"],np.float32); ns=np.array(cfg["data"]["norm_std"],np.float32)
    pl,pp,plv,ppv=[],[],[],[]
    for ip in sorted((Path(vdir)/"images").glob("*.npy")):
        arr=np.load(ip).astype(np.float32)
        spots=json.load(open(Path(vdir)/"labels"/(ip.stem+".json")))["spots"]
        gxy=np.array([[s["x"],s["y"]] for s in spots],np.float32)
        x=torch.from_numpy((arr-nm[:,None,None])/ns[:,None,None])[None].to(dev)
        with torch.no_grad(): out=model(x)
        out={k:v[0] for k,v in out.items()}
        dets=decode_image(out,model.out_stride,score_threshold=dec["score_threshold"],nms_kernel=dec["nms_kernel"])
        if not dets: continue
        dxy=np.array([[d["x"],d["y"]] for d in dets],np.float32); used=np.zeros(len(dets),bool)
        for i in range(len(spots)):
            dd=np.hypot(dxy[:,0]-gxy[i,0],dxy[:,1]-gxy[i,1]); dd[used]=1e9; j=int(dd.argmin())
            if dd[j]<=MATCH:
                used[j]=True
                pl.append(dets[j]["lipid_intensity"]); pp.append(dets[j]["protein_intensity"])
                plv.append(dets[j]["lipid_intensity_logvar"]); ppv.append(dets[j]["protein_intensity_logvar"])
    pl,pp,plv,ppv=map(lambda a:np.array(a,np.float64),(pl,pp,plv,ppv))
    return pl,pp,plv,ppv

def main():
    cfg=yaml.safe_load(open("configs/train/hrnet_diagnostic.yaml"))
    dev="cpu"  # CUDA hidden via env; isolated from the training run
    model=build_model(cfg).to(dev).eval()
    model.load_state_dict(torch.load(RUN/"checkpoint.pt",map_location=dev)["model"])
    print(f"{'true':>5} | {'a_OLS':>7} {'a_Dem_const':>12} {'a_WTLS_perspot':>15} | {'n':>7}")
    for ta,d in SETS:
        if not Path(d).exists(): print(f"{ta:>5} missing"); continue
        pl,pp,plv,ppv=collect(model,cfg,dev,d)
        X=np.log(np.clip(pl,1e-6,None)); Y=np.log(np.clip(pp,1e-6,None))
        vx=np.exp(plv)/np.clip(pl,1e-6,None)**2     # log-space var, lipid
        vy=np.exp(ppv)/np.clip(pp,1e-6,None)**2     # log-space var, protein
        a_ols  = 2*ols(X,Y)
        a_dc   = 2*deming_const(X,Y, vy.mean()/vx.mean())
        a_wtls = 2*wtls(X,Y,vx,vy)
        print(f"{ta:>5.2f} | {a_ols:>7.3f} {a_dc:>12.3f} {a_wtls:>15.3f} | {len(pl):>7}")

if __name__=="__main__": main()
