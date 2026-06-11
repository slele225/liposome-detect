"""Per-spot-WEIGHTED TLS vs constant-lambda Deming on best.pt, with bootstrap to
measure VARIANCE of recovered alpha. Weights use predicted log-space variances
exp(logvar) DIRECTLY (the NLL residual is already in log space -- no delta-method
conversion). Heteroscedastic weighting should reduce alpha scatter even if the
point estimate matches constant-lambda."""
import json
from pathlib import Path
import numpy as np, torch, yaml
from src.train.train import build_model
from src.models.decode import decode_image

RUN=Path("runs/hrnet_v1"); MATCH=4.0
SETS=[(0.50,"datasets/alpha_0p50"),(1.00,"datasets/alpha_1p00"),
      (1.50,"datasets/alpha_1p50"),(2.00,"datasets/alpha_2p00")]
NBOOT=200

def deming_const(x,y,lam):
    mx,my=x.mean(),y.mean()
    sxx=np.mean((x-mx)**2); syy=np.mean((y-my)**2); sxy=np.mean((x-mx)*(y-my))
    return (syy-lam*sxx+np.sqrt((syy-lam*sxx)**2+4*lam*sxy**2))/(2*sxy)

def wtls(x,y,vx,vy,iters=50):
    b=np.polyfit(x,y,1)[0]
    for _ in range(iters):
        w=1.0/(vy+(b**2)*vx+1e-12); wm=w.sum()
        mx=(w*x).sum()/wm; my=(w*y).sum()/wm
        sxx=(w*(x-mx)**2).sum()/wm; syy=(w*(y-my)**2).sum()/wm; sxy=(w*(x-mx)*(y-my)).sum()/wm
        lam=(w*vy).sum()/((w*vx).sum()+1e-12)
        bn=(syy-lam*sxx+np.sqrt((syy-lam*sxx)**2+4*lam*sxy**2))/(2*sxy)
        if abs(bn-b)<1e-8: b=bn; break
        b=bn
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
                used[j]=True; d=dets[j]
                pl.append(d["lipid_intensity"]); pp.append(d["protein_intensity"])
                plv.append(d["lipid_intensity_logvar"]); ppv.append(d["protein_intensity_logvar"])
    return map(lambda a:np.array(a,np.float64),(pl,pp,plv,ppv))

def main():
    cfg=yaml.safe_load(open("configs/train/hrnet_v1.yaml"))
    dev="cuda" if torch.cuda.is_available() else "cpu"
    model=build_model(cfg).to(dev).eval()
    model.load_state_dict(torch.load(RUN/"best.pt",map_location=dev)["model"])
    rng=np.random.default_rng(0)
    print(f"{'true':>5} | {'const a (mean +/- sd)':>24} | {'wtls a (mean +/- sd)':>24} | {'sd ratio':>8}")
    for ta,vdir in SETS:
        pl,pp,plv,ppv=collect(model,cfg,dev,vdir)
        X=np.log(np.clip(pl,1e-6,None)); Y=np.log(np.clip(pp,1e-6,None))
        vx=np.exp(plv); vy=np.exp(ppv)          # log-space variances, used directly
        n=len(X); ac,aw=[],[]
        lam0=vy.mean()/vx.mean()
        for _ in range(NBOOT):
            idx=rng.integers(0,n,n)
            ac.append(2*deming_const(X[idx],Y[idx],lam0))
            aw.append(2*wtls(X[idx],Y[idx],vx[idx],vy[idx]))
        ac,aw=np.array(ac),np.array(aw)
        sdr=aw.std()/ac.std() if ac.std()>0 else float('nan')
        print(f"{ta:>5.2f} | {ac.mean():>10.3f} +/- {ac.std():<9.4f} | {aw.mean():>10.3f} +/- {aw.std():<9.4f} | {sdr:>8.2f}")

if __name__=="__main__": main()
