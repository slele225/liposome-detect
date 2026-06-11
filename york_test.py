"""Settle whether per-spot weighting CAN beat constant-lambda, using a correct
per-point errors-in-variables fit (York 1968/2004 -- no global lambda).
Three tests:
  (A) synthetic known heteroscedastic noise: correct weighting MUST reduce variance.
  (B) TRUE intensities: clean noise, no model bias -> weighting should help if est ok.
  (C) PREDICTED intensities: real pipeline -> compare to (B) to isolate model bias.
"""
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

def york(x,y,sx2,sy2,iters=100,tol=1e-10):
    """York's best-fit straight line with per-point errors in both x and y.
    sx2, sy2 are per-point variances of x and y. Returns slope b.
    Weights wx=1/sx2, wy=1/sy2; W_i = wx*wy / (wx + b^2*wy) (no correlation term)."""
    wx=1.0/np.maximum(sx2,1e-12); wy=1.0/np.maximum(sy2,1e-12)
    b=np.polyfit(x,y,1)[0]  # init
    for _ in range(iters):
        W=wx*wy/(wx+(b**2)*wy)
        Wsum=W.sum()
        xbar=(W*x).sum()/Wsum; ybar=(W*y).sum()/Wsum
        U=x-xbar; V=y-ybar
        # beta_i per York; with zero x-y error correlation:
        beta=W*(U/wy + (b*V)/wx)
        b_new=(W*beta*V).sum()/(W*beta*U).sum()
        if abs(b_new-b)<tol: b=b_new; break
        b=b_new
    return b

def test_A():
    print("=== TEST A: synthetic known heteroscedastic noise (correct weighting MUST win) ===")
    rng=np.random.default_rng(1)
    true_b=0.75; n=4000
    xt=rng.uniform(0,4,n)                       # true log-lipid
    yt=1.0+true_b*xt                            # true log-protein, slope 0.75
    # heteroscedastic: noise grows with a per-point scale s_i (varies 0.02..0.5)
    s=rng.uniform(0.02,0.5,n)
    sx2=s**2; sy2=(1.3*s)**2                     # y a bit noisier
    x=xt+rng.normal(0,np.sqrt(sx2)); y=yt+rng.normal(0,np.sqrt(sy2))
    ac,aw=[],[]
    lam0=sy2.mean()/sx2.mean()
    for _ in range(NBOOT):
        idx=rng.integers(0,n,n)
        ac.append(deming_const(x[idx],y[idx],lam0))
        aw.append(york(x[idx],y[idx],sx2[idx],sy2[idx]))
    ac,aw=np.array(ac),np.array(aw)
    print(f"  true slope=0.750")
    print(f"  const-lam: {ac.mean():.4f} +/- {ac.std():.4f}")
    print(f"  york wtls: {aw.mean():.4f} +/- {aw.std():.4f}   sd_ratio={aw.std()/ac.std():.2f}")
    print(f"  -> if york sd_ratio < 1 (and mean nearer 0.75), estimator is correct.\n")

def collect(model,cfg,dev,vdir):
    dec=cfg["decode"]; nm=np.array(cfg["data"]["norm_mean"],np.float32); ns=np.array(cfg["data"]["norm_std"],np.float32)
    tl,tp,pl,pp,plv,ppv=[],[],[],[],[],[]
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
                tl.append(gl[i]); tp.append(gp[i])
                pl.append(d["lipid_intensity"]); pp.append(d["protein_intensity"])
                plv.append(d["lipid_intensity_logvar"]); ppv.append(d["protein_intensity_logvar"])
    return map(lambda a:np.array(a,np.float64),(tl,tp,pl,pp,plv,ppv))

def test_BC(model,cfg,dev):
    rng=np.random.default_rng(0)
    # for TRUE intensities we need per-point variances too; use the SAME predicted
    # log-variances as the per-point error estimate (best available); generative
    # noise is roughly lognormal so this is a reasonable proxy.
    print("=== TEST B/C: york WTLS vs const-lambda on TRUE and PRED intensities ===")
    print(f"{'true':>5} | {'TRUE const':>11} {'TRUE york':>11} | {'PRED const':>11} {'PRED york':>11}")
    for ta,vdir in SETS:
        tl,tp,pl,pp,plv,ppv=collect(model,cfg,dev,vdir)
        vx=np.exp(plv); vy=np.exp(ppv)                 # per-point log-space variances
        Xt,Yt=np.log(np.clip(tl,1e-6,None)),np.log(np.clip(tp,1e-6,None))
        Xp,Yp=np.log(np.clip(pl,1e-6,None)),np.log(np.clip(pp,1e-6,None))
        lam0=vy.mean()/vx.mean()
        # point estimates (no bootstrap here; just the central value *2 for alpha)
        a_t_c=2*deming_const(Xt,Yt,lam0); a_t_y=2*york(Xt,Yt,vx,vy)
        a_p_c=2*deming_const(Xp,Yp,lam0); a_p_y=2*york(Xp,Yp,vx,vy)
        print(f"{ta:>5.2f} | {a_t_c:>11.3f} {a_t_y:>11.3f} | {a_p_c:>11.3f} {a_p_y:>11.3f}")
    print("  -> TRUE york nearer truth than TRUE const => estimator helps on clean noise.")
    print("  -> if TRUE york helps but PRED york doesn't => model intensity bias defeats it.\n")

def main():
    test_A()
    cfg=yaml.safe_load(open("configs/train/hrnet_v1.yaml"))
    dev="cuda" if torch.cuda.is_available() else "cpu"
    model=build_model(cfg).to(dev).eval()
    model.load_state_dict(torch.load(RUN/"best.pt",map_location=dev)["model"])
    test_BC(model,cfg,dev)

if __name__=="__main__": main()
