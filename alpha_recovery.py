"""Fixed-alpha recovery with OLS vs errors-in-variables (Deming) fits.
Both axes are noisy (lipid: PMT noise; protein: PMT + eta heterogeneity), so OLS
attenuates the slope. Deming accounts for noise in both; for a_PRED we weight by
the model's predicted per-spot log-space variances."""
import json
from pathlib import Path
import numpy as np, torch, yaml
from src.train.train import build_model
from src.models.decode import decode_image

RUN = Path("runs/hrnet_v1"); MATCH_RADIUS = 4.0
TEST_SETS = [(0.50,"datasets/alpha_0p50"),(1.00,"datasets/alpha_1p00"),
             (1.50,"datasets/alpha_1p50"),(2.00,"datasets/alpha_2p00")]
ETA_LOGVAR = 0.1**2   # generative eta: lognormal(0,0.1) -> log-space var ~0.01

def ols(lx, ly): return np.polyfit(lx, ly, 1)[0]

def deming(lx, ly, lam):
    """Deming slope; lam = var(y)/var(x). lam=1 is total least squares."""
    mx, my = lx.mean(), ly.mean()
    sxx = np.mean((lx-mx)**2); syy = np.mean((ly-my)**2); sxy = np.mean((lx-mx)*(ly-my))
    return (syy - lam*sxx + np.sqrt((syy - lam*sxx)**2 + 4*lam*sxy**2)) / (2*sxy)

def recover(model, cfg, dev, val_dir):
    dec = cfg["decode"]
    nm=np.array(cfg["data"]["norm_mean"],np.float32); ns=np.array(cfg["data"]["norm_std"],np.float32)
    tl,tp,pl,pp,plv,ppv = [],[],[],[],[],[]
    for ip in sorted((Path(val_dir)/"images").glob("*.npy")):
        arr=np.load(ip).astype(np.float32)
        spots=json.load(open(Path(val_dir)/"labels"/(ip.stem+".json")))["spots"]
        gxy=np.array([[s["x"],s["y"]] for s in spots],np.float32)
        glip=np.array([s["lipid_intensity"] for s in spots],np.float32)
        gpro=np.array([s["protein_intensity"] for s in spots],np.float32)
        x=torch.from_numpy((arr-nm[:,None,None])/ns[:,None,None])[None].to(dev)
        with torch.no_grad(): out=model(x)
        out={k:v[0] for k,v in out.items()}
        dets=decode_image(out,model.out_stride,score_threshold=dec["score_threshold"],nms_kernel=dec["nms_kernel"])
        if not dets: continue
        dxy=np.array([[d["x"],d["y"]] for d in dets],np.float32); used=np.zeros(len(dets),bool)
        for i in range(len(spots)):
            dd=np.hypot(dxy[:,0]-gxy[i,0],dxy[:,1]-gxy[i,1]); dd[used]=1e9; j=int(dd.argmin())
            if dd[j]<=MATCH_RADIUS:
                used[j]=True
                tl.append(glip[i]); tp.append(gpro[i])
                pl.append(dets[j]["lipid_intensity"]); pp.append(dets[j]["protein_intensity"])
                plv.append(dets[j]["lipid_intensity_logvar"]); ppv.append(dets[j]["protein_intensity_logvar"])
    tl,tp,pl,pp=map(lambda a:np.array(a,np.float64),(tl,tp,pl,pp))
    plv,ppv=map(lambda a:np.array(a,np.float64),(plv,ppv))
    Lt,Pt=np.log(tl),np.log(tp)
    Lp,Pp=np.log(np.clip(pl,1e-6,None)),np.log(np.clip(pp,1e-6,None))
    # TRUE: lam from generative noise. x=log-lipid (PMT only), y=log-protein (PMT+eta).
    # crude lam: (var of residual proxy). Use lam slightly >1 due to eta; report a few.
    a_true_ols   = 2*ols(Lt,Pt)
    a_true_dem1  = 2*deming(Lt,Pt,1.0)        # TLS
    a_true_dem2  = 2*deming(Lt,Pt,2.0)        # y noisier (eta)
    # PRED: lam per the model's mean predicted log-space variances (delta method)
    lvar_x=np.mean(np.exp(plv)/np.clip(pl,1e-6,None)**2)
    lvar_y=np.mean(np.exp(ppv)/np.clip(pp,1e-6,None)**2)
    lam_pred=lvar_y/lvar_x if lvar_x>0 else 1.0
    a_pred_ols   = 2*ols(Lp,Pp)
    a_pred_dem   = 2*deming(Lp,Pp,lam_pred)
    return a_true_ols,a_true_dem1,a_true_dem2,a_pred_ols,a_pred_dem,lam_pred,len(tl)

def main():
    cfg=yaml.safe_load(open("configs/train/hrnet_diagnostic.yaml"))
    dev="cuda" if torch.cuda.is_available() else "cpu"
    model=build_model(cfg).to(dev).eval()
    model.load_state_dict(torch.load(RUN/"best.pt",map_location=dev)["model"])
    print(f"{'true':>5} | {'TRUE_ols':>8} {'TRUE_tls':>8} {'TRUE_dem2':>9} | {'PRED_ols':>8} {'PRED_dem':>8} {'lam_p':>6} | {'n':>7}")
    for ta,d in TEST_SETS:
        if not Path(d).exists(): print(f"{ta:>5} missing"); continue
        r=recover(model,cfg,dev,d)
        print(f"{ta:>5.2f} | {r[0]:>8.3f} {r[1]:>8.3f} {r[2]:>9.3f} | {r[3]:>8.3f} {r[4]:>8.3f} {r[5]:>6.2f} | {r[6]:>7}")

if __name__=="__main__": main()
