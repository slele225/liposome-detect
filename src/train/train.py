"""Config-driven training harness for the two-channel detector.

    python -m src.train.train --config configs/train/hrnet_v1.yaml [--n-workers N]
    python -m src.train.train --config configs/train/smoke.yaml --smoke

Reproducible from (config + seed + commit): seeds set, provenance written
(``src.provenance.write_provenance``), per-epoch metrics logged. Checkpoint +
resume supported. ``--smoke`` runs a tiny dataset for a couple epochs on CPU-or-GPU,
asserts the loop runs, and emits the decode OUTPUT SCHEMA to a JSON file so the
hard benchmark contract is verified before a real run.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.provenance import write_provenance
from src.models.decode import decode_image, write_detections
from src.train.dataset import SyntheticSpotDataset, collate, targets_to_device
from src.train.engine import (
    EarlyStopping,
    build_scheduler,
    evaluate,
    resolve_early_stopping,
    schedule_weights,
    train_one_epoch,
    weighted_terms,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg):
    m = cfg['model']
    bk = m['backbone']
    btype = bk['type']
    if btype == 'hrnet':
        from src.models.hrnet import HRNetBackbone
        backbone = HRNetBackbone(
            variant=bk.get('variant', 'hrnet_w18_small_v2'),
            out_index=int(bk.get('out_index', 1)),
            in_chans=2, pretrained=bool(bk.get('pretrained', False)))
    elif btype == 'dummy':
        from src.models.dummy import DummyBackbone
        backbone = DummyBackbone(in_chans=2,
                                 out_channels=int(bk.get('out_channels', 16)))
    else:
        raise ValueError(f"unknown backbone type '{btype}'")
    from src.models.interface import DetectorModel
    return DetectorModel(
        backbone, head_hidden=int(m.get('head_hidden', 64)),
        heatmap_prior=float(m.get('heatmap_prior', 0.01)),
        lipid_init_flux=float(m.get('lipid_init_flux', 1000.0)),
        protein_init_flux=float(m.get('protein_init_flux', 1000.0)))


def _split_indices(n, val_fraction, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    return perm[n_val:].tolist(), perm[:n_val].tolist()


def _count_images(root):
    return len(sorted((Path(root) / 'images').glob('img_*.npy')))


def _make_datasets(cfg, out_stride, max_images=None):
    d = cfg['data']
    root = Path(d['dataset'])
    n_total = _count_images(root)
    if n_total == 0:
        raise FileNotFoundError(f"no images under {root/'images'}")
    if max_images is not None:
        n_total = min(n_total, max_images)
    common = dict(out_stride=out_stride,
                  heatmap_sigma=float(cfg['targets']['heatmap_sigma']),
                  d_ref=float(cfg['loss']['size_weight']['d_ref']),
                  w_max=float(cfg['loss']['size_weight']['w_max']),
                  norm_mean=d['norm_mean'], norm_std=d['norm_std'])

    # A SEPARATE val dataset (different seed) is preferred for a real-scale
    # diagnostic: it decouples the val set from the train set instead of carving a
    # fraction out of one dataset. When absent, fall back to the val_fraction split.
    val_dataset = d.get('val_dataset')
    if val_dataset:
        val_root = Path(val_dataset)
        n_val = _count_images(val_root)
        if n_val == 0:
            raise FileNotFoundError(f"no images under {val_root/'images'}")
        if max_images is not None:
            n_val = min(n_val, max(1, max_images))
        train_ds = SyntheticSpotDataset(root, indices=list(range(n_total)), **common)
        val_ds = SyntheticSpotDataset(val_root, indices=list(range(n_val)), **common)
        return train_ds, val_ds

    train_idx, val_idx = _split_indices(
        n_total, float(d.get('val_fraction', 0.2)), int(cfg['seed']))
    train_ds = SyntheticSpotDataset(root, indices=train_idx, **common)
    val_ds = SyntheticSpotDataset(root, indices=val_idx, **common)
    return train_ds, val_ds


def run_training(cfg, config_path, n_workers=0, smoke=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(int(cfg['seed']))

    if smoke:
        s = cfg.get('smoke', {}) or {}
        cfg = dict(cfg)
        cfg['epochs'] = int(s.get('epochs', 2))
        cfg['nll_warmup_epochs'] = int(s.get('nll_warmup_epochs', 1))
        cfg['batch_size'] = int(s.get('batch_size', 2))
        max_images = int(s.get('max_images', 6))
    else:
        max_images = None

    output_dir = Path(cfg.get('output_dir') or f"runs/{cfg['name']}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    out_stride = model.out_stride

    train_ds, val_ds = _make_datasets(cfg, out_stride, max_images=max_images)
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg['batch_size']), shuffle=True,
        num_workers=n_workers, collate_fn=collate, drop_last=False)
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg['batch_size']), shuffle=False,
        num_workers=n_workers, collate_fn=collate)

    opt = torch.optim.AdamW(
        model.parameters(), lr=float(cfg['optim']['lr']),
        weight_decay=float(cfg['optim']['weight_decay']))
    epochs = int(cfg['epochs'])
    total_steps = max(1, epochs * len(train_loader))
    scheduler = build_scheduler(opt, total_steps,
                                float(cfg['optim']['lr_warmup_frac']))

    start_epoch = 0
    ckpt_path = output_dir / 'checkpoint.pt'
    if not smoke and bool(cfg.get('resume', False)) and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optim'])
        scheduler.load_state_dict(ck['scheduler'])
        start_epoch = ck['epoch'] + 1
        print(f"[train] resumed from epoch {start_epoch}")

    print(f"[train] name={cfg['name']} device={device} backbone="
          f"{cfg['model']['backbone']['type']} out_stride={out_stride} "
          f"train={len(train_ds)} val={len(val_ds)} epochs={epochs}")

    nll_warmup = int(cfg['nll_warmup_epochs'])
    # ``early_stop_metric`` selects the val key + direction; the two boundary-
    # consistent options (val_detection_f1, val_intensity_logmse) keep the same
    # meaning across the MSE->NLL switch. Burn-in ignores bad epochs until this
    # many epochs PAST the boundary so a metric artifact there cannot trip it.
    es = resolve_early_stopping(cfg.get('early_stopping', {}), nll_warmup)
    es_enabled = es['enabled']
    es_metric = es['metric']
    stopper = EarlyStopping(patience=es['patience'], min_delta=es['min_delta'],
                            mode=es['mode'], burnin_until=es['burnin_until'])
    best_path = output_dir / 'best.pt'

    metrics_path = output_dir / 'metrics.jsonl'
    static_weights = {k: float(cfg['loss']['weights'][k])
                      for k in ('w_hm', 'w_off', 'w_lip', 'w_pro')}
    t0 = time.time()
    last = {}
    for epoch in range(start_epoch, epochs):
        use_nll = epoch >= nll_warmup
        # Per-epoch emphasis weights (heatmap-led -> intensity-led at the NLL switch).
        weights = schedule_weights(cfg, epoch)
        tr = train_one_epoch(model, train_loader, opt, scheduler, cfg, device,
                             use_nll, weights=weights)
        ev = evaluate(model, val_loader, cfg, device)
        # Per-term WEIGHTED contributions for the diagnostic's term-balance check.
        # TRAIN uses the active per-epoch emphasis weights; VAL uses the STATIC
        # config weights (the fixed yardstick the ramp does not move).
        train_weighted = weighted_terms(tr, weights)
        val_parts = {k: ev[f'val_{k}'] for k in ('heatmap', 'offset', 'lipid',
                                                 'protein')}
        val_weighted = weighted_terms(val_parts, static_weights)
        rec = {'epoch': epoch, 'use_nll': use_nll, 'weights': weights,
               'lr': scheduler.get_last_lr()[0], 'train': tr, 'val': ev,
               'train_weighted': train_weighted, 'val_weighted': val_weighted}

        # Early stopping on a STATIC-weighted val metric (the emphasis ramp reweights
        # TRAIN only, so the yardstick doesn't move). Keep the best-val checkpoint.
        metric = ev.get(es_metric)
        # Skip nan metrics (e.g. logmse before any detection matches) — they would
        # poison the best-so-far comparison.
        metric_ok = metric is not None and metric == metric
        improved, should_stop = (stopper.update(metric, epoch)
                                 if (es_enabled and metric_ok)
                                 else (False, False))
        rec['early_stopping'] = {'metric': es_metric, 'value': metric,
                                 'best': stopper.best, 'best_epoch': stopper.best_epoch,
                                 'num_bad': stopper.num_bad, 'improved': improved}
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        mstr = f"{metric:.4f}" if metric is not None else "n/a"
        print(f"[train] epoch {epoch} nll={use_nll} "
              f"w=(hm={weights['w_hm']:.2f} off={weights['w_off']:.2f} "
              f"lip={weights['w_lip']:.2f} pro={weights['w_pro']:.2f}) "
              f"train_total={tr['total']:.4f} "
              f"(hm={tr['heatmap']:.4f} off={tr['offset']:.4f} "
              f"lip={tr['lipid']:.4f} pro={tr['protein']:.4f}) "
              f"val_f1={ev['f1']:.3f} {es_metric}={mstr} "
              f"best@{stopper.best_epoch} bad={stopper.num_bad}")
        ck = {'epoch': epoch, 'model': model.state_dict(),
              'optim': opt.state_dict(), 'scheduler': scheduler.state_dict(),
              'config': cfg}
        torch.save(ck, ckpt_path)                          # last (for resume)
        if improved:
            torch.save(ck, best_path)                      # best-val (kept on stop)
        last = rec
        if should_stop:
            print(f"[train] early stop at epoch {epoch}: no {es_metric} improvement "
                  f"for {stopper.patience} epochs (best@{stopper.best_epoch}="
                  f"{stopper.best:.4f}) -> {best_path}")
            break

    try:
        write_provenance(output_dir, config_path, name=cfg['name'],
                         epochs=epochs, seed=int(cfg['seed']),
                         dataset=str(cfg['data']['dataset']),
                         backbone=cfg['model']['backbone'].get('type'))
    except Exception as e:
        print(f"  warning: provenance write failed: {e}")

    print(f"[train] DONE in {time.time()-t0:.1f}s -> {output_dir}")

    if smoke:
        _smoke_emit_schema(model, val_loader, cfg, device, output_dir)
    return last


@torch.no_grad()
def _smoke_emit_schema(model, val_loader, cfg, device, output_dir):
    """Decode one val image and write the OUTPUT SCHEMA (verifies the contract)."""
    model.eval()
    images, _targets, _meta = next(iter(val_loader))
    out = model(images.to(device))
    single = {k: v[0].cpu() for k, v in out.items()}
    dec = cfg['decode']
    dets = decode_image(single, model.out_stride,
                        score_threshold=float(dec['score_threshold']),
                        nms_kernel=int(dec['nms_kernel']),
                        max_detections=dec.get('max_detections'))
    # An undertrained smoke model may produce no peaks above threshold; force the
    # real decode->schema path on a few local maxima so the per-detection schema
    # validation actually fires (this is a plumbing check, not an accuracy check).
    if not dets:
        dets = decode_image(single, model.out_stride, score_threshold=0.0,
                            nms_kernel=int(dec['nms_kernel']), max_detections=10)
        print("[train] smoke: no peaks above threshold; forced top local maxima "
              "to exercise the schema")
    path = output_dir / 'smoke_detections.json'
    write_detections(path, dets)            # validates schema keys/types
    print(f"[train] smoke: wrote {len(dets)} detections (schema-validated) -> {path}")


def main():
    ap = argparse.ArgumentParser(description='Train the two-channel detector.')
    ap.add_argument('--config', required=True)
    ap.add_argument('--n-workers', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_training(cfg, args.config, n_workers=args.n_workers, smoke=args.smoke)


if __name__ == '__main__':
    main()
