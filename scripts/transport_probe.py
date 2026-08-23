"""Does any mask predict how far a token has to MOVE in level-1 feature space?

This is the pre-test for latent bridge matching (LBM), and it is deliberately
cheaper than training one: it runs the RAE ENCODER only -- no diffusion, no
cascade, no sampler -- so the whole frozen holdout costs minutes rather than a
finetune.

## The two questions, and why they are separate

**Q1. Is a bridge shorter than a noise start?**  GLD's stage 2 transports
`x_1 = eps` to `x_0 = F_clean`. A bridge would transport `x_1 = F_artifact`
instead. That is only worth doing if the artifact features are meaningfully
CLOSER to clean than noise is, so the probe reports

    transport_ratio = ||F_clean - F_artifact|| / ||F_clean - eps||

per token, in the SAME normalised space the sampler operates in (see
`encode_artifact` -- the stats are load-bearing, and comparing a normalised
feature to a raw one is the failure mode this project keeps catching). A ratio
near 1 means the bridge saves nothing and LBM is a reparameterisation with no
transport advantage; a ratio near 0 means the artifact render is nearly the
answer already.

**Q2. Could a mask MODULATE that bridge?**  A mask-modulated bridge sets a
per-token transport schedule -- travel far here, stay put there. For that to be
worth anything two things must hold, and the probe measures both:

  (a) the per-token transport distance `d = ||F_clean - F_artifact||` must be
      HETEROGENEOUS within a frame. If every token has to move about equally
      there is nothing to modulate, whatever the mask says. Reported as the
      coefficient of variation, the p90/p10 ratio, and the share of total
      transport carried by the top quartile of tokens.

  (b) some mask must RANK tokens by `d`. Reported as Spearman rho and as the
      AUROC of the mask discriminating top-quartile-transport tokens, which puts
      it on session 6's scale (0.501 chance, 0.929 oracle).

Q2 can only be answered favourably if Q1 leaves room, but the reverse does not
hold, so both are reported unconditionally and neither gates the other.

## What is NOT measured here

This does not say a bridge would train better, or that a mask-modulated bridge
would beat conditioning. It says whether the PREMISE of one is true on our data.
`CLAUDE.md` set bridge matching aside for "absence of positive evidence"; this
probe is the cheapest thing that could supply some, or rule it out.

## The controls, and why each is here

- `rand`      per-token uniform noise. Chance. Spearman ~0, AUROC ~0.5.
- `art_norm`  `||F_artifact||` itself. The TRIVIAL predictor: if a mask ranks
              transport no better than the feature magnitude already lying
              around, the mask carries nothing a bridge could not read off its
              own input. This is the bar that matters, not chance.
- `oracle_abs` / `oracle_lpips`  the GT-derived planes. Ceilings, not methods.

A spatially UNIFORM mask needs no control row: Spearman and AUROC are invariant
to area by construction, so unlike every PSNR comparison in this project there
is no area confound to remove. That is the reason to score ranking here rather
than a blend delta.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

#: Planes read straight from the `.edit1.npz` pack. GeoFixPairs refuses any type
#: the MANIFEST does not list (deliberately -- a config silently widening the
#: mask stack caused a real bug on 2026-08-19), and the holdout manifest lists
#: only `oracle_lpips`. The probe reads no mask into the model, so it loads the
#: pack itself rather than relaxing that guard.
MASK_PLANES = ("oracle_lpips", "oracle_abs", "depth_disagree",
               "depth_disagree_disp", "sam_object", "sam_boundary", "opacity")
#: `fisher` is (3, 504, 504) -- the three gamma_c bands. Session 6.5 settled on
#: g0 (gamma_c = 0.001) at level 1, so band 0 is the one that would be deployed;
#: all three are scored because they are free once the pack is open.
FISHER_BANDS = 3


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared. Spearman is Pearson on these."""
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    # average tied ranks so a constant plane scores 0 correlation rather than an
    # arbitrary one -- `null_empty`, `opacity` and any saturated mask hit this.
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUROC of `score` against a boolean label. 0.5 if degenerate."""
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    r = _ranks(score) + 1.0
    return float((r[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def load_mask_planes(pack: pathlib.Path, grid: int) -> dict[str, np.ndarray]:
    """`.edit1.npz` -> `{name: (grid, grid) float}`, MAX-pooled (hard rule 6)."""
    from video.geofix_pairs import pool_mask

    z = np.load(pack, allow_pickle=True)
    pol = str(z["polarity"])
    if pol != "edit1":
        raise ValueError(f"{pack}: polarity {pol!r}, not 'edit1' (hard rule 7).")
    out: dict[str, np.ndarray] = {}
    for name in MASK_PLANES:
        if name in z.files:
            out[name] = pool_mask(z[name], grid).numpy()[0]
    if "fisher" in z.files:
        pooled = pool_mask(z["fisher"], grid).numpy()
        for b in range(min(FISHER_BANDS, pooled.shape[0])):
            out[f"fisher_g{b}"] = pooled[b]
    return out


def probe(args) -> int:
    device = torch.device(args.device)
    from latent_blending import build_context, encode_artifact
    from video.geofix_pairs import GeoFixPairs, assert_view_config
    from geofix_infer import build_batch

    manifest = json.loads(pathlib.Path(args.manifest).read_text())
    ctx = build_context(argparse.Namespace(
        eval_config=args.eval_config,
        model_config_level1=args.model_config_level1,
        model_config_cascade=args.model_config_cascade,
        checkpoint_level1=args.checkpoint_level1,
        checkpoint_cascade=args.checkpoint_cascade,
        stats_dir=args.stats_dir,
        num_views=args.num_views,
        cond_num=args.cond_num,
        image_size=args.image_size,
        cfg_scale=1.5,
        cfg_scale_cascade=1.5,
        clean_refs=None,
    ), device)
    v, cond = ctx["num_views"], ctx["cond_num"]
    assert_view_config(ctx["ref_view_sampling"], cond, manifest)
    grid = ctx["token_grid"]
    rae, stat_path = ctx["rae"], ctx["stat_path"]

    dataset = GeoFixPairs(
        args.manifest, mask_types=list(manifest["mask_types"]),
        token_grid=grid, gamma=1.0, pooling="max", return_gt=True)
    n = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    idx = list(range(n))
    if args.num_shards > 1:
        idx = idx[args.shard::args.num_shards]

    artifact_root = pathlib.Path(manifest["artifact_root"])
    rows = []
    for count, i in enumerate(idx):
        s = dataset.samples[i]
        split, stems = s["split"], list(s["targets"])
        batch = build_batch(dataset[i], device)
        lo, hi = float(batch["image"].min()), float(batch["image"].max())
        if lo < -0.01 or hi > 1.01:
            raise ValueError(
                f"batch['image'] range [{lo:.3f}, {hi:.3f}] is not [0,1]; "
                "build_batch must undo the loader's ImageNet normalisation.")

        with torch.no_grad():
            f_art = encode_artifact(rae, batch["image"], args.level, stat_path, device)
            f_cln = encode_artifact(rae, batch["gt"], args.level, stat_path, device)
        if f_art.shape[-2:] != (grid, grid):
            raise ValueError(
                f"level-{args.level} features are {tuple(f_art.shape)}, whose grid "
                f"is not {grid}x{grid}. The mask planes pool to {grid}; correlating "
                "across two grids would silently compare different tokens.")

        # eps in the SAME normalised space. Seeded off the global sample index so
        # a resumed or resharded run draws the identical noise -- `transport_ratio`
        # is a ratio of two norms and a different eps moves its denominator.
        g = torch.Generator(device="cpu").manual_seed(args.seed + i)
        eps = torch.randn(f_cln.shape, generator=g).to(device=device, dtype=f_cln.dtype)

        d_all = torch.linalg.vector_norm(f_cln - f_art, dim=1)       # (V, g, g)
        n_all = torch.linalg.vector_norm(f_cln - eps, dim=1)
        a_all = torch.linalg.vector_norm(f_art, dim=1)

        for k, stem in enumerate(stems):
            view = cond + k
            d = d_all[view].float().cpu().numpy().reshape(-1)
            nz = n_all[view].float().cpu().numpy().reshape(-1)
            an = a_all[view].float().cpu().numpy().reshape(-1)

            planes = load_mask_planes(
                artifact_root / split / "masks" / f"{stem}.edit1.npz", grid)
            rng = np.random.default_rng(args.seed + 1000 * i + k)
            planes["rand"] = rng.random(grid * grid).reshape(grid, grid)
            planes["art_norm"] = an.reshape(grid, grid)

            q75 = float(np.quantile(d, 0.75))
            hot = d > q75
            order = np.sort(d)[::-1]
            top25 = int(round(0.25 * len(order)))

            row = {
                "split": split, "scene": s["scene"], "stem": stem,
                "level": args.level,
                # Q1
                "transport_ratio_mean": float(d.mean() / nz.mean()),
                "transport_ratio_median_tok": float(np.median(d / np.maximum(nz, 1e-12))),
                "d_mean": float(d.mean()), "noise_mean": float(nz.mean()),
                # Q2a -- heterogeneity of what a bridge would have to schedule
                "d_cv": float(d.std() / max(d.mean(), 1e-12)),
                "d_p90_p10": float(np.quantile(d, 0.9) / max(np.quantile(d, 0.1), 1e-12)),
                "d_top25_share": float(order[:top25].sum() / max(order.sum(), 1e-12)),
                # Q2b -- can any mask rank it
                "rho": {}, "auroc": {}, "area": {},
            }
            for name, plane in sorted(planes.items()):
                p = plane.reshape(-1).astype(np.float64)
                row["rho"][name] = spearman(p, d.astype(np.float64))
                row["auroc"][name] = auroc(p, hot)
                row["area"][name] = float(p.mean())
            rows.append(row)

            # Hard rule 10 wants a picture beside the numbers, and for a ranking
            # probe the picture is the transport map itself next to the masks that
            # are supposed to predict it. Dumped only when asked, for a strided
            # subset -- the full set would be 776 npz files nothing reads.
            if args.dump_maps:
                md = pathlib.Path(args.dump_maps)
                md.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    (md / f"{split.replace('/', '__')}__{stem}.npz").open("wb"),
                    d=d.reshape(grid, grid).astype(np.float32),
                    render=(batch["image"][0, view].permute(1, 2, 0)
                            .float().cpu().numpy() * 255).astype(np.uint8),
                    gt=(batch["gt"][0, view].permute(1, 2, 0)
                        .float().cpu().numpy() * 255).astype(np.uint8),
                    **{f"m_{k}": v.astype(np.float32) for k, v in planes.items()})

        if count % 10 == 0:
            print(f"[probe] {count + 1}/{len(idx)} samples", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps({
        "manifest": str(args.manifest), "level": args.level,
        "checkpoint_level1": str(args.checkpoint_level1),
        "stats_dir": str(args.stats_dir),
        "shard": args.shard, "num_shards": args.num_shards,
        "n_views": len(rows), "seed": args.seed,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "rows": rows,
    }, indent=1))
    tmp.replace(out)
    print(f"[probe] wrote {len(rows)} view rows -> {out}", flush=True)
    return 0


def _paired(per_scene: dict[str, list[float]]) -> dict:
    """Scene means -> mean, t against 0, and the count of scenes above 0.

    Paired PER SCENE because per-scene variance on this data is 3.71 dB against
    the ~1.00 dB refinement is worth; the same argument applies to a correlation.
    """
    vals = np.array([float(np.mean(v)) for _, v in sorted(per_scene.items())])
    m = float(vals.mean())
    sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    t = m / (sd / np.sqrt(len(vals))) if sd > 0 else float("nan")
    return {"mean": m, "t": float(t), "n_scenes": int(len(vals)),
            "wins": int((vals > 0).sum())}


def report(args) -> int:
    rows = []
    for f in sorted(pathlib.Path(p) for p in args.rows):
        rows.extend(json.loads(f.read_text())["rows"])
    if not rows:
        raise SystemExit("no rows")
    scenes = sorted({r["scene"] for r in rows})

    def by_scene(fn):
        d: dict[str, list[float]] = {}
        for r in rows:
            d.setdefault(r["scene"], []).append(fn(r))
        return d

    out = {
        "n_views": len(rows), "n_scenes": len(scenes), "level": rows[0]["level"],
        "transport": {k: _paired(by_scene(lambda r, k=k: r[k])) for k in
                      ("transport_ratio_mean", "transport_ratio_median_tok",
                       "d_mean", "noise_mean", "d_cv", "d_p90_p10",
                       "d_top25_share")},
        "rho": {}, "auroc": {}, "area": {},
    }
    names = sorted(rows[0]["rho"])
    for name in names:
        out["rho"][name] = _paired(by_scene(lambda r, n=name: r["rho"][n]))
        out["auroc"][name] = _paired(by_scene(lambda r, n=name: r["auroc"][n] - 0.5))
        out["auroc"][name]["mean_auroc"] = out["auroc"][name]["mean"] + 0.5
        out["area"][name] = _paired(by_scene(lambda r, n=name: r["area"][n]))

    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))

    print(f"\n{len(rows)} target views, {len(scenes)} scenes, level {out['level']}\n")
    print("Q1  is a bridge shorter than a noise start?")
    tr = out["transport"]["transport_ratio_mean"]
    print(f"    ||F_clean-F_art|| / ||F_clean-eps||   {tr['mean']:.4f}   "
          f"(per-token median {out['transport']['transport_ratio_median_tok']['mean']:.4f})")
    print(f"    mean d {out['transport']['d_mean']['mean']:.4f}   "
          f"mean noise distance {out['transport']['noise_mean']['mean']:.4f}")
    print("\nQ2a is per-token transport heterogeneous enough to modulate?")
    for k in ("d_cv", "d_p90_p10", "d_top25_share"):
        print(f"    {k:16s} {out['transport'][k]['mean']:.4f}")
    print("\nQ2b can any mask rank it?   (paired per scene, t vs 0, wins/20)")
    print(f"    {'mask':22s} {'rho':>8s} {'t':>8s} {'win':>5s} "
          f"{'AUROC':>7s} {'t':>8s} {'win':>5s} {'area':>6s}")
    for name in sorted(names, key=lambda n: -out["rho"][n]["mean"]):
        r, a = out["rho"][name], out["auroc"][name]
        print(f"    {name:22s} {r['mean']:8.4f} {r['t']:8.2f} {r['wins']:3d}/{r['n_scenes']:<2d} "
              f"{a['mean_auroc']:7.4f} {a['t']:8.2f} {a['wins']:3d}/{a['n_scenes']:<2d} "
              f"{out['area'][name]['mean']:6.3f}")
    print(f"\nwrote {p}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("probe", "report"), default="probe")
    ap.add_argument("--manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", nargs="*", default=[])
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--eval-config")
    ap.add_argument("--model-config-level1")
    ap.add_argument("--model-config-cascade")
    ap.add_argument("--checkpoint-level1")
    ap.add_argument("--checkpoint-cascade")
    ap.add_argument("--stats-dir")
    ap.add_argument("--num-views", type=int, default=8)
    ap.add_argument("--cond-num", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=504)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--dump-maps", default=None,
                    help="Directory for per-view npz (transport map, render, GT, "
                         "every mask plane) so a panel can be drawn. Use with "
                         "--num-shards 32 --shard 0 for a strided 7-scene subset.")
    args = ap.parse_args()
    return report(args) if args.mode == "report" else probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
