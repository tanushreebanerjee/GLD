"""Linearity probe for DA3 feature space — does linear interpolation behave?

The gating experiment for GeoFix. Two candidate training formulations hang on
the answer:

  (a) conditioning — keep GLD's noise->clean flow matching, feed artifact
      features as conditioning. Makes no assumption about feature geometry.
  (b) bridge — LBM-style artifact->clean transport, single-step inference.
      Assumes the straight line between an artifact feature and its clean
      counterpart stays on the manifold the decoder understands.

Route (b) is only viable if that assumption holds. This script tests it.

WHAT IS MEASURED
----------------
For a (degraded, clean) pair we encode both, interpolate

    F_t = (1-t) * F_degraded + t * F_clean

and decode each F_t with the released RGB decoder. Two different questions get
asked of the result, and they are not the same question:

  1. `psnr_gt`   — PSNR(decode(F_t), clean_GT). Does quality improve smoothly
                   and monotonically as we walk toward the clean feature? This
                   is the curve the session prompt asks for. A collapse in the
                   middle means the straight line leaves the manifold.

  2. `psnr_lerp` — PSNR(decode(F_t), (1-t)*decode(F_0) + t*decode(F_1)), i.e.
                   how close feature-space interpolation is to pixel-space
                   interpolation. THIS is the one that decides route (b).
                   Bridge matching transports along straight lines in feature
                   space; if those lines are wildly non-linear in image space
                   the transport still works, but if they are *exactly* linear
                   the feature space adds nothing over pixels. Both extremes
                   are informative, so the number is reported rather than
                   thresholded.

Interpolation happens in the POST-`apply_da3_norm` space. That is deliberate:
`apply_da3_norm` applies a LayerNorm to the upper 768 dims, which is nonlinear,
so lerp-then-norm != norm-then-lerp. The normalized space is what `rae.decode()`
consumes and what GLD's DiT operates in, so it is the space a bridge would
actually transport in.

LEVELS
------
DA3-Base exposes L=4 feature levels (blocks 5/7/9/11). The decoder consumes all
four at once, so "interpolate level L" needs a policy for the other three. Two
modes are run:

  joint   — all four levels interpolated together. This is what a bridge would
            actually do end to end, and it is the headline curve.
  level_L — only level L interpolated; the other three are pinned at the
            DEGRADED endpoint. Isolates each level's individual contribution
            and answers whether linearity varies with depth in the backbone.
            Note t=1 here is not the clean reconstruction — it is "level L
            clean, everything else degraded" — so these curves are read for
            their shape, not their endpoint.

THE CONTROL
-----------
Interpolating between two UNRELATED CLEAN images is run as a control. If that
also morphs smoothly then a smooth degraded->clean curve proves nothing: the
decoder would simply be tolerant of any convex combination, and the probe
cannot discriminate. The control is what makes the main result interpretable.

PROXY WARNING
-------------
The real 3DGS pipeline does not exist yet, so degradations here are SYNTHETIC:
heavy Gaussian blur, opaque blobs standing in for floaters, and block
artifacts. Real sparse-view 3DGS artifacts are view-dependent, geometrically
structured, and correlated across views in ways none of these reproduce. Every
conclusion below is provisional on that substitution.

Written for GeoFix; lives in the fork so it can import GLD internals directly,
but it adds a new file and modifies nothing (CLAUDE.md hard rule 9).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

PATCH = 14
LEVELS = 4


# ---------------------------------------------------------------------------
# GLD imports
# ---------------------------------------------------------------------------

def load_gld(gld_root: pathlib.Path):
    src = str(gld_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from utils.da3_validation_metric import apply_da3_norm
    from utils.metrics import compute_psnr
    from utils.model_utils import instantiate_from_config

    return instantiate_from_config, apply_da3_norm, compute_psnr


# ---------------------------------------------------------------------------
# Synthetic degradations — the proxy. See PROXY WARNING in the module docstring.
# ---------------------------------------------------------------------------

def degrade_blur(img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Heavy Gaussian blur. Stands in for the over-smoothing 3DGS shows in
    under-observed regions, where too few views leave Gaussians large and soft."""
    from torchvision.transforms.functional import gaussian_blur

    sigma = float(rng.uniform(4.0, 8.0))
    k = int(2 * round(3 * sigma) + 1)  # odd kernel covering +-3 sigma
    return gaussian_blur(img, kernel_size=[k, k], sigma=[sigma, sigma])


def degrade_floaters(img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Opaque elliptical blobs. Stands in for floaters — spurious Gaussians
    hanging in free space that occlude the true surface.

    Real floaters are semi-transparent, view-dependent and often desaturated;
    these are hard-edged and static. The shape statistics are the part that
    carries over, not the appearance.
    """
    from PIL import Image, ImageDraw

    c, h, w = img.shape
    arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil)

    for _ in range(int(rng.integers(6, 14))):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        rx, ry = rng.integers(10, 45), rng.integers(10, 45)
        # Floaters read as bright haze more often than as dark holes.
        val = int(rng.integers(140, 256))
        colour = (val, val, int(rng.integers(140, 256)))
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=colour)

    out = torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0)
    return out.permute(2, 0, 1).to(img.device)


def degrade_blocks(img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Block artifacts by mean-pooling within a coarse grid and upsampling back.

    Stands in for the piecewise-constant patches that appear when a region is
    covered by too few, too-large Gaussians.
    """
    c, h, w = img.shape
    block = int(rng.choice([14, 21, 28]))
    ph, pw = h // block, w // block
    x = img[:, : ph * block, : pw * block]
    x = x.reshape(c, ph, block, pw, block).mean(dim=(2, 4))
    x = torch.nn.functional.interpolate(
        x[None], size=(ph * block, pw * block), mode="nearest"
    )[0]
    out = img.clone()
    out[:, : ph * block, : pw * block] = x
    return out


DEGRADATIONS = {
    "blur": degrade_blur,
    "floaters": degrade_floaters,
    "blocks": degrade_blocks,
}


# ---------------------------------------------------------------------------
# Encode / decode, mirroring the gate-passing path in geofix.eval_stage1
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_normed(rae, apply_da3_norm, img: torch.Tensor):
    """(3,H,W) in [0,1] -> list of 4 (patches_normed, cls) with batch dim 1.

    Follows `geofix.eval_stage1.reconstruct`, which is the path that reproduced
    GLD's 35.362 dB stage-1 number. The prefix width is derived from the token
    grid rather than assuming a single CLS, so this survives extra register
    tokens and other resolutions.
    """
    x = img[None, None]  # (1,1,3,H,W)
    H, W = img.shape[-2:]
    mean = rae.encoder_mean.to(x.device)
    std = rae.encoder_std.to(x.device)
    raw = rae.encode((x - mean[None]) / std[None], mode="all")

    n_patches = (H // PATCH) * (W // PATCH)
    out = []
    for lvl in range(LEVELS):
        f = raw[lvl]
        if f.ndim == 4:
            f = f.reshape(-1, *f.shape[-2:])
        n, c = f.shape[-2:]
        n_prefix = n - n_patches
        if n_prefix < 0:
            raise ValueError(f"level {lvl}: {n} tokens < {n_patches} patches")
        prefix, patch_tokens = f[:, :n_prefix], f[:, n_prefix:]
        patches = apply_da3_norm(rae, patch_tokens.reshape(1, 1, n_patches, c))
        cls = (prefix[:, 0].reshape(1, 1, c) if n_prefix
               else torch.zeros(1, 1, c, device=f.device, dtype=f.dtype))
        out.append((patches, cls))
    return out


@torch.no_grad()
def decode_feats(rae, feats, H: int, W: int) -> torch.Tensor:
    """list of (patches, cls) -> (3,H,W) in [0,1]."""
    dtype = (next(rae.rae_cl_decoder.parameters()).dtype
             if rae.rae_cl_decoder is not None else torch.float32)
    feats = [(p.to(dtype), c.to(dtype)) for p, c in feats]
    rgb = rae.decode(feats, H, W)["rgb"]
    if rgb.ndim == 5:
        rgb = rgb.reshape(-1, *rgb.shape[-3:])
    return rgb.float()[0]


def lerp_feats(fa, fb, t: float, levels):
    """Interpolate only the levels in `levels`; pin the rest at `fa`."""
    out = []
    for lvl in range(LEVELS):
        pa, ca = fa[lvl]
        pb, cb = fb[lvl]
        if lvl in levels:
            out.append(((1 - t) * pa + t * pb, (1 - t) * ca + t * cb))
        else:
            out.append((pa, ca))
    return out


# ---------------------------------------------------------------------------
# Contact sheets and plots
# ---------------------------------------------------------------------------

def contact_sheet(rows: list[list[torch.Tensor]], path: pathlib.Path,
                  col_labels: list[str], row_labels: list[str]) -> None:
    """rows[i][j] = (3,H,W) in [0,1]. Writes a labelled grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nr, nc = len(rows), len(rows[0])
    fig, axes = plt.subplots(nr, nc, figsize=(2.1 * nc, 2.1 * nr), squeeze=False)
    for i in range(nr):
        for j in range(nc):
            ax = axes[i][j]
            ax.imshow(rows[i][j].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(col_labels[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_curves(results: dict, ts: list[float], out_dir: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, ylabel in [("psnr_gt", "PSNR vs clean GT (dB)"),
                           ("psnr_lerp", "PSNR vs pixel-space lerp (dB)")]:
        degs = sorted(results)
        fig, axes = plt.subplots(1, len(degs), figsize=(4.4 * len(degs), 3.8),
                                 squeeze=False, sharey=True)
        for k, deg in enumerate(degs):
            ax = axes[0][k]
            for mode in ["joint"] + [f"level_{i}" for i in range(LEVELS)]:
                if mode not in results[deg]:
                    continue
                have = [t for t in ts if str(t) in results[deg][mode][metric]]
                y = [results[deg][mode][metric][str(t)]["mean"] for t in have]
                ax.plot(have, y, marker="o",
                        lw=2.4 if mode == "joint" else 1.3,
                        ls="-" if mode == "joint" else "--",
                        color="black" if mode == "joint" else None,
                        label=mode)
            ax.set_title(deg); ax.set_xlabel("t (0=degraded, 1=clean)")
            ax.grid(alpha=0.3)
            if k == 0:
                ax.set_ylabel(ylabel); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"curves_{metric}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gld-root", type=pathlib.Path, required=True)
    ap.add_argument("--model-config", type=pathlib.Path, required=True)
    ap.add_argument("--data-root", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--num-samples", type=int, default=24,
                    help="pairs contributing to the mean curves")
    ap.add_argument("--num-sheet", type=int, default=5,
                    help="samples shown in each contact sheet")
    ap.add_argument("--t-values", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--degradations", default="blur,floaters,blocks,control")
    args = ap.parse_args()

    from omegaconf import OmegaConf
    from PIL import Image

    ts = [float(x) for x in args.t_values.split(",")]
    degs = [d.strip() for d in args.degradations.split(",") if d.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    instantiate_from_config, apply_da3_norm, compute_psnr = load_gld(args.gld_root)

    # --- model ------------------------------------------------------------
    cfg = OmegaConf.load(args.model_config)
    import os
    cwd = os.getcwd()
    os.chdir(args.gld_root)  # model config carries paths relative to the repo
    try:
        rae = instantiate_from_config(cfg.stage_1)
    finally:
        os.chdir(cwd)
    rae = rae.to(device).eval()
    for p in rae.parameters():
        p.requires_grad_(False)
    print(f"[probe] model loaded on {device}", flush=True)

    # --- images -----------------------------------------------------------
    scenes = sorted(p for p in args.data_root.iterdir() if p.is_dir())
    pool = []
    for s in scenes:
        pngs = sorted(s.glob("*.png"))
        if pngs:
            pool.append(pngs[0])  # one view per scene, so samples are independent
    if len(pool) < args.num_samples + 1:
        print(f"[probe] only {len(pool)} scenes available", file=sys.stderr)
        return 2
    rng = np.random.default_rng(args.seed)
    pool = [pool[i] for i in rng.permutation(len(pool))]
    print(f"[probe] {len(pool)} scenes; using {args.num_samples}", flush=True)

    def load(path) -> torch.Tensor:
        im = Image.open(path).convert("RGB")
        a = np.asarray(im).astype(np.float32) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1).to(device)

    modes = {"joint": set(range(LEVELS))}
    for i in range(LEVELS):
        modes[f"level_{i}"] = {i}

    results = {d: {m: {"psnr_gt": {}, "psnr_lerp": {}} for m in modes} for d in degs}
    acc = {d: {m: {"psnr_gt": {t: [] for t in ts}, "psnr_lerp": {t: [] for t in ts}}
               for m in modes} for d in degs}
    sheets = {d: [] for d in degs}

    for si in range(args.num_samples):
        clean = load(pool[si])
        H, W = clean.shape[-2:]
        drng = np.random.default_rng(args.seed * 100003 + si)

        for deg in degs:
            if deg == "control":
                # Unrelated clean image — the discriminative control.
                src = load(pool[(si + args.num_samples) % len(pool)])
            else:
                src = DEGRADATIONS[deg](clean, drng).clamp(0, 1)

            f_src = encode_normed(rae, apply_da3_norm, src)
            f_cln = encode_normed(rae, apply_da3_norm, clean)

            # Endpoints for the pixel-space lerp reference, per mode.
            for mode, lvls in modes.items():
                dec = {}
                for t in ts:
                    dec[t] = decode_feats(rae, lerp_feats(f_src, f_cln, t, lvls), H, W)

                t0, t1 = min(ts), max(ts)
                d0, d1 = dec[t0], dec[t1]
                for t in ts:
                    with torch.no_grad():
                        gt_p = compute_psnr(clean[None], dec[t][None]).item()
                    acc[deg][mode]["psnr_gt"][t].append(gt_p)
                    # At the endpoints the pixel-space reference IS the decode,
                    # so PSNR is +inf by construction and carries no
                    # information. Recording it would poison the mean.
                    if t in (t0, t1):
                        continue
                    with torch.no_grad():
                        ref = (1 - t) * d0 + t * d1
                        lp_p = compute_psnr(ref[None], dec[t][None]).item()
                    acc[deg][mode]["psnr_lerp"][t].append(lp_p)

                if mode == "joint" and si < args.num_sheet:
                    sheets[deg].append([src.cpu()] + [dec[t].cpu() for t in ts]
                                       + [clean.cpu()])

            del f_src, f_cln
            torch.cuda.empty_cache()

        print(f"[probe] sample {si + 1}/{args.num_samples}", flush=True)

    # --- aggregate --------------------------------------------------------
    for deg in degs:
        for mode in modes:
            for metric in ("psnr_gt", "psnr_lerp"):
                for t in ts:
                    v = np.array(acc[deg][mode][metric][t], dtype=np.float64)
                    if v.size == 0:  # psnr_lerp at the degenerate endpoints
                        continue
                    results[deg][mode][metric][str(t)] = {
                        "mean": float(v.mean()),
                        "std": float(v.std()),
                        "n": int(v.size),
                    }

    for deg in degs:
        if sheets[deg]:
            contact_sheet(
                sheets[deg], args.out_dir / f"sheet_{deg}.png",
                col_labels=(["degraded" if deg != "control" else "other image"]
                            + [f"t={t:g}" for t in ts] + ["clean GT"]),
                row_labels=[f"s{i}" for i in range(len(sheets[deg]))],
            )
    plot_curves(results, ts, args.out_dir)

    summary = {
        "results": results,
        "t_values": ts,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "data_root": str(args.data_root),
        "model_config": str(args.model_config),
        "image_hw": [H, W],
        "interp_space": "post-apply_da3_norm (LayerNorm'd), the space rae.decode consumes",
        "note": "SYNTHETIC degradation proxy; real 3DGS artifacts may differ.",
    }
    (args.out_dir / "linearity_probe.json").write_text(json.dumps(summary, indent=2))

    print("\n=== linearity probe: mean PSNR vs clean GT (dB) ===")
    for deg in degs:
        print(f"\n[{deg}]")
        print("  mode      " + "".join(f"  t={t:<6g}" for t in ts))
        for mode in ["joint"] + [f"level_{i}" for i in range(LEVELS)]:
            row = "".join(f"  {results[deg][mode]['psnr_gt'][str(t)]['mean']:<8.2f}"
                          for t in ts)
            print(f"  {mode:<10}{row}")
    print("\n=== mean PSNR vs pixel-space lerp (dB) — high = feature lerp ~ pixel lerp ===")
    for deg in degs:
        print(f"\n[{deg}]")
        for mode in ["joint"]:
            row = "".join(
                f"  t={t:g}:{results[deg][mode]['psnr_lerp'][str(t)]['mean']:<7.2f}"
                for t in ts if str(t) in results[deg][mode]['psnr_lerp'])
            print(f"  {mode:<10}{row}")
    print(f"\n[probe] wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
