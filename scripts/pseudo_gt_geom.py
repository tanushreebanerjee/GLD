#!/usr/bin/env python
"""Build the PSEUDO-GT geometry reference for `geofix.depth_eval --reference pseudo_gt`.

    PYTHONPATH=.:src python scripts/pseudo_gt_geom.py \
        --manifest <geofix>/eval/train/manifest_holdout_it03000.json \
        --eval-config <geofix>/configs/eval/blend_6_5.yaml \
        --model-config-level1  configs/training/DA3_geofix_level1_depth.yaml \
        --model-config-cascade <geofix>/third_party/gld/configs/training/DA3_cascade.yaml \
        --checkpoint-level1  <geofix>/third_party/gld/checkpoints/da3_level1.pt \
        --checkpoint-cascade <geofix>/third_party/gld/checkpoints/da3_cascade.pt \
        --stats-dir <geofix>/eval/feature_stats/legs/dl3dv_art_504__imagenet \
        --out <geofix>/eval/session8/pseudo_gt_geom

## What this produces, and what it is NOT

For every TARGET view of the holdout manifest it writes
`<out>/<split>/<frame>.geom.npz` with exactly the key layout
`geofix_infer.py --dump-depth` writes -- `depth (504,504)`,
`depth_conf (504,504)`, `ray (288,288,6)`, `ray_conf (288,288)` -- so
`depth_eval --pseudo-gt-root <out>` reads it with the same `load_geom`.

The content is the CLEAN GROUND-TRUTH PHOTOGRAPH encoded by the same DA3 encoder
at all four levels and decoded through the same RAE and the same DPT head that
produce every arm's geometry. **No diffusion, no cascade, no generation.** That
is the whole point: stage-2 flow matching transports noise to the clean image's
DA3 features, so the clean features' decode is definitionally the output a
perfect model would produce, and putting prediction and reference through the
identical frozen head cancels that head's own bias.

**It is NOT `geofix_infer --cond-source gt`.** That arm GENERATES with the clean
target sitting in a conditioning slot; it is an inference arm and a positive
control, and substituting it here would score arms against another arm's output.

**It is not metric ground truth either.** DL3DV ships no sensor depth. A perfect
score against this reference means "indistinguishable from encoding the clean
photograph", which is the right target for a refiner and is not the same claim as
"accurate geometry". `depth_eval`'s docstring says this at length; it has to be
said wherever a pseudo-GT number is quoted.

## The recipe is `eval_gld_metric.py:832-855` + its GT decode at 1145-1155

Reused rather than reimplemented, because a second encode/decode path is a second
place for the normalisation to be wrong:

    for lvl in range(4):
        rae.level = lvl
        rae._init_normalization(stat_path[lvl])
        images_norm = (images - rae.encoder_mean) / rae.encoder_std
        gt_feat, _ = rae.encode(images_norm, return_cls=True, mode='single', level=lvl)
        feat_gt_norm[lvl]   = gt_feat                  # [latent_norm]
        feat_gt_denorm[lvl] = rae._denormalize(gt_feat) # [raw]
    decode_into_images(rae, feat_gt_norm, feat_gt_denorm, level=3, ...)

`level=3` and not 1: all four levels come from the clean encode, so
`decode_into_images` takes its no-propagation branch and nothing is generated or
propagated. Note that the encode's `_normalize` and the following `_denormalize`
are exact inverses under the same statistics, so `--stats-dir` CANCELS on this
path -- it is passed anyway, and recorded in the provenance, so that the
reference cannot be mistaken for having been built under a different leg than the
arms it scores.

**ONE LINE OF THAT RECIPE IS NOT COPIED, AND IT IS THE IMPORTANT ONE.**
`eval_gld_metric.py:844` flattens the batch first -- `images_flat =
images_norm.reshape(B * V, C, H, W)` -- and `RAE_DA3._prepare_input` turns a 4D
input into `(B*V, S=1, ...)`. DA3's backbone is multi-view (alternating
local/global attention over S, plus a camera token), so the flatten encodes every
view IN ISOLATION. Training does not: `train_multiview_da3.py:123` encodes the 5D
`(B, V, C, H, W)` tensor, and so does every conditioning encode. We pass the 5D
tensor. See the comment at the `rae.encode` call for the measurement.

## THE GATE, and why it is not optional -- with the calibration it actually needs

A mis-set `stat_path`, a wrong decode level, the artifact render in place of the
photograph, or a collapsed view axis all produce a plausible-looking depth map
and a silently wrong reference. The RGB round trip catches every one of them and
costs nothing, since the decode has already run.

**Do not calibrate it against 35.362 dB.** That is session 2's stage-1 gate on
RE10K. The RAE's ceiling on OUR data, measured the same way (`geofix.eval_stage1`
over 32 clean DL3DV holdout target frames, job 7274940), is **30.073 dB**, range
29.72-30.24 across the splits measured. A 30 dB floor calibrated on RE10K would
reject a perfectly correct reference; that was the first thing this gate did.

So the gate is two checks, and the second is the one that works:

1. `--gate-psnr`, an absolute floor, deliberately LOOSE (25 dB). Per-scene ceiling
   on DL3DV is only measured on a handful of splits of one scene, so a tight
   absolute threshold would be a coin flip on the rest of the holdout.
2. `--gate-tol`, the SHARP check. On the first sample of each shard the same
   frames are decoded again by `geofix.eval_stage1.reconstruct` -- the route that
   produced the stage-1 gate -- and the two must agree to within 0.25 dB. The
   routes differ only in the cls token (measured irrelevant to RGB: real and zero
   cls agree to 3 decimals, job 7274949) and a normalise/denormalise round trip,
   so any real gap means the features are not the clean image's. This is what
   caught the flattened multi-view encode, at 1.6 dB -- a gap no absolute
   threshold could have told apart from "DL3DV is harder than RE10K".

`--gate-only` runs both on `--limit` samples and writes nothing. `--diag` prints
all three decode routes side by side.

## Protocol

Target views only, `[cond_num, num_views)` -- the same slots `depth_eval` scores,
for the reason `CLAUDE.md` records ("Score `*_tgt`, never full-frame"). The
reference slots are clean photographs in every arm and would decode identically
anyway.

The full 8-view batch is still ENCODED and DECODED, never a 4-view one: the ray
head's directions are `H_v @ [u,v,1]` with `H_v` relative to view 0, so dropping
view 0 would silently change the convention the arms' rays were written under.

Resumable and shardable exactly like `geofix_infer.py` -- `--shard/--num-shards`
slice the sample list, finished samples are skipped, and every `.npz` is written
to a temp file and `os.replace`d into place. `np.savez_compressed` is handed an
OPEN FILE HANDLE because it appends ".npz" to any path that does not end in one,
which has bitten this project twice.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

GEOM_KEYS = ("depth", "depth_conf", "ray", "ray_conf")
GEOM_SUFFIX = ".geom.npz"

#: The RAE's own reconstruction ceiling ON THIS DATA, measured (job 7274940,
#: 2026-08-18) with `geofix.eval_stage1` -- the exact path that produced session
#: 2's stage-1 gate -- over 32 clean DL3DV holdout target frames: 30.073 dB,
#: per-scene range 29.72-30.24.
#:
#: NOT 35.36. That figure is session 2's RE10K number, and quoting it as the
#: target here would set a threshold nothing on DL3DV can reach. The 5.3 dB
#: difference is the dataset, not a fault: RE10K is smooth indoor video and
#: DL3DV target frames are high-frequency outdoor content at the same 36x36
#: token grid.
RAE_CEILING_DB = 30.073


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR in dB between two [0,1] tensors, MSE over every element given."""
    mse = torch.mean((a.double() - b.double()) ** 2).item()
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def sample_is_complete(d: pathlib.Path, stems: list[str]) -> bool:
    """True if every target `.geom.npz` exists and opens.

    Same discipline as `geofix_infer.sample_is_complete`: a job killed mid-write
    must not be skipped as finished. `np.load` + touching `.files` reads the zip
    directory, which is where a truncated file fails.
    """
    for stem in stems:
        f = d / f"{stem}{GEOM_SUFFIX}"
        if not f.is_file():
            return False
        try:
            with np.load(f) as z:
                if "depth" not in z.files:
                    return False
        except Exception:
            return False
    return True


def save_geom_atomic(path: pathlib.Path, arrays: dict[str, np.ndarray]) -> None:
    """temp file + os.replace, and savez through an OPEN HANDLE.

    `np.savez_compressed` APPENDS ".npz" to any path not already ending in it, so
    passing the temp path directly writes "<...>.npz.tmp.npz" and the replace then
    fails on a file that was never created. Handing it a file object is the only
    way to keep the name you chose.
    """
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def load_target_rgb(root: pathlib.Path, split: str, stems: list[str],
                    device, ref: torch.Tensor) -> torch.Tensor:
    """The refined RGB of ANOTHER method, in the target slots, as `(1, n, 3, H, W)`.

    This is the machinery behind `--target-rgb-root`, which turns this script from
    "build the pseudo-GT reference" into "run the SAME depth head on somebody
    else's refined RGB" -- the control `depth_eval`'s docstring calls THE
    EXPERIMENT THAT ACTUALLY DECIDES THE CLAIM, and warns is the arm most likely
    to beat us.

    It is a tighter control than DA3-off-the-shelf would be. Running DA3 proper on
    a baseline's RGB would differ from our arms in TWO ways at once -- where the
    RGB came from, and which head produced the depth. Encoding the baseline's RGB
    with the same DA3 encoder and decoding it through the same GLD-finetuned DPT
    head leaves exactly one difference: the pixels. The pseudo-GT reference is
    still not neutral between the sides (it comes from our encoder and our head),
    and that bias favours us and must be disclosed wherever this is quoted.

    Hard rule 3: 504 native, never resized. A baseline that wrote a different
    resolution is an error here, not something to interpolate away.
    """
    from PIL import Image
    imgs = []
    for stem in stems:
        f = root / split / f"{stem}.png"
        if not f.is_file():
            raise SystemExit(
                f"--target-rgb-root: no {f}. Frames are never imputed (hard rule 8); "
                "score the arm on the frames it actually produced or produce them.")
        a = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
        if a.shape[:2] != tuple(ref.shape[-2:]):
            raise SystemExit(
                f"{f} is {a.shape[:2]}, expected {tuple(ref.shape[-2:])}. Nothing in "
                "this project is resized (hard rule 3).")
        imgs.append(torch.from_numpy(a).permute(2, 0, 1))
    return torch.stack(imgs).unsqueeze(0).to(device=device, dtype=ref.dtype)


def diag_paths(rae, gt: torch.Tensor, cond: int) -> dict[str, float]:
    """Decode the same clean frames by the OTHER two routes, for comparison.

    Run by `--diag`. The pseudo-GT route is GLD's own
    (`decode_into_images(level=3)`, four independent `mode='single'` encodes,
    ZERO cls); `geofix.eval_stage1.reconstruct` is the route that produced this
    project's 35.362 dB stage-1 gate (ONE `mode='all'` encode, REAL cls). If the
    two disagree on identical pixels then one of them is not the RAE's ceiling,
    and the gate cannot be calibrated until it is known which.

    `stage1_real_cls` is `reconstruct` verbatim, imported rather than
    transcribed. `stage1_zero_cls` is the same features with the cls replaced by
    zeros, which isolates the cls from the encode mode: the MAE decoder is
    documented to ignore the cls, and this is what checks that claim rather than
    repeating it.
    """
    import importlib
    geofix_src = pathlib.Path(__file__).resolve().parents[2] / "geofix" / "src"
    if str(geofix_src) not in sys.path:
        sys.path.insert(0, str(geofix_src))
    # `geofix.eval_stage1` MOVED to `geofix.experiments.eval_stage1` in the
    # 2026-08-25 reorganisation (docs/MODULE_MAP.md). This call site was missed,
    # and it only fires deep inside `diag_paths` -- AFTER the model is loaded and
    # the first samples have already been written -- so the job burns several
    # minutes of GPU and a plausible log before dying. Nothing re-ran
    # pseudo_gt_geom between the refactor and 2026-08-29, which is why it sat
    # broken for four days.
    #
    # Both names are tried, new one first, so the script works against a checkout
    # from either side of the rename rather than pinning it to one.
    for _name in ("geofix.experiments.eval_stage1", "geofix.eval_stage1"):
        try:
            es = importlib.import_module(_name)
            break
        except ModuleNotFoundError:
            es = None
    if es is None:
        raise ModuleNotFoundError(
            "eval_stage1 not found as geofix.experiments.eval_stage1 nor "
            f"geofix.eval_stage1; looked under {geofix_src}")
    from utils.da3_validation_metric import apply_da3_norm

    out: dict[str, float] = {}
    b, v, c, H, W = gt.shape
    ref = gt[0].float()

    rgb_b = es.reconstruct(rae, apply_da3_norm, gt)
    rgb_b = rgb_b.float().clamp(0, 1)
    out["stage1_real_cls"] = float(np.mean([psnr(rgb_b[k], ref[k]) for k in range(cond, v)]))

    # Same features, cls zeroed. Mirrors `reconstruct` up to the decode.
    with torch.no_grad():
        mean, std = rae.encoder_mean.to(gt.device), rae.encoder_std.to(gt.device)
        raw = rae.encode((gt - mean[None]) / std[None], mode="all")
        ph, pw = H // 14, W // 14
        n_patches = ph * pw
        feats = []
        for lvl in range(4):
            f = raw[lvl]
            if f.ndim == 4:
                f = f.reshape(-1, *f.shape[-2:])
            n, ch = f.shape[-2:]
            patches = apply_da3_norm(rae, f[:, n - n_patches:].reshape(b, v, n_patches, ch))
            feats.append((patches, torch.zeros(b, v, ch, device=f.device, dtype=f.dtype)))
        dtype = (next(rae.rae_cl_decoder.parameters()).dtype
                 if rae.rae_cl_decoder is not None else torch.float32)
        feats = [(p.to(dtype), cc.to(dtype)) for p, cc in feats]
        rgb_c = rae.decode(feats, H, W)["rgb"]
    if rgb_c.ndim == 5:
        rgb_c = rgb_c.reshape(-1, *rgb_c.shape[-3:])
    rgb_c = rgb_c.float().clamp(0, 1)
    out["stage1_zero_cls"] = float(np.mean([psnr(rgb_c[k], ref[k]) for k in range(cond, v)]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True,
                    help="pseudo-GT root; <out>/<split>/<frame>.geom.npz")
    ap.add_argument("--eval-config", required=True)
    ap.add_argument("--model-config-level1", required=True,
                    help="MUST be the *_depth.yaml variant: the DPT head is built "
                         "only when dpt_decoder_path AND da3_weights_path are both "
                         "set, and without it rae.decode returns rgb alone.")
    ap.add_argument("--model-config-cascade", required=True)
    ap.add_argument("--checkpoint-level1", required=True,
                    help="Loaded by build_context and NEVER USED here -- no "
                         "diffusion runs on this path. Passed so the RAE is built "
                         "by exactly the function the arms use.")
    ap.add_argument("--checkpoint-cascade", required=True, help="Also unused; see above.")
    ap.add_argument("--stats-dir", required=True,
                    help="Cancels arithmetically (encode normalises, _denormalize "
                         "inverts it), but is recorded in the provenance so the "
                         "reference cannot be mistaken for a different leg's.")
    ap.add_argument("--num-views", type=int, default=None)
    ap.add_argument("--cond-num", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=504)
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Unused by the decode (no mask enters this path); kept so "
                         "GeoFixPairs is constructed identically to the arms'.")
    ap.add_argument("--target-rgb-root", default=None, metavar="DIR",
                    help="CONTROL MODE. Replace the target-half images with "
                         "<DIR>/<split>/<frame>.png -- another method's refined "
                         "RGB -- and push THAT through the same encoder and the "
                         "same DPT head. Reference slots keep the clean "
                         "photographs, matching what every arm saw. Default (None) "
                         "is the pseudo-GT reference and is unchanged. The output "
                         "of this mode is NOT a reference and must never be passed "
                         "as --pseudo-gt-root.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--gate-psnr", type=float, default=25.0,
                    help="ABSOLUTE floor on the decoded RGB against the clean "
                         f"photograph. The RAE's ceiling on DL3DV is "
                         f"{RAE_CEILING_DB} dB (NOT session 2's 35.36, which is "
                         "RE10K), and the per-scene spread outside the splits "
                         "measured is unknown -- so this floor is deliberately "
                         "loose and catches only a gross wiring failure. The "
                         "sharp check is --gate-tol.")
    ap.add_argument("--gate-tol", type=float, default=0.25,
                    help="THE REAL GATE, in dB. On the first sample of every shard "
                         "the same frames are also decoded by "
                         "`geofix.eval_stage1.reconstruct` -- the route that "
                         "produced this project's stage-1 gate -- and the two must "
                         "agree to within this. Scene-independent, unlike an "
                         "absolute floor: it is what caught the flattened "
                         "multi-view encode (1.6 dB), which no plausible absolute "
                         "threshold would have separated from 'DL3DV is harder "
                         "than RE10K'. Set to 0 to skip (do not).")
    ap.add_argument("--gate-only", action="store_true",
                    help="Run the RGB gate on --limit samples and write NOTHING.")
    ap.add_argument("--diag", action="store_true",
                    help="Write nothing; instead decode the SAME clean frames three "
                         "ways and print the differences. Exists because the gate's "
                         "first calibration was wrong in two independent ways at "
                         "once (see the '--diag' section below).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("the DA3 encode/decode needs a GPU; got CPU only.")

    from latent_blending import build_context
    from geofix_infer import build_batch
    from video.geofix_pairs import GeoFixPairs, assert_view_config

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
        cfg_scale=None,
        cfg_scale_cascade=None,
        clean_refs=None,
    ), device)

    rae, stat_path = ctx["rae"], ctx["stat_path"]
    if getattr(rae, "rae_cl_decoder", None) is None:
        raise SystemExit(
            "rae.rae_cl_decoder is None: the DPT head was not built, so decode() "
            "returns rgb alone and there is no geometry to write. Point "
            "--model-config-level1 at the *_depth.yaml variant (it sets "
            "dpt_decoder_path alongside da3_weights_path).")

    v, cond = ctx["num_views"], ctx["cond_num"]
    assert_view_config(ctx["ref_view_sampling"], cond, manifest)
    if v != int(manifest["num_views"]) or cond != int(manifest["cond_num"]):
        raise ValueError(
            f"view protocol mismatch: context {v}/{cond}, manifest "
            f"{manifest['num_views']}/{manifest['cond_num']}.")
    if v == 1:
        raise RuntimeError("cannot separate the batch axis from the view axis at v == 1.")

    pooling = "max"
    if "max" not in str(manifest["mask_pooling"]).lower():
        raise ValueError(f"manifest mask_pooling {manifest['mask_pooling']!r} is not MAX "
                         "(hard rule 6).")
    dataset = GeoFixPairs(
        args.manifest, mask_types=list(manifest["mask_types"]),
        token_grid=ctx["token_grid"], gamma=args.gamma, pooling=pooling,
        return_gt=True,
    )
    n = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    indices = list(range(n))
    if args.num_shards > 1:
        if not 0 <= args.shard < args.num_shards:
            raise ValueError(f"shard {args.shard} out of range for {args.num_shards}.")
        indices = indices[args.shard::args.num_shards]

    out_root = pathlib.Path(args.out)
    print(f"[pgt] {len(indices)} of {n} samples (shard {args.shard}/{args.num_shards}), "
          f"{v} views, cond_num={cond}, gate >= {args.gate_psnr} dB "
          f"(RAE ceiling ~{RAE_CEILING_DB})", flush=True)

    from utils.da3_validation_metric import decode_into_images

    psnr_tgt: list[float] = []
    psnr_all: list[float] = []
    written = skipped = processed = 0
    t0 = time.time()

    for done, i in enumerate(indices):
        sample = dataset.samples[i]
        split, stems = sample["split"], list(sample["targets"])
        if len(stems) != v - cond:
            raise ValueError(f"sample {i} ({split}) has {len(stems)} targets, expected {v - cond}.")
        if not (args.gate_only or args.diag) and sample_is_complete(out_root / split, stems):
            skipped += 1
            continue

        batch = build_batch(dataset[i], device)
        # THE CLEAN PHOTOGRAPH, in every slot. `batch["gt"]` is the reference
        # photograph for slots [0, cond_num) (a clean photo is its own GT) and the
        # clean target photograph for [cond_num, num_views). `batch["image"]`
        # would be the artifact RENDER on the target half -- encoding that would
        # produce the render's geometry under the name "pseudo-GT", which is the
        # single most dangerous mistake available on this path.
        gt = batch["gt"]
        if args.target_rgb_root is not None:
            # CONTROL MODE: refs stay clean photographs, targets become the other
            # method's refined RGB. Everything downstream -- the encode, the
            # decode, the gate, the write -- is byte-for-byte the pseudo-GT path,
            # which is the entire point: one difference, and it is the pixels.
            gt = gt.clone()
            sub = load_target_rgb(
                pathlib.Path(args.target_rgb_root), split, stems, device, gt)
            if processed == 0:
                # PROOF THE SUBSTITUTION HAPPENED, in the log, permanently.
                # Both the pseudo-GT path and this one reconstruct whatever they
                # are fed to ~30 dB, so the gate CANNOT tell them apart and a
                # silently ineffective --target-rgb-root would look like a clean
                # run. This prints the substituted pixels' PSNR against the clean
                # photograph: it is the baseline's own RGB score, finite and well
                # below the ~100 dB an identical tensor would give.
                print(f"[pgt] SUBSTITUTED targets from {args.target_rgb_root}: "
                      f"{psnr(sub, gt[:, cond:]):.3f} dB vs the clean photograph "
                      f"(inf would mean the flag did nothing)", flush=True)
            gt[:, cond:] = sub
        if processed == 0:
            lo, hi = float(gt.min()), float(gt.max())
            if lo < -0.01 or hi > 1.01:
                raise ValueError(
                    f"batch['gt'] range [{lo:.3f}, {hi:.3f}] is not [0,1]; the loader "
                    "emits ImageNet-normalised tensors and build_batch must undo it.")
            print(f"[pgt] gt range [{lo:.3f}, {hi:.3f}] OK", flush=True)
        # decode_into_images reads batch['image'] only for (B,V,C,H,W); on the
        # level=3 branch it never encodes it. Passing the clean images anyway so
        # the batch is self-consistent if that branch ever changes.
        geo_batch = {"image": gt, "c2w": batch["c2w"], "intrinsic": batch["intrinsic"]}

        B, V, C, H, W = gt.shape
        feat_gt_norm, feat_gt_denorm = {}, {}
        with torch.no_grad():
            for lvl in range(4):
                if not os.path.exists(stat_path[lvl]):
                    raise SystemExit(
                        f"no statistics for level {lvl} at {stat_path[lvl]}. All four "
                        "levels are required: decode_into_images(level=3) reads every "
                        "one of them and a missing level is not a degraded reference, "
                        "it is a crash or a wrong one.")
                rae.level = lvl
                rae._init_normalization(stat_path=stat_path[lvl])
                imgs = (gt - rae.encoder_mean[None].to(device)) / rae.encoder_std[None].to(device)
                # THE 5D TENSOR IS PASSED WHOLE, AND THIS IS THE ONE LINE THAT
                # MATTERS MOST IN THIS FILE.
                #
                # `eval_gld_metric.py:844` writes `images_flat =
                # images_norm.reshape(B * V, C, H, W)` before this call. DA3's
                # backbone is MULTI-VIEW -- `_run_backbone_loop` alternates local
                # and global attention across the S axis and injects a camera
                # token at `alt_start` -- and `RAE_DA3._prepare_input` turns a 4D
                # input into `(B*V, S=1, ...)`. Flattening therefore encodes every
                # view IN ISOLATION, with no cross-view attention at all.
                #
                # Training does not do that: `train_multiview_da3.py:123` calls
                # `rae.encode(images_norm)` on the 5D `(B, V, C, H, W)` tensor, so
                # the features stage 2 is supervised toward are multi-view. So is
                # every conditioning encode (`da3_validation_metric.py:177`).
                # Copying the flatten would build a reference out of a feature
                # distribution the model is never asked to produce.
                #
                # MEASURED, on the same holdout frames (job 7274949): flattened
                # 28.49/28.60 dB against the clean photograph, multi-view
                # 30.10/30.26 -- and the RAE's own ceiling on this data is 30.07
                # (job 7274940). The flattened path is 1.6 dB below the ceiling
                # and the multi-view path sits on it. Nothing about the depth maps
                # would have looked wrong.
                f, _ = rae.encode(imgs, return_cls=True, mode="single", level=lvl)
                feat_gt_norm[lvl] = f                      # [latent_norm]
                feat_gt_denorm[lvl] = rae._denormalize(f)  # [raw]

            out = decode_into_images(
                rae=rae, features=feat_gt_norm, feat_latent_denorm=feat_gt_denorm,
                level=3, total_view=v, cond_num=cond, device=device,
                batch=geo_batch, stat_path=stat_path, sample_idx=0,
            )

        rgb = out["rgb"]
        if rgb is None:
            raise RuntimeError("decode returned no 'rgb'; the gate cannot run.")
        if rgb.ndim == 5:
            if rgb.shape[0] != 1:
                raise RuntimeError(f"expected batch 1, got {rgb.shape[0]}")
            rgb = rgb[0]
        if rgb.shape[0] != v:
            raise RuntimeError(f"decoded {rgb.shape[0]} views, expected {v}")
        rgb = rgb.float().clamp(0, 1)

        # THE GATE. Per view, so one broken frame cannot hide inside a mean.
        ref = gt[0].float()
        p_all = [psnr(rgb[k], ref[k]) for k in range(v)]
        p_tgt = p_all[cond:]
        psnr_all += p_all
        psnr_tgt += p_tgt
        run_mean = float(np.mean(psnr_tgt))

        if args.diag:
            d = diag_paths(rae, gt, cond)
            print(f"[diag] {split}\n"
                  f"        pseudo_gt (decode_into_images level=3, zero cls): "
                  f"{np.mean(p_tgt):.3f} dB\n"
                  f"        stage1 reconstruct (mode='all', REAL cls):        "
                  f"{d['stage1_real_cls']:.3f} dB\n"
                  f"        stage1 reconstruct (mode='all', ZERO cls):        "
                  f"{d['stage1_zero_cls']:.3f} dB", flush=True)
            processed += 1
            continue

        if processed == 0 or (done + 1) % 10 == 0:
            print(f"[pgt] {split}: target PSNR {np.mean(p_tgt):.3f} dB "
                  f"(views {' '.join(f'{x:.2f}' for x in p_tgt)}), running mean "
                  f"{run_mean:.3f} dB over {len(psnr_tgt)} views", flush=True)
        if run_mean < args.gate_psnr:
            raise SystemExit(
                f"GATE FAILED (absolute floor): pseudo-GT RGB reaches only "
                f"{run_mean:.3f} dB against the clean photograph over "
                f"{len(psnr_tgt)} target views, below the {args.gate_psnr} dB floor "
                f"(RAE ceiling on this data {RAE_CEILING_DB} dB). The statistics "
                "path, the decode level or the images being encoded are wrong; a "
                "depth reference built from this would be silently bad. Nothing "
                "further was written.")

        if processed == 0 and args.gate_tol > 0:
            # THE SHARP GATE. Same frames, same RAE, the stage-1 route -- which
            # differs from this one only in the cls (measured irrelevant to RGB)
            # and a normalise/denormalise round trip. Any real gap means the
            # features being decoded are not the clean image's features.
            d = diag_paths(rae, gt, cond)
            gap = float(np.mean(p_tgt)) - d["stage1_real_cls"]
            print(f"[pgt] gate: pseudo_gt {np.mean(p_tgt):.3f} dB vs stage1 "
                  f"reconstruct {d['stage1_real_cls']:.3f} dB, gap {gap:+.3f} dB "
                  f"(tol {args.gate_tol})", flush=True)
            if abs(gap) > args.gate_tol:
                raise SystemExit(
                    f"GATE FAILED (identity): the pseudo-GT decode is {gap:+.3f} dB "
                    f"from `geofix.eval_stage1.reconstruct` on the SAME frames, "
                    f"outside +/-{args.gate_tol}. The two routes differ only in the "
                    "cls token and a normalise/denormalise round trip, neither of "
                    "which can move RGB, so a gap means the features are not the "
                    "clean image's. This is exactly how the flattened multi-view "
                    "encode was found. Nothing was written.")

        geom: dict[str, np.ndarray] = {}
        for key in GEOM_KEYS:
            t = out.get(key, None)
            if t is None:
                continue
            # Drop leading singleton axes BY SIZE, not by rank: depth/depth_conf
            # come back (1,V,504,504) and ray (1,V,288,288,6). Rays live on the
            # 288 grid and are never upsampled -- one grid for both is wrong.
            while t.ndim > 1 and t.shape[0] == 1 and t.shape[0] != v:
                t = t[0]
            if t.shape[0] != v:
                raise RuntimeError(
                    f"decoded '{key}' has leading axis {t.shape[0]}, expected {v} views "
                    f"(full shape {tuple(t.shape)}).")
            geom[key] = t.detach().float().cpu().numpy()
        if processed == 0:
            missing = [k for k in GEOM_KEYS if k not in geom]
            print(f"[pgt] geometry emitted: {{{', '.join(f'{k}: {geom[k].shape}' for k in geom)}}}",
                  flush=True)
            if missing:
                print(f"[pgt] geometry ABSENT from rae.decode: {missing}", flush=True)
        if "depth" not in geom:
            raise RuntimeError(
                "rae.decode returned no 'depth'. Writing an rgb-only file under a "
                "geometry name would be a silent no-op -- fix the config (the DPT "
                "head needs dpt_decoder_path) rather than the file layout.")

        processed += 1
        if args.gate_only:
            continue
        d = out_root / split
        d.mkdir(parents=True, exist_ok=True)
        for k, stem in enumerate(stems):
            save_geom_atomic(d / f"{stem}{GEOM_SUFFIX}",
                             {kk: vv[cond + k] for kk, vv in geom.items()})
            written += 1

        if (done + 1) % 10 == 0 or done + 1 == len(indices):
            el = time.time() - t0
            print(f"[pgt] {done + 1}/{len(indices)} samples, {written} frames, "
                  f"{el / max(processed, 1):.1f}s/sample", flush=True)

    mean_tgt = float(np.mean(psnr_tgt)) if psnr_tgt else float("nan")
    mean_all = float(np.mean(psnr_all)) if psnr_all else float("nan")
    print(f"\n[pgt] GATE: decoded pseudo-GT RGB vs the clean photograph\n"
          f"        target views [{cond},{v}): {mean_tgt:.3f} dB over {len(psnr_tgt)} views\n"
          f"        all views:               {mean_all:.3f} dB over {len(psnr_all)} views\n"
          f"        RAE stage-1 ceiling MEASURED ON DL3DV: {RAE_CEILING_DB} dB "
          f"(session 2's 35.36 is RE10K and does not apply here)", flush=True)
    print(f"[pgt] wrote {written} frames, skipped {skipped} complete samples, "
          f"processed {processed}", flush=True)

    if not (args.gate_only or args.diag) and processed:
        prov = {
            "what": ("pseudo-GT geometry: the CLEAN photograph encoded at all 4 levels "
                     "and decoded through the same RAE + DPT head as the arms. No "
                     "diffusion, no cascade. NOT --cond-source gt, and NOT metric "
                     "ground truth."
                     if args.target_rgb_root is None else
                     "CONTROL, NOT A REFERENCE: another method's refined RGB "
                     f"({args.target_rgb_root}) in the target slots, clean "
                     "photographs in the reference slots, encoded and decoded "
                     "through the same RAE + DPT head. Answers 'is our NATIVE "
                     "depth better than the same head run on somebody else's "
                     "refined pixels?'. Never pass this as --pseudo-gt-root."),
            "target_rgb_root": args.target_rgb_root,
            "manifest": str(args.manifest),
            "model_config_level1": str(args.model_config_level1),
            "stats_dir": str(args.stats_dir),
            "stats_note": "cancels arithmetically on this path (encode normalises, "
                          "_denormalize inverts); recorded for provenance only",
            "decode": "decode_into_images(level=3) -- all 4 levels from the clean "
                      "encode, so the no-propagation branch runs",
            "num_views": v, "cond_num": cond,
            "views_written": "target only, [cond_num, num_views)",
            "shard": args.shard, "num_shards": args.num_shards,
            "n_samples": len(indices), "n_samples_total": n,
            "n_frames": written, "samples_skipped_complete": skipped,
            "gate_psnr_target_views": mean_tgt,
            "gate_psnr_all_views": mean_all,
            "gate_n_target_views": len(psnr_tgt),
            "gate_floor": args.gate_psnr,
            "rae_ceiling_db": RAE_CEILING_DB,
            "gpu": torch.cuda.get_device_name(0),
            "geom_keys": sorted(geom),
            "command": " ".join(sys.argv),
        }
        out_root.mkdir(parents=True, exist_ok=True)
        pj = out_root / f"provenance_shard{args.shard}of{args.num_shards}.json"
        tmp = pj.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(prov, indent=1))
        os.replace(tmp, pj)
        print(f"[pgt] provenance -> {pj}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
