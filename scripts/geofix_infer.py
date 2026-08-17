#!/usr/bin/env python
"""Session 8: run a finetuned GeoFix checkpoint over the frozen holdout and dump PNGs.

    PYTHONPATH=.:src python scripts/geofix_infer.py \
        --manifest   <geofix>/eval/train/manifest_holdout_it03000.json \
        --eval-config <geofix>/configs/eval_blend.yaml \
        --model-config-level1  configs/training/DA3_geofix_level1.yaml \
        --model-config-cascade <geofix>/third_party/gld/configs/training/DA3_cascade.yaml \
        --checkpoint-level1  results/geofix/level1/<run>/checkpoints/0003000.pt \
        --checkpoint-cascade <geofix>/third_party/gld/checkpoints/da3_cascade.pt \
        --stats-dir <geofix>/eval/feature_stats/legs/dl3dv_art_504__imagenet \
        --cond-artifact --out <geofix>/eval/session8/geofix_l1

## What this is, and what it deliberately is NOT

It is `latent_blending.run_sample` with the blend hook removed and the two GeoFix
conditioning slots switched on. Every heavy component is IMPORTED from that script
-- `build_context` for the RAE, both models, the transport, the sampler factory and
the normalisation statistics; `run_sample`'s two-stage structure for level-1
generation, the learned L1->L0 cascade and the RGB decode.

That reuse is the point and not laziness. Every number this project has on the
generated half -- arm A, the oracle bounds, the clean-reference result -- came
through exactly this decode. A fresh sampler here would produce a figure that could
not be compared to any of them, and the discrepancy would look like a result.

It is NOT a scorer. It writes `<out>/K_XX/<split>/<frame>.png` and stops.

**Score these arms with `geofix.baselines.flux2d --mode score/report`, not with
`geofix.score_arms`.** Both read this layout, but `score_arms` enumerates splits off
the export tree instead of the manifest and delegates to
`blend.report.paired_deltas`, which emits **no t statistic** and does not aggregate
`hf_ratio`. `flux2d.py` implements exactly the protocol these arms need -- target
views from the manifest, paired per scene, t, win count, PSNR + LPIPS + DSIM +
hf_ratio, `render` as the control endpoint -- and it is what produced the FLUX
baseline numbers, so using it keeps GeoFix and its 2D control on one scorer.
(`score_arms` is not *wrong* on these directories, because it intersects arm and GT
stems and only target frames are written here; it is just weaker.)

## The eval set comes from the MANIFEST, not from a glob

`GeoFixPairs` is the same loader the training run uses, pointed at the holdout
manifest. So the view protocol, the reference placement, the mask polarity check and
the `it_03000`-only holdout filter are all inherited rather than restated -- which
is the only way to be sure the model is evaluated on inputs shaped like the ones it
trained on.

## --stats-dir MUST match training, and that costs a fresh arm A

Session 6.5 ran with the RELEASED statistics (`third_party/gld/model_stats/da3`).
Session 8 trains with our recomputed leg (`eval/feature_stats/legs/
dl3dv_art_504__imagenet`) because hard rule 5 requires it -- the measurement says
the dataset term dominates, so the released RE10K-derived files are wrong for DL3DV
even setting artifacts aside.

Inference has to use the statistics the model was TRAINED with, or every latent is
mis-normalised. That is not negotiable, and it has a consequence: **6.5's published
arm-A number is not a valid comparison point for this model**, because it was
decoded under different normalisation. Do not quote a GeoFix-minus-arm-A delta
against it.

The fix is cheap, and it is the intended way to use this script -- generate all
three arms here, under identical statistics, seeds and decode:

    arm `render`  --dump-render        the unrefined 3DGS render (no model at all)
    arm `gld_a`   neither slot         stock GLD generation, the MECHANISM baseline
    arm `geofix`  --cond-artifact      refinement                 (+ --mask-in-camera)

`render` is the endpoint the project's claim is made against; `gld_a` is what
isolates the conditioning from the generator. Both are required, for a reason
session 6.5 established the hard way: under degraded references arm A sat 0.584 dB
BELOW the renders, so any arm that pastes render content back gains from area
alone -- and the sign of that confound flips with reference mode.

## Only target views are written

Slots `[0, cond_num)` are the clean photographs. They are inputs, they decode
near-identically in every arm, and scoring them dilutes every delta exactly 2x --
the session-6.5 mistake this project has already paid for once. `score_arms` scores
what it finds, so writing a reference view here would silently reintroduce that
dilution. Target frames only, named by their manifest stem.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def build_batch(views: list[dict], device) -> dict:
    """Stack one sample's per-view dicts into the (1, V, ...) batch `run_sample` wants.

    `GeoFixPairs.__getitem__` returns CUT3R's list-of-per-view-dicts. The training
    path stacks it with `collate` and then `convert_cut3r_batch`; here there is one
    sample and no DataLoader, so the stack is explicit -- and the key names are the
    ones `get_denoised_features` accepts (`image` / `c2w` / `intrinsic`), not
    CUT3R's.
    """
    def stack(key):
        return torch.stack([v[key] for v in views]).unsqueeze(0).to(device)

    return {
        "image": stack("img"),            # the 3DGS renders + clean refs, as loaded
        "c2w": stack("camera_pose"),
        "intrinsic": stack("camera_intrinsics"),
        "gt": stack("gt"),
        "mask": stack("mask"),
    }


def to_uint8(rgb: torch.Tensor) -> np.ndarray:
    """(3, H, W) in [0, 1] -> (H, W, 3) uint8, matching the export's PNG convention."""
    x = rgb.detach().float().clamp(0, 1).cpu().numpy()
    return (np.transpose(x, (1, 2, 0)) * 255.0 + 0.5).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True,
                    help="eval/train/manifest_holdout_it03000.json -- the FROZEN holdout.")
    ap.add_argument("--out", required=True, help="arm root; <out>/K_XX/<split>/<frame>.png")
    # Passed through to latent_blending.build_context unchanged.
    ap.add_argument("--eval-config", required=True)
    ap.add_argument("--model-config-level1", required=True)
    ap.add_argument("--model-config-cascade", required=True)
    ap.add_argument("--checkpoint-level1", required=True)
    ap.add_argument("--checkpoint-cascade", required=True)
    ap.add_argument("--stats-dir", required=True)
    ap.add_argument("--num-views", type=int, default=None)
    ap.add_argument("--cond-num", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=504)
    ap.add_argument("--cfg-scale", type=float, default=None)
    ap.add_argument("--cfg-scale-cascade", type=float, default=None)
    # The ablation ladder, same two booleans as training.
    ap.add_argument("--cond-artifact", action="store_true",
                    help="Fill latents_cond[:, cond_num:] with render features.")
    ap.add_argument("--mask-in-camera", action="store_true",
                    help="Grade camera channel 0 by M_edit. Trains, but only after "
                         "~70 steps of LR warmup absorb the distribution shift -- see "
                         "src/utils/geofix_slots.py before reading an early curve.")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Contrast exponent on the pooled mask; must match training.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Fixed across arms so xT is identical and the comparison is paired.")
    ap.add_argument("--limit", type=int, default=None, help="Smoke: first N samples only.")
    ap.add_argument("--dump-render", action="store_true",
                    help="Also write the unrefined render as a sibling arm. The "
                         "artifact endpoint is a REQUIRED control on this data (arm A "
                         "sat below it under degraded refs), so having it produced by "
                         "the same walk removes any chance of a misaligned pairing.")
    args = ap.parse_args()

    if not (args.cond_artifact or args.mask_in_camera):
        print("[infer] NOTE: neither slot enabled -- this is stock GLD generation "
              "(session 6.5's arm A), which is a legitimate arm but not GeoFix.",
              flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("inference needs a GPU; got CPU only.")

    from latent_blending import build_context
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
        cfg_scale=args.cfg_scale,
        cfg_scale_cascade=args.cfg_scale_cascade,
        # The manifest already placed clean photographs in [0, cond_num); this flag
        # is build_context's own reference-swapping path, which would do it twice.
        clean_refs=None,
    ), device)

    v, cond = ctx["num_views"], ctx["cond_num"]
    # Fail before loading 194 samples if the protocol disagrees. `assert_view_config`
    # compares the manifest's recorded ordering against what is actually in force --
    # a mismatch here means renders would land in reference slots.
    assert_view_config(ctx["ref_view_sampling"], cond, manifest)
    if v != int(manifest["num_views"]) or cond != int(manifest["cond_num"]):
        raise ValueError(
            f"view protocol mismatch: context has num_views={v} cond_num={cond}, "
            f"manifest has {manifest['num_views']}/{manifest['cond_num']}.")

    # `manifest["mask_pooling"]` is PROSE, not an enum -- it reads "MAX to 36x36,
    # applied in the fork at load time (hard rule 6)". Passing it straight through
    # raises in GeoFixPairs, which is the good outcome; the bad one would have been
    # a field that happened to parse as the wrong mode. So the value is hard rule 6
    # ("MAX, never MEAN") and the manifest string is CHECKED against it rather than
    # read from.
    pooling = "max"
    if "max" not in str(manifest["mask_pooling"]).lower():
        raise ValueError(
            f"manifest mask_pooling is {manifest['mask_pooling']!r}, which does not "
            f"mention MAX. Hard rule 6 requires max pooling to the token grid; a "
            f"manifest built with mean pooling cannot be scored as if it were max.")
    dataset = GeoFixPairs(
        args.manifest,
        mask_types=list(manifest["mask_types"]),
        token_grid=ctx["token_grid"],
        gamma=args.gamma,
        pooling=pooling,
        return_gt=True,
    )
    n = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    out_root = pathlib.Path(args.out)
    render_root = out_root.parent / f"{out_root.name}__render" if args.dump_render else None
    print(f"[infer] {n} samples, {v} views, cond_num={cond}, "
          f"cond_artifact={args.cond_artifact} mask_in_camera={args.mask_in_camera}",
          flush=True)

    from utils.da3_validation_metric import get_denoised_features, decode_into_images
    from eval_gld_metric import get_cascade_features

    rae, stat_path = ctx["rae"], ctx["stat_path"]
    written = 0
    t_start = time.time()

    for i in range(n):
        sample = dataset.samples[i]
        split = sample["split"]                     # already "K_06/<scene>__run_XXX__it_YYYYY"
        stems = list(sample["targets"])
        if len(stems) != v - cond:
            raise ValueError(
                f"sample {i} ({split}) has {len(stems)} targets, expected {v - cond}.")

        batch = build_batch(dataset[i], device)
        geo_batch = {k: batch[k] for k in ("image", "c2w", "intrinsic")}

        art = batch["image"] if args.cond_artifact else None
        msk = batch["mask"] if args.mask_in_camera else None
        if msk is not None and msk.shape[2] != 1:
            raise ValueError(
                f"camera-channel injection takes ONE mask plane, manifest stacks "
                f"{msk.shape[2]}. Reducing several planes to one is a modelling "
                "decision; make it explicitly rather than here.")

        feat, feat_denorm = {}, {}

        # --- stage 1: noise -> L1, with the GeoFix slots ---------------------
        rae.level = 1
        rae._init_normalization(stat_path=stat_path[1])
        # Reseeded PER SAMPLE from a fixed base, so arm-vs-arm is paired at every
        # sample and not merely on average -- and so a `--limit` smoke reproduces
        # the same xT as the corresponding prefix of the full run.
        torch.manual_seed(args.seed + i)
        torch.cuda.manual_seed_all(args.seed + i)
        feat[1] = get_denoised_features(
            rae=rae, model=ctx["model1"], transport=ctx["transport"],
            sampler=ctx["make_sample_fn"](None), loader=None, device=device,
            total_view=v, cond_num=cond, val_num_batches=1,
            use_prope=ctx["use_prope"], rank=0, world_size=1,
            prope_image_size=ctx["hw"], predict_cls=False, joint_ode=False,
            ref_view_sampling=ctx["ref_view_sampling"], camera_mode=ctx["camera_mode"],
            is_concat_mode=True, pag_scale=None, pag_layer_idx=None,
            cfg_scale=ctx["cfg_scale"], use_camera_drop=ctx["use_camera_drop"],
            cfg_uncond_mode=ctx["cfg_uncond_mode"], batch=geo_batch,
            geofix_artifact_images=art, geofix_mask=msk,
        )
        feat_denorm[1] = rae._denormalize(feat[1])

        # --- stage 2: L1 -> L0, the learned cascade (UNCHANGED, released) -----
        torch.manual_seed(args.seed + i + 1)
        torch.cuda.manual_seed_all(args.seed + i + 1)
        feat[0] = get_cascade_features(
            rae=rae, cascade_model=ctx["cascade_model"],
            sampler=ctx["make_sample_fn"](None),
            source_features=feat[1], source_stat_path=stat_path[1],
            target_stat_path=stat_path[0], batch=geo_batch, device=device,
            total_view=v, cond_num=cond, camera_mode=ctx["camera_mode"],
            use_prope=ctx["use_prope"], cfg_scale=ctx["cfg_scale_cascade"],
            use_camera_drop=ctx["use_camera_drop"],
            cfg_uncond_mode=ctx["cfg_uncond_mode"], noise_tau=0.0,
            prope_image_size=ctx["hw"], eval_mode="cascade",
        )
        rae.level = 0
        rae._init_normalization(stat_path=stat_path[0])
        feat_denorm[0] = rae._denormalize(feat[0])

        out = decode_into_images(
            rae=rae, features=feat, feat_latent_denorm=feat_denorm, level=1,
            total_view=v, cond_num=cond, device=device, batch=geo_batch,
            stat_path=stat_path, sample_idx=0,
        )
        rgb = out["rgb"]
        # Drop the batch axis ONCE, here. Getting this wrong does not crash in the
        # obvious place -- it broadcasts (latent_blending.run_sample says the same).
        if rgb.ndim == 5:
            if rgb.shape[0] != 1:
                raise RuntimeError(f"expected batch 1, got {rgb.shape[0]}")
            rgb = rgb[0]
        if rgb.shape[0] != v:
            raise RuntimeError(f"decoded {rgb.shape[0]} views, expected {v}")

        d = out_root / split
        d.mkdir(parents=True, exist_ok=True)
        for k, stem in enumerate(stems):
            Image.fromarray(to_uint8(rgb[cond + k])).save(d / f"{stem}.png")
            written += 1
        if render_root is not None:
            dr = render_root / split
            dr.mkdir(parents=True, exist_ok=True)
            for k, stem in enumerate(stems):
                Image.fromarray(to_uint8(batch["image"][0, cond + k])).save(dr / f"{stem}.png")

        if (i + 1) % 10 == 0 or i + 1 == n:
            el = time.time() - t_start
            print(f"[infer] {i + 1}/{n} samples, {written} frames, "
                  f"{el / (i + 1):.1f}s/sample, eta {(n - i - 1) * el / (i + 1) / 60:.1f}min",
                  flush=True)

    prov = {
        "manifest": str(args.manifest),
        "n_samples": n,
        "n_frames": written,
        "checkpoint_level1": str(args.checkpoint_level1),
        "checkpoint_cascade": str(args.checkpoint_cascade),
        "stats_dir": str(args.stats_dir),
        "cond_artifact": bool(args.cond_artifact),
        "mask_in_camera": bool(args.mask_in_camera),
        "mask_types": list(manifest["mask_types"]),
        "mask_polarity": manifest["mask_polarity"],
        "mask_pooling": pooling,
        "mask_pooling_manifest": manifest["mask_pooling"],
        "gamma": args.gamma,
        "num_views": v,
        "cond_num": cond,
        "views_written": "target only, [cond_num, num_views)",
        "seed_scheme": "args.seed + sample_index (paired across arms)",
        "cfg_scale": ctx["cfg_scale"],
        "cfg_scale_cascade": ctx["cfg_scale_cascade"],
        "decode": "latent_blending.build_context + L1 gen + learned cascade + decode_into_images",
        "command": " ".join(sys.argv),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "_provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"[infer] done: {written} frames from {n} samples -> {out_root}", flush=True)
    print(f"[infer] score with: python -m geofix.score_arms --arm geofix={out_root} ...",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
