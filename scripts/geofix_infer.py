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

and, for any claim made about the mask, its area-matched UNIFORM control:

    arm `geofix_const`  --cond-artifact --mask-in-camera --mask-const match

which feeds a spatially constant mask of the SAME per-frame area, so the only
thing removed is WHERE the mask points. Session 6.5 measured this as the
difference between a result and its opposite: `depth_disagree_a45` reads -0.151 dB
against no mask and +0.375 dB against its area-matched control. Point the control
at its own --out; the sbatch default derives the directory from the arm name.

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
import os
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


#: CUT3R's convention, and the source of a bug that cost a whole smoke run.
#: `GeoFixPairs._img` returns IMAGENET-NORMALISED tensors (range about [-2.1, 2.6]),
#: because every CUT3R loader does; `cut3r_adapter.convert_cut3r_batch` then converts
#: them BACK to [0, 1] (`gt_inp = imgs * std + mean`, line 62) before
#: `prepare_data`/`get_denoised_features` apply the normalisation once, for real.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_batch(views: list[dict], device) -> dict:
    """Stack one sample's per-view dicts into the (1, V, ...) batch the sampler wants.

    `GeoFixPairs.__getitem__` returns CUT3R's list-of-per-view-dicts. The training
    path stacks it with `collate` and then `convert_cut3r_batch`; here there is one
    sample and no DataLoader, so the stack is explicit -- and the key names are the
    ones `get_denoised_features` accepts (`image` / `c2w` / `intrinsic`).

    **The de-normalisation below is NOT optional, and omitting it does not crash.**
    Skipping it double-normalises every image: the loader's ImageNet transform, then
    `get_denoised_features`' own `(images - rae.encoder_mean) / rae.encoder_std`.
    Measured cost of that bug: stock GLD generation at 13.25 dB against the render's
    17.39, decoding to blocky, cyan-cast patch mush -- which looks enough like "the
    model is bad" to be mistaken for a result. It is the reason this function exists
    instead of a bare `torch.stack`, and the reason `convert_cut3r_batch` is quoted
    above rather than paraphrased.
    """
    def stack(key):
        return torch.stack([v[key] for v in views]).unsqueeze(0).to(device)

    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)

    def denorm(x):
        return (x * std + mean).clamp(0, 1)

    return {
        # [0, 1], exactly what convert_cut3r_batch hands the training path.
        "image": denorm(stack("img")),    # 3DGS renders in [cond_num, V), clean refs below
        "c2w": stack("camera_pose"),
        "intrinsic": stack("camera_intrinsics"),
        "gt": denorm(stack("gt")),
        "mask": stack("mask"),            # a mask is not an image; never normalised
    }


def parse_mask_const(text: str):
    """`--mask-const`: the literal `match`, or a float in [0, 1]."""
    s = text.strip().lower()
    if s == "match":
        return "match"
    try:
        x = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--mask-const takes 'match' or a float in [0,1]; got {text!r}.")
    if not 0.0 <= x <= 1.0:
        raise argparse.ArgumentTypeError(
            f"--mask-const {x} is outside [0,1]. The mask is `edit1` in [0,1]; a "
            "constant outside that range is not a control, it is an "
            "out-of-distribution input.")
    return x


def constant_mask(msk: torch.Tensor, const, cond_num: int):
    """(B,V,1,g,g) `edit1` mask -> the same tensor with every plane made UNIFORM.

    The control asks whether the model reads WHERE the mask points or only HOW
    MUCH of it there is. Removing placement while holding area fixed is the only
    way to separate those, and getting the area wrong turns the control into a
    second confound -- 6.5's `const_a{NN}` arms are matched per arm, and the
    oracle's advantage over them swings +0.318 -> +0.985 -> -0.076 dB across the
    area range, so a mismatched constant can manufacture or erase the whole effect.

    Two modes, and `match` is the one to prefer:

    - `match`  the constant is that VIEW's own mean, so area is matched exactly
               per frame -- stronger than per-sample and much stronger than
               6.5's per-arm matching. It preserves whatever frame-level severity
               the mask carries (which our masks barely do -- `loso_logistic`
               moves 0.493 -> 0.518 while the truth swings 0.053 -> 0.828) and
               destroys placement alone. That is the clean isolation.
    - a float  one fixed constant everywhere. It does NOT match area per sample,
               so it confounds placement with area unless the arm it controls
               happens to sit at that area; kept because a fixed level is also
               how you ask "does the model respond to the scalar at all".

    Polarity and range are untouched: `edit1`, [0, 1], no sign flip (hard rule 7).
    The mean is taken AFTER `--gamma`, i.e. on the tensor the network actually
    sees, so the match is in the space where it has to hold.

    Only slots `[cond_num, V)` are written. Reference slots stay zero because the
    model ignores them anyway -- `train_multiview_da3` assigns
    `random_masks[:, cond_views:] = m_img[:, cond_views:]` -- and putting a
    nonzero constant there would be a second, unmeasured change of input.
    """
    if msk[:, :cond_num].abs().max() > 0:
        raise ValueError(
            "reference slots [0, cond_num) carry a nonzero mask; GeoFixPairs "
            "emits zeros there and the area match assumes it.")
    out = torch.zeros_like(msk)
    tgt = msk[:, cond_num:]
    if const == "match":
        fill = tgt.mean(dim=(-2, -1), keepdim=True).expand_as(tgt)
    else:
        fill = torch.full_like(tgt, float(const))
    out[:, cond_num:] = fill
    return out, float(fill.mean())


def to_uint8(rgb: torch.Tensor) -> np.ndarray:
    """(3, H, W) in [0, 1] -> (H, W, 3) uint8, matching the export's PNG convention."""
    x = rgb.detach().float().clamp(0, 1).cpu().numpy()
    return (np.transpose(x, (1, 2, 0)) * 255.0 + 0.5).astype(np.uint8)


def sample_is_complete(d: pathlib.Path, stems: list[str]) -> bool:
    """True if every target PNG for this sample is present AND decodable.

    Needed because a scavenger preemption kills the process mid-sample. Without
    this the shard restarts at index 0 and redoes up to ~1.4 h of finished work,
    which makes preemptible capacity a bad trade rather than free throughput.

    `Image.verify()` rather than a size check: a job killed mid-write leaves a
    TRUNCATED png, and skipping one of those would silently ship a corrupt frame
    to the scorer. Verify is cheap (header + CRC, no full decode) and it is the
    difference between "resume" and "resume, usually".
    """
    for stem in stems:
        f = d / f"{stem}.png"
        if not f.is_file():
            return False
        try:
            with Image.open(f) as im:
                im.verify()
        except Exception:
            return False
    return True


def save_atomic(arr, path: pathlib.Path) -> None:
    """Write via a temp file in the same directory, then os.replace.

    os.replace is atomic within a filesystem, so a reader (or a restarted shard)
    never observes a half-written png. Writing in place is what would make
    sample_is_complete's verify necessary in the first place.
    """
    tmp = path.with_suffix(".png.tmp")
    # format="PNG" is REQUIRED: PIL infers the encoder from the extension, and
    # ".tmp" is unknown to it. Without this every resumed shard dies on its first
    # write -- caught by the round-trip test, not by inspection.
    Image.fromarray(arr).save(tmp, format="PNG")
    os.replace(tmp, path)


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
    ap.add_argument("--cond-source", default="render",
                    choices=("render", "gt", "zeros", "shuffled"),
                    help="WHAT goes in slot 1, holding everything else fixed. This is "
                         "the causal test of whether the finetune learned to READ that "
                         "slot or merely ignores it and does novel-view synthesis:\n"
                         "  render   - the degraded 3DGS render (normal operation)\n"
                         "  gt       - the CLEAN target. Positive control: a model that "
                         "copies from slot 1 should jump toward the ceiling; a model "
                         "doing pure NVS is INVARIANT and will not move.\n"
                         "  zeros    - stock GLD content. Equals the gld_a arm; if "
                         "`render` scores the same, the slot is being ignored.\n"
                         "  shuffled - ANOTHER sample's render. Negative control: "
                         "invariance means ignored, corruption means genuinely read.")
    ap.add_argument("--blend-mask", action="store_true",
                    help="Composite toward the render's features at every ODE step, "
                         "weighted by M_edit -- session 6.5's training-free mechanism, "
                         "applied to a FINETUNED model. Preservation outside the mask "
                         "then holds BY CONSTRUCTION instead of being learned, which "
                         "is the failure measured on the slot route (edits only 1.76x "
                         "stronger inside a PERFECT mask). Pair it with a model trained "
                         "cond_artifact=T / mask_in_camera=F, i.e. the `geofix` arm: "
                         "the mask arrives at sampling time, so the network needs no "
                         "mask input at all.")
    ap.add_argument("--mask-const", type=parse_mask_const, default=None,
                    metavar="match|FLOAT",
                    help="AREA-MATCHED UNIFORM CONTROL. Replace the mask fed to "
                         "camera channel 0 with a spatially CONSTANT plane, "
                         "keeping polarity (`edit1`, [0,1]) and area but "
                         "destroying placement. Requires --mask-in-camera.\n"
                         "  match  - the constant is each TARGET VIEW's own mean "
                         "mask value, so area is matched exactly per frame. This "
                         "is the mode to use: it isolates placement and nothing "
                         "else.\n"
                         "  FLOAT  - one fixed constant on every target view. "
                         "Does not match area per sample, so it re-confounds "
                         "placement with area; use it only to probe the scalar.\n"
                         "Every deployable-mask claim needs this arm: 6.5 measured "
                         "an arm at -0.151 dB against no mask and +0.375 dB "
                         "against its area control -- ignoring it inverted the "
                         "sign of the conclusion.")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Contrast exponent on the pooled mask; must match training.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Fixed across arms so xT is identical and the comparison is paired.")
    ap.add_argument("--limit", type=int, default=None, help="Smoke: first N samples only.")
    ap.add_argument("--weights", choices=("ema", "model"), default="ema",
                    help="Which tensors to load from the checkpoint. EMA is the "
                         "right choice at convergence and the WRONG one early: "
                         "ema_decay 0.9995 is a 2000-step time constant, so at step "
                         "500 the EMA is still 78%% the released initialisation "
                         "(measured drift 0.0021 vs the raw model's 0.0109). An "
                         "early EMA evaluated with the conditioning slots on is "
                         "essentially the RELEASED model fed inputs it never saw, "
                         "which reads as a broken finetune rather than a young one.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Slice samples [shard::num_shards]. Needed, not merely "
                         "faster: one arm is ~6.5h unsharded and the full grid of "
                         "3 arms x 3 checkpoints is ~58h. Shards write disjoint "
                         "split directories and per-shard provenance, and the "
                         "per-sample seed keys off the GLOBAL index so an 8-shard "
                         "run reproduces a 1-shard run exactly.")
    ap.add_argument("--dump-render", action="store_true",
                    help="Also write the unrefined render as a sibling arm. The "
                         "artifact endpoint is a REQUIRED control on this data (arm A "
                         "sat below it under degraded refs), so having it produced by "
                         "the same walk removes any chance of a misaligned pairing.")
    args = ap.parse_args()

    if args.mask_const is not None:
        if not args.mask_in_camera:
            raise SystemExit(
                "--mask-const is the area-matched control for the camera-channel "
                "mask and does nothing without --mask-in-camera. Add it, or drop "
                "--mask-const rather than shipping an arm whose name says control "
                "and whose inputs say otherwise.")
        if args.blend_mask:
            # The blend path reads batch["mask"] directly, so a constant here
            # would leave the composite running on the REAL mask -- a half-control
            # that reads as a control. Refuse rather than mislabel it.
            raise SystemExit(
                "--mask-const with --blend-mask is not supported: the "
                "sampling-time composite reads the real mask, so the run would be "
                "a control in name only.")

    if not (args.cond_artifact or args.mask_in_camera):
        print("[infer] NOTE: neither slot enabled -- this is stock GLD generation "
              "(session 6.5's arm A), which is a legitimate arm but not GeoFix.",
              flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("inference needs a GPU; got CPU only.")

    import latent_blending as _lb
    if args.weights == "model":
        # build_context's load_ckpt does ckpt.get("ema", ckpt.get("model", ckpt)).
        # Rather than fork that function, drop the EMA key on the way past so the
        # same code path picks `model` -- one place, and it still enforces the
        # 0-missing/0-unexpected check that catches a config/checkpoint mismatch.
        _orig_load = _lb.torch.load
        def _load_no_ema(path, *a, **kw):
            d = _orig_load(path, *a, **kw)
            if isinstance(d, dict) and "model" in d:
                d = {k: v for k, v in d.items() if k != "ema"}
            return d
        _lb.torch.load = _load_no_ema
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
    # Sharding, and the GLOBAL index is what survives it. `indices` holds the real
    # dataset positions, so the per-sample seed below stays `seed + global_i` and a
    # sample lands on the same xT no matter which shard drew it. Keying the seed off
    # a shard-local counter would silently make an 8-shard run a different
    # experiment from a 1-shard run -- and the images would look perfectly fine.
    #
    # --limit is applied FIRST, so `--limit 8 --num-shards 4` is a four-way smoke
    # over 8 samples rather than four shards of the whole set (the same convention
    # geofix.score_arms uses).
    indices = list(range(n))
    if args.num_shards > 1:
        if not 0 <= args.shard < args.num_shards:
            raise ValueError(f"shard {args.shard} out of range for "
                             f"num_shards {args.num_shards}.")
        indices = indices[args.shard::args.num_shards]
    out_root = pathlib.Path(args.out)
    render_root = out_root.parent / f"{out_root.name}__render" if args.dump_render else None
    print(f"[infer] {len(indices)} of {n} samples (shard {args.shard}/{args.num_shards}), "
          f"{v} views, cond_num={cond}, cond_artifact={args.cond_artifact} "
          f"mask_in_camera={args.mask_in_camera}", flush=True)

    from utils.da3_validation_metric import get_denoised_features, decode_into_images
    from eval_gld_metric import get_cascade_features

    rae, stat_path = ctx["rae"], ctx["stat_path"]
    const_areas: list[float] = []
    written = 0
    t_start = time.time()

    skipped = 0
    processed = 0
    for done, i in enumerate(indices):
        sample = dataset.samples[i]
        split = sample["split"]                     # already "K_06/<scene>__run_XXX__it_YYYYY"
        stems = list(sample["targets"])
        if len(stems) != v - cond:
            raise ValueError(
                f"sample {i} ({split}) has {len(stems)} targets, expected {v - cond}.")

        # RESUME. The per-sample seed keys off the GLOBAL index i, not off the
        # loop counter, so skipping a finished sample leaves every remaining
        # sample's noise identical to an uninterrupted run -- the arms stay
        # paired. That is the property that makes resuming safe here.
        if sample_is_complete(out_root / split, stems):
            skipped += 1
            continue

        batch = build_batch(dataset[i], device)
        # Cheap, and it catches the double-normalisation class of bug at the one
        # place it can enter. [0,1] imagery is the contract every downstream
        # normalise() assumes; ImageNet-normalised input reads about [-2.1, 2.6].
        if processed == 0:
            lo, hi = float(batch["image"].min()), float(batch["image"].max())
            if lo < -0.01 or hi > 1.01:
                raise ValueError(
                    f"batch['image'] range [{lo:.3f}, {hi:.3f}] is not [0,1]. The "
                    "loader emits ImageNet-normalised tensors and build_batch must "
                    "undo that, exactly as cut3r_adapter.convert_cut3r_batch does.")
            print(f"[infer] image range [{lo:.3f}, {hi:.3f}] OK", flush=True)
        geo_batch = {k: batch[k] for k in ("image", "c2w", "intrinsic")}

        art = None
        if args.cond_artifact:
            if args.cond_source == "render":
                art = batch["image"]
            elif args.cond_source == "gt":
                art = batch["gt"]
            elif args.cond_source == "zeros":
                # Leave the slot at zeros, which IS stock GLD. Deliberately not a
                # zero-valued image encoded through the RAE -- that would be the
                # features of a black frame, not an empty slot.
                art = None
            elif args.cond_source == "shuffled":
                # A different sample's render, same shapes. Uses the NEXT index in the
                # dataset rather than a random one so the arm is reproducible.
                art = build_batch(dataset[(i + 1) % n], device)["image"]
        msk = batch["mask"] if args.mask_in_camera else None
        if msk is not None and msk.shape[2] != 1:
            raise ValueError(
                f"camera-channel injection takes ONE mask plane, manifest stacks "
                f"{msk.shape[2]}. Reducing several planes to one is a modelling "
                "decision; make it explicitly rather than here.")
        if msk is not None and args.mask_const is not None:
            # Placement out, area held. Recorded per sample so the provenance can
            # state the area this control actually ran at.
            msk, const_area = constant_mask(msk, args.mask_const, cond)
            const_areas.append(const_area)
            if done == 0:
                print(f"[infer] mask-const {args.mask_const!r}: uniform plane, "
                      f"mean {const_area:.4f} on target views", flush=True)

        feat, feat_denorm = {}, {}
        blend1 = None
        if args.blend_mask:
            from stage2.transport.blending import LatentBlend
            from latent_blending import encode_artifact
            # Level 1 ONLY. Session 6.5 measured compositing at level 0 as well to be
            # worth +0.100 dB for one mask and -0.268 for another, and the cascade
            # already propagates the level-1 composite -- so a second composite is a
            # moving part for no gain.
            f_art = encode_artifact(rae, batch["image"], 1, stat_path, device)
            m = batch["mask"]
            if m.shape[2] != 1:
                raise ValueError(
                    f"blend takes ONE mask plane, got {m.shape[2]}; reducing several "
                    "planes to one is a modelling decision, not this script's.")
            # (B,V,1,g,g) -> (B*V,1,g,g), matching f_art's flattened batch.
            m = m.reshape(-1, 1, m.shape[3], m.shape[4]).to(device=device,
                                                            dtype=f_art.dtype)
            blend1 = LatentBlend(total_view=v, cond_num=cond).arm(f_art, m)

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
            sampler=ctx["make_sample_fn"](blend1), loader=None, device=device,
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
            save_atomic(to_uint8(rgb[cond + k]), d / f"{stem}.png")
            written += 1
        if render_root is not None:
            dr = render_root / split
            dr.mkdir(parents=True, exist_ok=True)
            for k, stem in enumerate(stems):
                save_atomic(to_uint8(batch["image"][0, cond + k]), dr / f"{stem}.png")
        processed += 1

        if blend1 is not None and processed == 0:
            # A hook that never fires returns arm A while claiming to have blended --
            # the exact failure session 6.5 built its validation around.
            if blend1.n_calls == 0:
                raise RuntimeError(
                    "blend hook armed but n_calls == 0: the sampler never called it, "
                    "so this run is unblended generation wearing a blend's name.")
            print(f"[infer] blend hook fired {blend1.n_calls} times on sample 0",
                  flush=True)
        if (done + 1) % 10 == 0 or done + 1 == len(indices):
            el = time.time() - t_start
            print(f"[infer] {done + 1}/{len(indices)} samples, {written} frames, "
                  f"{el / (done + 1):.1f}s/sample, eta {(len(indices) - done - 1) * el / (done + 1) / 60:.1f}min",
                  flush=True)

    prov = {
        "manifest": str(args.manifest),
        "n_samples": len(indices),
        "n_samples_total": n,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "n_frames": written,
        "checkpoint_level1": str(args.checkpoint_level1),
        "checkpoint_cascade": str(args.checkpoint_cascade),
        "stats_dir": str(args.stats_dir),
        # Recorded because it CHANGES THE PIXELS: A6000 and H200 diverge
        # structurally over a 50-step bf16 ODE (max|diff| 0.93). Two arms with
        # different values here are not comparable, whatever their seeds say.
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "weights": args.weights,
        "cond_source": args.cond_source,
        "samples_skipped_complete": skipped,
        "blend_mask": bool(args.blend_mask),
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
    if args.mask_const is not None:
        # Written ONLY for a control run, so an arm produced before this flag
        # existed keeps a byte-identical provenance file -- and any arm that IS a
        # control says so, with the area it ran at, next to `gamma` and
        # `mask_polarity`. An unlabelled control is a trap.
        prov["mask_const"] = args.mask_const
        prov["mask_const_mode"] = ("per-view area match" if args.mask_const == "match"
                                   else "fixed constant (area NOT matched per sample)")
        prov["mask_const_area_mean"] = (sum(const_areas) / len(const_areas)
                                        if const_areas else None)
        prov["mask_const_area_min"] = min(const_areas) if const_areas else None
        prov["mask_const_area_max"] = max(const_areas) if const_areas else None
        prov["mask_spatial"] = "UNIFORM (area-matched control; placement removed)"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / (f"_provenance.shard{args.shard}.json" if args.num_shards > 1 else "_provenance.json")).write_text(json.dumps(prov, indent=2))
    print(f"[infer] done: {written} frames from {n} samples "
          f"({skipped} samples skipped as already complete) -> {out_root}", flush=True)
    print(f"[infer] score with: python -m geofix.baselines.flux2d --mode score "
          f"--manifest {args.manifest} --arm render --arm this={out_root} ...",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
