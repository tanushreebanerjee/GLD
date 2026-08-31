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


def assert_ref_slots_zero(msk: torch.Tensor, cond_num: int) -> None:
    """Reference slots `[0, cond_num)` must be all-zero before any mask control runs.

    Lifted out of `constant_mask` verbatim -- message included -- so the
    `--mask-transform` controls enforce the SAME invariant instead of a lookalike
    that could drift from it. Every mask control here rewrites TARGET planes only:
    the model ignores reference planes anyway (`train_multiview_da3` assigns
    `random_masks[:, cond_views:] = m_img[:, cond_views:]`), so writing them would
    be a second, unmeasured change of input -- and an area figure averaged over
    them would not be the area the network sees.
    """
    if msk[:, :cond_num].abs().max() > 0:
        raise ValueError(
            "reference slots [0, cond_num) carry a nonzero mask; GeoFixPairs "
            "emits zeros there and the area match assumes it.")


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
    assert_ref_slots_zero(msk, cond_num)
    out = torch.zeros_like(msk)
    tgt = msk[:, cond_num:]
    if const == "match":
        fill = tgt.mean(dim=(-2, -1), keepdim=True).expand_as(tgt)
    else:
        fill = torch.full_like(tgt, float(const))
    out[:, cond_num:] = fill
    return out, float(fill.mean())


#: `--mask-transform` arms. `none` is the default and MUST stay a no-op: array
#: jobs queued against this file predate the flag and have to reproduce exactly.
MASK_TRANSFORMS = ("none", "invert", "roll", "shuffle")


def roll_offset(index: int, g_h: int, g_w: int) -> tuple[int, int]:
    """Deterministic, sample-dependent, SUBSTANTIAL token-grid shift for `roll`.

    Derived from the GLOBAL sample index rather than an RNG draw, for the same
    reason the per-sample seed is: an 8-shard run must reproduce a 1-shard run,
    and a control arm that cannot be reproduced is not a control.

    The shift is confined to `[g/4, 3g/4)` on each axis rather than to `[1, g)`.
    "Non-zero" is not enough: a one-token roll is a permutation, so it passes the
    area check exactly, but it leaves a blob almost on top of itself and the arm
    would read as "placement does not matter" when placement had barely changed.
    Excluding the top quarter matters for the same reason, because the grid wraps
    -- a 35-token shift on a 36-token grid is a one-token shift the other way. At
    the 36x36 grid this is 9 to 26 tokens, i.e. at least 126 px at 504.
    """
    if g_h < 4 or g_w < 4:
        raise ValueError(f"token grid {g_h}x{g_w} is too small to roll.")
    # Knuth's multiplicative hash, so consecutive samples do not get consecutive
    # offsets and the two axes are not locked to one another.
    h = (index * 2654435761) % 4294967296
    return g_h // 4 + h % (g_h // 2), g_w // 4 + (h // 4093) % (g_w // 2)


def transform_mask(msk: torch.Tensor, transform: str, cond_num: int, index: int,
                   donor: torch.Tensor | None = None):
    """(B,V,1,g,g) `edit1` mask -> the same tensor with PLACEMENT destroyed.

    The companion to `constant_mask`, and the reason both exist: a spatially
    UNIFORM mask at M=0.78 is the best deployable configuration this project has
    measured, so "the model responds to the mask" has to be separated from "the
    model responds to how much mask there is". `constant_mask` removes placement
    by flattening the plane; these three remove it while leaving the plane's
    HISTOGRAM alone:

    - `roll`     circularly shift the plane on the TOKEN grid by `roll_offset`.
                 Area is preserved EXACTLY (a permutation of the same values),
                 so it is the clean placement control -- the mask is as
                 structured, as sparse and as severe as the real one, and only
                 points somewhere else.
    - `shuffle`  another sample's mask planes. Structure and area are both
                 plausible but unrelated to this frame's damage. NEAR
                 area-preserving, not exactly -- the donor has its own area, and
                 both are recorded so the difference cannot be mistaken for
                 placement.
    - `invert`   `1 - M`. **NOT area-preserving**: area a becomes 1 - a, so this
                 tests POLARITY sensitivity ("does the model treat 1 as repair
                 here?") and NOT placement. Do not quote it as a placement
                 control; at the areas our masks run at it also changes area by
                 more than any placement effect measured so far.

    Polarity (hard rule 7): `roll` and `shuffle` move values without touching
    them -- no sign flip anywhere. `invert` IS the flip, deliberately, and is the
    only member of this family that changes a value.

    Hard rule 6: `msk` arrives already MAX-pooled to the 36x36 token grid by
    `GeoFixPairs`, and `roll` shifts it THERE, before any upsample. Rolling a
    504x504 plane and re-pooling would smear a thin floater across a token
    boundary, which is the failure max pooling exists to prevent.

    Applied AFTER `--gamma`, i.e. to the tensor the network actually sees --
    the same convention `constant_mask` uses for its area match.

    Only slots `[cond_num, V)` are written; reference slots stay zero.
    """
    assert_ref_slots_zero(msk, cond_num)
    out = torch.zeros_like(msk)
    tgt = msk[:, cond_num:]
    area_pre = float(tgt.mean())
    info: dict = {"transform": transform}
    if transform == "invert":
        new = 1.0 - tgt
    elif transform == "roll":
        sh, sw = roll_offset(index, int(msk.shape[-2]), int(msk.shape[-1]))
        new = torch.roll(tgt, shifts=(sh, sw), dims=(-2, -1))
        info["roll"] = (sh, sw)
    elif transform == "shuffle":
        if donor is None:
            raise ValueError("--mask-transform shuffle needs a donor mask.")
        if tuple(donor.shape) != tuple(msk.shape):
            raise ValueError(
                f"donor mask shape {tuple(donor.shape)} != {tuple(msk.shape)}; a "
                "shuffle control must swap like for like or it is a shape change "
                "wearing a control's name.")
        assert_ref_slots_zero(donor, cond_num)
        new = donor[:, cond_num:].to(device=tgt.device, dtype=tgt.dtype)
    else:
        raise ValueError(
            f"unknown --mask-transform {transform!r}; expected one of "
            f"{MASK_TRANSFORMS}.")
    out[:, cond_num:] = new
    return out, area_pre, float(new.mean()), info


def to_uint8(rgb: torch.Tensor) -> np.ndarray:
    """(3, H, W) in [0, 1] -> (H, W, 3) uint8, matching the export's PNG convention."""
    x = rgb.detach().float().clamp(0, 1).cpu().numpy()
    return (np.transpose(x, (1, 2, 0)) * 255.0 + 0.5).astype(np.uint8)


def sample_is_complete(d: pathlib.Path, stems: list[str],
                       need_geom: bool = False) -> bool:
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
        # A run that also wants geometry is NOT complete just because the RGB is
        # there. Without this, adding --dump-depth to an arm that already has PNGs
        # would skip every sample and produce an empty geometry set that looks
        # finished.
        if need_geom:
            g = d / f"{stem}.geom.npz"
            if not g.is_file():
                return False
            try:
                with np.load(g) as z:
                    _ = z.files
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
    # ------------------------------------------------------------------------
    # LEVEL 0. The same two slots again, on the L1->L0 cascade.
    #
    # WHY TWO BOOLEANS AND NOT `--mask-level {1,0,both}`. An enum would couple
    # the two slots to each other and to the level: it can express "mask at
    # both" but not "render at level 1, mask at level 0", and the ablation
    # ladder this project actually runs is the CROSS PRODUCT of {slot 1, slot 2}
    # x {level 1, level 0}. Session 6.5 had to separate slot 1 from slot 2 to
    # find that slot 2 alone "says preserve while supplying nothing to
    # preserve"; collapsing the level axis into an enum would re-create exactly
    # that confound one level down. Four independent booleans, four independent
    # arms, and each one names precisely what it turns on.
    #
    # Both default OFF, so every existing command line is byte-identical.
    #
    # NOTHING HAS BEEN FINETUNED FOR THESE YET. `da3_cascade.pt` never saw a
    # filled condition slot or a graded camera channel -- see the long block
    # above `geofix_artifact_images` in src/eval_gld_metric.py. Run these
    # against a cascade finetuned with
    # gld-session7/configs/training/DA3_geofix_cascade.yaml; against the
    # released cascade they measure a distribution shift, not a conditioning
    # signal, and a null from that run must not be written down as a null about
    # level-0 conditioning (hard rule 8).
    # ------------------------------------------------------------------------
    ap.add_argument("--cond-artifact-l0", action="store_true",
                    help="LEVEL 0 slot 1: fill the cascade's cond_channel[:, "
                         "cond_num:] -- zeros in stock GLD -- with the render's "
                         "LEVEL-0 features. Independent of --cond-artifact, "
                         "which does the same thing at level 1. Requires a "
                         "cascade finetuned for it; the released da3_cascade.pt "
                         "has never seen a non-zero value there.")
    ap.add_argument("--mask-in-camera-l0", action="store_true",
                    help="LEVEL 0 slot 2: grade the cascade's camera channel 0 "
                         "by M_edit instead of the constant 1.0 on target "
                         "views. Independent of --mask-in-camera. Reads the "
                         "SAME batch['mask'] -- including any --mask-const / "
                         "--mask-transform control -- so a control arm controls "
                         "both levels rather than only one.")
    ap.add_argument("--gamma-level0", type=float, default=None,
                    help="PER-LEVEL contrast exponent for the level-0 mask. "
                         "Defaults to --gamma, i.e. the same mask both levels "
                         "get today, which makes it INERT unless set.\n"
                         "Higher gamma = smaller effective mask area (measured "
                         "on this data: 1.0 -> 0.632, 1.5 -> 0.523, 2.0 -> "
                         "0.438 mean area). Applied as a RESIDUAL exponent, "
                         "gamma_level0 / gamma, on top of the exponent "
                         "GeoFixPairs already applied at load time -- exact, "
                         "because (m**g1)**(g0/g1) == m**g0 for m >= 0, and "
                         "cheaper than loading the dataset twice.\n"
                         "NOT a FreeFix transplant. FreeFix's gamma_c schedule "
                         "is per-TIMESTEP and pre-rasterisation; it has one "
                         "latent resolution and no pyramid. Per-LEVEL gamma is "
                         "ours -- do not cite them as precedent.")
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
    ap.add_argument("--bridge-x0", action="store_true",
                    help="LATENT BRIDGE MATCHING at sampling time: start the ODE "
                         "at the artifact features instead of at N(0, I). REQUIRED "
                         "for a checkpoint trained with --geofix-bridge-x0 and "
                         "WRONG for any other -- a bridge model sampled from noise "
                         "returns a bad number rather than an error. Needs the "
                         "render, so it implies --cond-source render's artifact "
                         "images being loaded (they always are).")
    ap.add_argument("--bridge-noise-tau", type=float, default=0.0,
                    help="Start-noise scale for --bridge-x0. Set it to the value "
                         "the checkpoint trained with (geofix.bridge_noise_tau).")
    ap.add_argument("--bridge-mask-noise", action="store_true",
                    help="THE MASK-MODULATED BRIDGE at sampling time: sigma_i = "
                         "tau * M_edit_i per token, matching training. Needs "
                         "--bridge-x0, a positive --bridge-noise-tau and a mask.")
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
    ap.add_argument("--mask-transform", default="none", choices=MASK_TRANSFORMS,
                    help="CAUSAL CONTROL ON THE MASK CHANNEL, the slot-2 analogue "
                         "of --cond-source. Requires --mask-in-camera; refuses to "
                         "combine with --mask-const or --blend-mask.\n"
                         "  none    - default, a strict no-op (arms queued before "
                         "this flag existed reproduce byte-identically).\n"
                         "  invert  - 1 - M. NOT AREA-PRESERVING: area a becomes "
                         "1 - a. This is a POLARITY test ('does the model treat 1 "
                         "as repair here?'), not a placement control, and must "
                         "never be quoted as one.\n"
                         "  roll    - circularly shift the mask on the 36x36 TOKEN "
                         "grid by a fixed offset derived from the sample index "
                         "(reproducible, recorded in provenance; 9-26 tokens on "
                         "each axis, so never a near-identity). "
                         "EXACTLY area-preserving, same structure and sparsity, "
                         "pointing in the wrong place. This is THE placement "
                         "control.\n"
                         "  shuffle - another sample's mask (the NEXT dataset "
                         "index mod n, mirroring --cond-source shuffled). Near "
                         "area-preserving; both areas are recorded.\n"
                         "Read with --mask-const, not instead of it: a constant "
                         "removes structure AND placement, roll removes placement "
                         "alone.")
    ap.add_argument("--cfg-mask-mode", default="none",
                    choices=("none", "centered", "linear"),
                    help="MASK-MODULATED GUIDANCE -- use the mask at SAMPLING time "
                         "as a per-token CFG weight instead of (or as well as) "
                         "feeding it to the network.\n"
                         "  none     - default, a strict no-op.\n"
                         "  centered - scale = cfg_scale + gain*(M - mean(M)) per "
                         "view. MEAN GUIDANCE IS EXACTLY PRESERVED, so only WHERE "
                         "the guidance goes varies and the arm cannot be explained "
                         "as 'more guidance overall'. Needs --cfg-mask-gain.\n"
                         "  linear   - scale = 1 + (cfg_scale-1)*M: no guidance "
                         "where the mask is clean, full guidance where it is "
                         "damaged. NOT mean-preserving; read it beside `centered`, "
                         "never instead of it.\n"
                         "Requires --mask-in-camera (the mask reaches the guidance "
                         "term through camera channel 0) and cfg_scale > 1.0. "
                         "Works on ANY checkpoint, mask-conditioned or not, "
                         "because the mask never enters the network.")
    ap.add_argument("--cfg-mask-gain", type=float, default=0.0,
                    help="Modulation depth for --cfg-mask-mode centered. 0.0 is a "
                         "no-op, so a nonzero value is required for that mode.")
    ap.add_argument("--mask-pack", default=None, metavar="NPZ",
                    help="OVERRIDE the mask with an external pack, one 36x36 "
                         "`edit1` plane per (split, frame), written by "
                         "`geofix.blend.containment_gate pack`.\n"
                         "It exists because the masks this pipeline can consume "
                         "all come from the export tree's `<frame>.edit1.npz`, "
                         "and a mask computed AFTER an arm was generated -- the "
                         "pointwise-optimal selector between that arm and the "
                         "render -- cannot be written there without corrupting "
                         "every other arm that reads the same files. So it "
                         "arrives beside the manifest instead.\n"
                         "The pack replaces the TARGET-view planes of "
                         "batch['mask'] and nothing else; reference slots stay "
                         "zero, exactly as GeoFixPairs emits them. Polarity is "
                         "checked on load (`edit1`, 1 = take the GENERATION), so "
                         "it enters --blend-mask with no flip. Every target view "
                         "of the manifest must be present or the run is refused: "
                         "a missing plane would silently become an all-generate "
                         "mask (hard rule 8).\n"
                         "It REPLACES the manifest mask outright, so --gamma "
                         "(which GeoFixPairs applies to that mask) does not "
                         "reach a pack plane -- shape the pack when you build "
                         "it. Needs --blend-mask or --mask-in-camera to reach "
                         "the model at all, and refuses to combine with the "
                         "--mask-const / --mask-transform controls.")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Contrast exponent on the pooled mask; must match training.")
    ap.add_argument("--pooling", default="max",
                    choices=("max", "mean", "rms", "max_arearef_rms"),
                    help="How the 504x504 mask plane is reduced to the token grid. "
                         "MUST match training. 'max' is hard rule 6 and the only "
                         "mode a deployable mask may use; 'rms' and "
                         "'max_arearef_rms' are the L2-oracle pair and both read "
                         "the oracle plane, so neither is deployable.")
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
    ap.add_argument("--dump-depth", action="store_true",
                    help="Also write the decoded DEPTH, depth confidence and RAYS "
                         "per target view, as <split>/<stem>.geom.npz.\n"
                         "This is the project's stated differentiator and it is "
                         "currently thrown away: rae.decode already returns "
                         "{rgb, depth, depth_conf, ray, ray_conf} on EVERY run and "
                         "this script kept only 'rgb'. No prior refiner (Difix3D+, "
                         "SyncFix, FreeFix) emits geometry at all, so an RGB-only "
                         "evaluation competes with them on their own ground and "
                         "silently drops ours. Costs one decode's worth of nothing "
                         "-- the tensors are already in memory.")
    ap.add_argument("--dump-render", action="store_true",
                    help="Also write the unrefined render as a sibling arm. The "
                         "artifact endpoint is a REQUIRED control on this data (arm A "
                         "sat below it under degraded refs), so having it produced by "
                         "the same walk removes any chance of a misaligned pairing.")
    args = ap.parse_args()

    # `--gamma-level0` is a residual exponent on the mask, so it does nothing
    # unless the level-0 mask slot is actually on. Refusing rather than warning:
    # a run whose name says "gamma 2.0 at level 0" and whose mask never reached
    # level 0 is a mislabelled arm, which is the failure --mask-const's guards
    # already exist to prevent.
    if args.gamma_level0 is not None and not args.mask_in_camera_l0:
        raise SystemExit(
            "--gamma-level0 sets the contrast of the LEVEL-0 mask and does "
            "nothing without --mask-in-camera-l0. Add it, or drop "
            "--gamma-level0.")
    gamma_l0 = args.gamma if args.gamma_level0 is None else float(args.gamma_level0)
    if gamma_l0 == args.gamma:
        # Exactly 1.0 and taken as an explicit no-op branch downstream, so the
        # default path never applies a pow at all.
        gamma_ratio_l0 = 1.0
    else:
        if args.gamma <= 0:
            raise SystemExit(
                f"--gamma {args.gamma} is not positive, so the residual exponent "
                f"gamma_level0/gamma is undefined. A per-level gamma needs a "
                f"positive base gamma to divide out.")
        if args.mask_const is not None:
            # `constant_mask` matches area AFTER --gamma, per its own docstring.
            # Raising the matched constant to a further exponent moves the area
            # it was matched to, so the "area-matched" control would no longer be
            # area-matched at level 0 -- a control in name only, which is exactly
            # what the other guards in this block refuse.
            raise SystemExit(
                "--gamma-level0 with --mask-const is not supported: the constant "
                "mask is area-matched AFTER --gamma, and a second exponent moves "
                "that area, leaving the level-0 control matched to nothing.")
        gamma_ratio_l0 = gamma_l0 / args.gamma
        print(f"[infer] level-0 mask gamma {gamma_l0} (level-1 {args.gamma}); "
              f"residual exponent {gamma_ratio_l0:.6f}", flush=True)

    mask_pack = mask_pack_meta = None
    if args.mask_pack is not None:
        if not (args.blend_mask or args.mask_in_camera or args.mask_in_camera_l0):
            raise SystemExit(
                "--mask-pack replaces the mask, and with neither --blend-mask nor "
                "--mask-in-camera the mask never reaches the model -- the run "
                "would be plain refinement wearing a mask arm's name.")
        if args.mask_const is not None or args.mask_transform != "none":
            # Both rewrite the mask too. Layering them would leave the arm's
            # name describing one intervention and its inputs carrying two.
            raise SystemExit(
                "--mask-pack with --mask-const/--mask-transform is not "
                "supported: each of the three rewrites the mask, and stacking "
                "them makes the arm uninterpretable. Run them separately.")
        # Imported from the GeoFix package rather than reimplemented, so the
        # writer and the reader of a pack cannot drift apart (a pack read under
        # the wrong polarity preserves exactly what it meant to repair).
        try:
            from geofix.blend.containment_gate import load_pack, pack_key
        except ImportError as e:
            raise SystemExit(
                f"--mask-pack needs geofix/src on PYTHONPATH ({e}). Add "
                "`${GEOFIX_ROOT}/src` -- slurm/containment_gate.sbatch does.")
        mask_pack, mask_pack_meta = load_pack(args.mask_pack)
        print(f"[infer] mask-pack {args.mask_pack}: {len(mask_pack)} planes, "
              f"family {mask_pack_meta.get('family')}, mean M_edit "
              f"{float(np.mean(list(mask_pack.values()))):.4f}", flush=True)

    if args.mask_const is not None:
        if not (args.mask_in_camera or args.mask_in_camera_l0):
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

    # MASK-MODULATED GUIDANCE: the same three guards the other mask controls carry.
    # Every one of them turns a silently-inert flag into a refusal.
    if args.cfg_mask_mode != "none":
        # DELIBERATELY NOT requiring --mask-in-camera. Requiring it is what broke the
        # first run: it forces the mask into a network input, so on a checkpoint not
        # trained with it every arm is a train/test mismatch. The guidance weight now
        # gets its own copy of the mask and the network's input is untouched.
        if args.mask_in_camera:
            print("[cfg-mask] NOTE --mask-in-camera is also on, so the mask is BOTH a "
                  "network input and a guidance weight. That is a valid arm only on a "
                  "checkpoint trained with mask_in_camera; the parity gate checks it.",
                  flush=True)
        if args.cfg_mask_mode == "centered" and args.cfg_mask_gain == 0.0:
            raise SystemExit(
                "--cfg-mask-mode centered with --cfg-mask-gain 0.0 is EXACTLY the "
                "unmodulated arm, so it would run, finish, and be labelled as a "
                "mechanism arm while being its own control. Pass a nonzero gain.")
    elif args.cfg_mask_gain != 0.0:
        raise SystemExit(
            f"--cfg-mask-gain {args.cfg_mask_gain} without --cfg-mask-mode is inert.")

    if args.mask_transform != "none":
        if not (args.mask_in_camera or args.mask_in_camera_l0):
            raise SystemExit(
                f"--mask-transform {args.mask_transform} rewrites the mask fed to "
                "camera channel 0 and does nothing without --mask-in-camera. Add "
                "it, or drop --mask-transform rather than shipping an arm whose "
                "name says control and whose inputs say otherwise.")
        if args.mask_const is not None:
            # A constant plane has no placement left to destroy, so rolling or
            # shuffling one is indistinguishable from the constant itself -- an
            # arm that would report as two controls and act as one.
            raise SystemExit(
                f"--mask-transform {args.mask_transform} with --mask-const is "
                "contradictory: a spatially constant mask has no placement to "
                "destroy. Run them as SEPARATE arms.")
        if args.blend_mask:
            # Same failure --mask-const already refuses: the sampling-time
            # composite reads batch["mask"] directly, so the transform would
            # change only the camera channel while the blend kept running on the
            # REAL mask -- a control in name only.
            raise SystemExit(
                f"--mask-transform {args.mask_transform} with --blend-mask is not "
                "supported: the sampling-time composite reads the real mask, so "
                "the run would be a control in name only.")

    if not (args.cond_artifact or args.mask_in_camera
            or args.cond_artifact_l0 or args.mask_in_camera_l0):
        print("[infer] NOTE: no slot enabled at either level -- this is stock GLD "
              "generation (session 6.5's arm A), which is a legitimate arm but "
              "not GeoFix.", flush=True)
    if args.cond_artifact_l0 or args.mask_in_camera_l0:
        # Loud, every run, because the flags themselves cannot tell whether the
        # cascade checkpoint behind them was finetuned. Nothing on disk records
        # it either -- `da3_cascade.pt` carries only an `ema` key, no config --
        # so this print plus the provenance below is the whole audit trail.
        print("[infer] LEVEL-0 CONDITIONING ON (cond_artifact_l0="
              f"{args.cond_artifact_l0}, mask_in_camera_l0="
              f"{args.mask_in_camera_l0}). The RELEASED cascade "
              "(da3_cascade.pt) has NEVER seen either input: it trained with an "
              "all-zero condition slot and a spatially constant camera channel "
              "0. If --checkpoint-cascade is the released file, this run "
              "measures a DISTRIBUTION SHIFT, not a conditioning signal, and a "
              "null from it is not a null about level-0 conditioning. Finetune "
              "with configs/training/DA3_geofix_cascade.yaml first.",
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
        cfg_mask_mode=args.cfg_mask_mode,
        cfg_mask_gain=args.cfg_mask_gain,
        cfg_scale_cascade=args.cfg_scale_cascade,
        # The manifest already placed clean photographs in [0, cond_num); this flag
        # is build_context's own reference-swapping path, which would do it twice.
        clean_refs=None,
    ), device)

    # The cfg_scale that matters is the EFFECTIVE one: `--cfg-scale` falls back to
    # the eval config's `guidance.scale`, so checking `args.cfg_scale` at parse time
    # refuses a perfectly good run (it did, job 7381913). Check it here, where the
    # resolved value exists, and still before any sample is loaded.
    if ctx["cfg_mask_mode"] != "none":
        _cs = ctx["cfg_scale"]
        if _cs is None or _cs <= 1.0:
            raise SystemExit(
                f"--cfg-mask-mode {ctx['cfg_mask_mode']} scales the guidance term, "
                f"and the effective cfg_scale is {_cs} -- there is no guidance term "
                "to scale. Raise --cfg-scale, or the eval config's guidance.scale.")
        print(f"[cfg-mask] mode={ctx['cfg_mask_mode']} gain={ctx['cfg_mask_gain']} "
              f"on cfg_scale={_cs}", flush=True)

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
    #
    # `--pooling` exists for the L2-ORACLE PAIR ONLY (`rms` and its area control
    # `max_arearef_rms`, see `video.geofix_pairs.POOLING_MODES`). It MUST match the
    # value the arm trained with: pooling is applied at load time, so a mask
    # rms-pooled in training and max-pooled here is a different conditioning signal
    # and nothing downstream would notice. Nothing records it in the checkpoint --
    # the same parity gap the bridge_x0 gate exists to close -- so until the gate
    # covers it, this flag is the operator's responsibility.
    pooling = args.pooling
    if pooling == "max" and "max" not in str(manifest["mask_pooling"]).lower():
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
    # --mask-transform bookkeeping. Empty unless a transform arm is running, so
    # the provenance of a `none` run is unchanged.
    tf_area_pre: list[float] = []
    tf_area_post: list[float] = []
    roll_offsets: list[list[int]] = []
    donor_indices: list[list[int]] = []
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
        if sample_is_complete(out_root / split, stems,
                              need_geom=args.dump_depth):
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
        if mask_pack is not None:
            # Replace the TARGET planes of the mask in place, before anything
            # reads it -- the camera-channel path and the sampling-time composite
            # both take `batch["mask"]`, so overriding here keeps them consistent
            # instead of steering one and leaving the other on the manifest mask.
            # Reference slots keep GeoFixPairs' zeros: they are clean photographs
            # and the pack has no plane for them.
            if batch["mask"].shape[2] != 1:
                raise SystemExit(
                    f"--mask-pack carries ONE plane per view but the manifest "
                    f"stacks {batch['mask'].shape[2]}. Broadcasting one pack "
                    "plane across several mask channels is a modelling decision, "
                    "not this script's.")
            m_new = torch.zeros_like(batch["mask"])
            for k, stem in enumerate(stems):
                key = pack_key(split, stem)
                if key not in mask_pack:
                    raise SystemExit(
                        f"--mask-pack has no plane for {key}. Refusing to run: a "
                        "missing plane defaults to zeros, which under `edit1` "
                        "means KEEP THE RENDER everywhere -- a silently different "
                        "arm, not a gap (hard rule 8).")
                plane = torch.as_tensor(mask_pack[key], device=device,
                                        dtype=batch["mask"].dtype)
                if plane.shape != batch["mask"].shape[-2:]:
                    raise SystemExit(
                        f"--mask-pack plane {key} is {tuple(plane.shape)}, the "
                        f"batch grid is {tuple(batch['mask'].shape[-2:])}.")
                m_new[0, cond + k] = plane
            batch["mask"] = m_new
        geo_batch = {k: batch[k] for k in ("image", "c2w", "intrinsic")}

        # ONE source image, selected once, then routed to whichever levels are
        # switched on. Selecting it separately per level would let --cond-source
        # mean two different things in one run -- e.g. the `shuffled` negative
        # control drawing a different donor at each level, which would no longer
        # be a control of anything.
        art_img = None
        if args.cond_artifact or args.cond_artifact_l0:
            if args.cond_source == "render":
                art_img = batch["image"]
            elif args.cond_source == "gt":
                art_img = batch["gt"]
            elif args.cond_source == "zeros":
                # Leave the slot at zeros, which IS stock GLD. Deliberately not a
                # zero-valued image encoded through the RAE -- that would be the
                # features of a black frame, not an empty slot.
                art_img = None
            elif args.cond_source == "shuffled":
                # A different sample's render, same shapes. Uses the NEXT index in the
                # dataset rather than a random one so the arm is reproducible.
                art_img = build_batch(dataset[(i + 1) % n], device)["image"]
        art = art_img if args.cond_artifact else None
        art_l0 = art_img if args.cond_artifact_l0 else None
        # ONE mask tensor, controls applied ONCE, then routed per level. The
        # alternative -- transforming the level-1 copy and handing level 0 the
        # untouched manifest mask -- would make `--mask-transform shuffle` a
        # control at one level and the real thing at the other, and the arm would
        # still be called a control. With both level-0 flags off this is `msk`
        # under a different name and every existing run is unchanged.
        msk_any = (batch["mask"]
                   if (args.mask_in_camera or args.mask_in_camera_l0) else None)
        # The mask-modulated bridge needs the mask even with slot 2 OFF --
        # that combination is the arm that isolates the transport schedule
        # from the input channel.
        msk_bridge = batch["mask"] if args.bridge_mask_noise else None
        if msk_any is not None and msk_any.shape[2] != 1:
            raise ValueError(
                f"camera-channel injection takes ONE mask plane, manifest stacks "
                f"{msk_any.shape[2]}. Reducing several planes to one is a modelling "
                "decision; make it explicitly rather than here.")
        if msk_any is not None and args.mask_const is not None:
            # Placement out, area held. Recorded per sample so the provenance can
            # state the area this control actually ran at.
            msk_any, const_area = constant_mask(msk_any, args.mask_const, cond)
            const_areas.append(const_area)
            if done == 0:
                print(f"[infer] mask-const {args.mask_const!r}: uniform plane, "
                      f"mean {const_area:.4f} on target views", flush=True)

        if msk_any is not None and args.mask_transform != "none":
            # Placement destroyed, values untouched (except `invert`, which IS the
            # flip). Areas are recorded per sample on BOTH sides so a control arm
            # can never be read as a real one.
            donor = None
            if args.mask_transform == "shuffle":
                donor_i = (i + 1) % n
                if donor_i == i:
                    raise RuntimeError(
                        "--mask-transform shuffle needs at least 2 samples; with "
                        f"n={n} the donor is the sample itself, which is a no-op "
                        "wearing a control's name.")
                # Same donor rule as --cond-source shuffled: the NEXT index, not a
                # random one, so the arm is reproducible and shard-invariant.
                donor = build_batch(dataset[donor_i], device)["mask"]
                donor_indices.append([i, donor_i])
            msk_any, area_pre, area_post, tinfo = transform_mask(
                msk_any, args.mask_transform, cond, i, donor=donor)
            tf_area_pre.append(area_pre)
            tf_area_post.append(area_post)
            if "roll" in tinfo:
                roll_offsets.append([i, int(tinfo["roll"][0]), int(tinfo["roll"][1])])
            if done == 0:
                print(f"[infer] mask-transform {args.mask_transform!r}: target-view "
                      f"mean area {area_pre:.4f} -> {area_post:.4f}"
                      + (f", roll {tinfo['roll']}" if "roll" in tinfo else "")
                      + (f", donor {donor_indices[-1][1]}" if donor_indices else ""),
                      flush=True)

        # Route AFTER the controls, so both levels see the same tensor the arm's
        # name describes. `msk_any` is also what the GUIDANCE weight uses -- taken
        # here, after --mask-const / --mask-transform, so the controls apply to the
        # guidance mechanism exactly as they do to the conditioning one.
        if args.cfg_mask_mode != "none" and msk_any is None:
            raise SystemExit(
                f"--cfg-mask-mode {args.cfg_mask_mode} but this sample carries no mask. "
                "The guidance weight is built from the mask; without one the arm would "
                "run as an unmodulated no-op wearing a mechanism's name.")
        msk = msk_any if args.mask_in_camera else None
        msk_l0 = msk_any if args.mask_in_camera_l0 else None

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
            cfg_mask_mode=ctx["cfg_mask_mode"], cfg_mask_gain=ctx["cfg_mask_gain"],
            # SEPARATE from `geofix_mask`: this copy only weights the guidance term
            # and never reaches the network, so the mechanism does not require
            # --mask-in-camera and cannot create a train/test mismatch.
            cfg_mask_plane=(msk_any if ctx["cfg_mask_mode"] != "none" else None),
            cfg_uncond_mode=ctx["cfg_uncond_mode"], batch=geo_batch,
            geofix_artifact_images=art, geofix_mask=msk,
            # A bridge-trained checkpoint has to be SAMPLED as a bridge. These
            # must match the flags the checkpoint was trained with; there is no
            # way to read them off the .pt, so they are stated on the command
            # line and recorded in the provenance.
            geofix_bridge_x0=args.bridge_x0,
            geofix_bridge_noise_tau=args.bridge_noise_tau,
            geofix_bridge_mask_noise=args.bridge_mask_noise,
            geofix_bridge_mask=msk_bridge,
        )
        feat_denorm[1] = rae._denormalize(feat[1])

        # --- stage 2: L1 -> L0, the learned cascade ---------------------------
        # UNCHANGED and byte-identical to the released path unless one of the two
        # level-0 flags is set: `geofix_artifact_images=None`, `geofix_mask=None`
        # and `geofix_gamma=1.0` are the defaults inside get_cascade_features and
        # each is an explicit skip, not a multiply-by-one.
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
            geofix_artifact_images=art_l0, geofix_mask=msk_l0,
            # The RESIDUAL exponent, not the absolute one: GeoFixPairs already
            # raised this mask to `--gamma` at load time, and
            # (m ** gamma) ** (gamma_l0 / gamma) == m ** gamma_l0 exactly for
            # m >= 0. 1.0 whenever --gamma-level0 is unset or equal to --gamma,
            # which is the branch get_cascade_features skips entirely.
            geofix_gamma=gamma_ratio_l0,
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
        geom = {}
        if args.dump_depth:
            # Whatever the RAE actually emits, defensively -- which fields are
            # non-None at level 1 is a property of the decoder, not something to
            # assume. The first sample prints what it found, so the run itself
            # records the answer instead of a comment claiming one.
            for key in ("depth", "depth_conf", "ray", "ray_conf"):
                t = out.get(key, None)
                if t is None:
                    continue
                # Drop leading singleton axes BY SIZE, not by rank. The geometry
                # keys do not share a rank: depth/depth_conf are (1, V, 504, 504)
                # while ray is (1, V, 288, 288, 6) and ray_conf (1, V, 288, 288).
                # An `ndim == 5` test therefore squeezes ray and leaves depth at
                # (1, V, ...), which then trips the view check with "has 1 views,
                # expected 8". Loud, but wrong, and it would have blocked the whole
                # depth pass.
                #
                # Note also that RAY IS ON A DIFFERENT GRID: 288 = 8 * 36 tokens,
                # because the aux pyramid's last level is returned at its own scale
                # and never upsampled. Anything downstream that assumes one grid
                # for depth and rays is wrong.
                if v == 1:
                    raise RuntimeError(
                        "cannot disambiguate the batch axis from the view axis at "
                        "v == 1; --dump-depth expects the 8-view protocol.")
                while t.ndim > 1 and t.shape[0] == 1 and t.shape[0] != v:
                    t = t[0]
                if t.shape[0] != v:
                    raise RuntimeError(
                        f"decoded '{key}' has leading axis {t.shape[0]}, expected "
                        f"{v} views (full shape {tuple(t.shape)}). Do not silently "
                        "write a mis-shaped geometry tensor.")
                geom[key] = t.detach().float().cpu().numpy()
            if processed == 0:
                found = {k: tuple(t.shape) for k, t in geom.items()}
                missing = [k for k in ("depth", "depth_conf", "ray", "ray_conf")
                           if k not in geom]
                print(f"[infer] geometry emitted: {found}", flush=True)
                if missing:
                    print(f"[infer] geometry ABSENT from rae.decode: {missing}",
                          flush=True)
                if "depth" not in geom:
                    raise RuntimeError(
                        "--dump-depth given but rae.decode returned no 'depth'. "
                        "Writing rgb-only files under a geometry flag would be a "
                        "silent no-op; fix the decode level instead.")
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
            if geom:
                # Same atomic discipline as the PNGs: a shard killed mid-write must
                # not leave a truncated .npz that the resume check then accepts.
                gt_path = d / f"{stem}.geom.npz"
                tmp = gt_path.with_suffix(".npz.tmp")
                # Write through a FILE HANDLE, not a path. np.savez_compressed
                # APPENDS ".npz" to any path that does not already end in it, so
                # passing `tmp` writes "...geom.npz.tmp.npz" and the os.replace
                # below then fails on a file that was never created. Same class of
                # trap as PIL inferring its encoder from the extension (see
                # save_atomic): the library silently rewrites the name you gave it.
                with open(tmp, "wb") as fh:
                    np.savez_compressed(
                        fh, **{kk: vv[cond + k] for kk, vv in geom.items()})
                os.replace(tmp, gt_path)
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
        "dump_depth": bool(args.dump_depth),
        "blend_mask": bool(args.blend_mask),
        # WHICH mask ran. `mask_types` below names the manifest's mask, which a
        # pack overrides entirely -- without this the two arms of the containment
        # gate would be indistinguishable on disk from a plain oracle blend.
        "mask_pack": args.mask_pack,
        "mask_pack_meta": mask_pack_meta,
        # There is no way to recover these from the checkpoint file, so a
        # mismatch between how a bridge was trained and how it was sampled
        # would be unrecoverable after the fact. Record them.
        "bridge_x0": bool(args.bridge_x0),
        "bridge_noise_tau": float(args.bridge_noise_tau),
        "bridge_mask_noise": bool(args.bridge_mask_noise),
        "cond_artifact": bool(args.cond_artifact),
        "mask_in_camera": bool(args.mask_in_camera),
        # LEVEL 0. Recorded unconditionally, including the False/False default,
        # so that an arm directory can be told apart from one written before
        # these flags existed by the PRESENCE of the keys rather than by their
        # absence -- an absent key is indistinguishable from an old run, and
        # "which arm produced this directory" is the question provenance exists
        # to answer.
        "cond_artifact_l0": bool(args.cond_artifact_l0),
        "mask_in_camera_l0": bool(args.mask_in_camera_l0),
        "gamma_level0": gamma_l0,
        "gamma_level0_residual_exponent": gamma_ratio_l0,
        # The caveat, in the file, next to the numbers it qualifies. A reader
        # who finds a level-0 arm months from now needs to know whether the
        # cascade behind it was ever finetuned, and nothing in da3_cascade.pt
        # records that -- it holds an `ema` key and nothing else.
        "cascade_conditioning_note": (
            "The RELEASED da3_cascade.pt was trained with an all-zero condition "
            "slot and a spatially CONSTANT camera channel 0. If "
            "checkpoint_cascade is that file and either level-0 flag is true, "
            "this arm measures an off-distribution input, not a conditioning "
            "signal."
            if (args.cond_artifact_l0 or args.mask_in_camera_l0)
            else "level-0 conditioning off; cascade path is the released one."),
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
        # Rule 14: a conditioning knob goes in the provenance dict in the SAME
        # change that adds the knob, and is asserted at the point of USE (the
        # ValueError in get_denoised_features when CFG is off).
        "cfg_mask_mode": ctx["cfg_mask_mode"],
        "cfg_mask_gain": ctx["cfg_mask_gain"],
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
    if args.mask_transform != "none":
        # Written ONLY for a transform run, so an arm produced before this flag
        # existed keeps a byte-identical provenance file. Everything needed to
        # tell a control from a real arm is here: what was done, where it was
        # rolled to, whose mask was borrowed, and the area on both sides.
        prov["mask_transform"] = args.mask_transform
        prov["mask_transform_area_preserving"] = {
            "invert": False, "roll": True, "shuffle": "approximately",
        }[args.mask_transform]
        prov["mask_transform_note"] = {
            "invert": "1 - M. POLARITY test, NOT a placement control: area a -> 1-a.",
            "roll": ("circular shift on the TOKEN grid by a per-sample offset "
                     "from roll_offset(i, g, g), confined to [g/4, 3g/4) on each "
                     "axis (9-26 tokens at g=36, both directions, so never a "
                     "near-identity via wraparound); EXACTLY area-preserving; "
                     "this is the placement control."),
            "shuffle": ("mask of dataset sample (i + 1) % n; structure and area "
                        "plausible but unrelated to this frame's damage."),
        }[args.mask_transform]
        prov["mask_transform_area_pre_mean"] = (
            sum(tf_area_pre) / len(tf_area_pre) if tf_area_pre else None)
        prov["mask_transform_area_post_mean"] = (
            sum(tf_area_post) / len(tf_area_post) if tf_area_post else None)
        prov["mask_transform_area_pre_min"] = min(tf_area_pre) if tf_area_pre else None
        prov["mask_transform_area_pre_max"] = max(tf_area_pre) if tf_area_pre else None
        prov["mask_transform_area_post_min"] = min(tf_area_post) if tf_area_post else None
        prov["mask_transform_area_post_max"] = max(tf_area_post) if tf_area_post else None
        prov["mask_transform_n_applied"] = len(tf_area_pre)
        if args.mask_transform == "roll":
            prov["mask_roll_offsets"] = roll_offsets     # [sample_index, dy, dx]
            prov["mask_roll_offsets_key"] = "[sample_index, shift_rows, shift_cols]"
        if args.mask_transform == "shuffle":
            prov["mask_donor_indices"] = donor_indices   # [sample_index, donor_index]
            prov["mask_donor_indices_key"] = "[sample_index, donor_sample_index]"
            prov["mask_donor_rule"] = "(i + 1) % n, global dataset index"
        prov["mask_spatial"] = (
            "INVERTED (polarity control; area NOT preserved)"
            if args.mask_transform == "invert" else
            "MISPLACED (placement control; structure and area preserved)"
            if args.mask_transform == "roll" else
            "FOREIGN (placement control; another sample's mask)")
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
