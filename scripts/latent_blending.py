"""Training-free mask-weighted latent blending in GLD — GeoFix session 6.5.

DIAGNOSTIC. No training, no model surgery, no patch-embedder changes: released
checkpoints only (GeoFix CLAUDE.md hard rule 2). Answers two questions that
otherwise cost sessions 7 and 8 to find out:

  1. Do our uncertainty masks carry information GLD can use?
  2. Can a blend at level 1 reach appearance at all?

Lives in the fork so it can import GLD internals directly. It adds this file
plus a `blend_fn` keyword threaded through `Sampler.sample_ode` and
`integrators.ode`; nothing else is modified (hard rule 9). The composite itself
is `stage2/transport/blending.py`, which carries the derivation.


THE ARMS

    A  none            plain GLD
    D  oracle_bin      the TRUE ceiling -- the thresholded region as a mask
       oracle_abs      the continuous oracle as stored
    B  fisher_g0/1/2   the measurement, one arm per gamma_c
       loso_logistic   the fitted combination (0.600, session 6 addendum)
    E  fisher_sched    gamma_c SCHEDULED across steps, FreeFix's design
    C  null_decoy      the control

D runs FIRST and is load-bearing. Two oracle rows because the stored
`oracle_abs` is CONTINUOUS and max-pools to a mean of 0.245 against the
thresholded region's 0.370 -- a blend driven by it composites at ~0.25 weight
and mostly PRESERVES the artifact even where the oracle knows the frame is
broken. That is the mask faithfully, but it is not "what can a blend do knowing
exactly where the damage is", and a false negative on D caused by the squash
rather than by the mechanism is the most expensive mistake available here.

Session 6 established mask quality, not the token grid, as the bottleneck --
`oracle_abs` mean-pools to AUROC 1.000 and max-pools to 0.929, so there is no
resolution ceiling at 36x36, and the best deployable single mask closes 16.6% of
the 0.501->0.929 gap. B is therefore EXPECTED to be weak, and `B ~ A` on its own
would be uninterpretable. D separates "our masks are bad" (D works, B does not)
from "the blending mechanism is dead" (D does not work either). Only the second
kills the approach, which is why D runs before anything else.


TWO LEVELS, BECAUSE ONE CANNOT BE INTERPRETED ALONE

Each blend arm runs at level 1 and at level 0. ARCH_NOTES.md establishes the
cascade is a LEARNED L1->L0 diffusion, so level 1 can in principle reach
appearance through a model -- but session 3's probe found the RGB decoder leans
almost entirely on level 0 (+6 to +14 dB) while level 1 moved RGB by under 1 dB
in isolation with the other levels pinned. That measurement does not bind the
cascade, and CLAUDE.md gates session 7 on settling it. If we blended only at
level 1 and saw nothing, "masks are uninformative" and "level 1 has no leverage
on appearance" would be indistinguishable. Level 0 is the control that separates
them, and the answer is a plumbing finding about GLD rather than about our masks.


EVERY COMPARISON IS PAIRED PER SCENE

Session 5 measured 3.71 dB of per-scene PSNR variance against the ~1.00 dB that
refinement is worth. Averaging arms independently across scenes hides a real
effect. Per-sample rows are written out and the aggregation is paired; the
initial noise is seeded per sample so every arm integrates from the SAME xT and
the difference between two arms is the blend rather than the draw.


THE SCORING REGION IS THE ORACLE'S, FOR EVERY ARM

Never each arm's own mask -- that scores the arms on different pixels and they
stop being comparable. The direct parallel of session 6 holding the labels fixed
across pooling modes. Built in `geofix.blend.prepare` and carried here.


WHAT IS DELIBERATELY NOT DONE

No `* M_alpha`. Opacity saturates on this data (alpha_mean 0.9964, AUROC 0.501),
so the multiply is inert and a null result here says nothing about it either way.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import re
import torch

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PATCH = 14

#: Arm E. Uses all three gamma_c channels under a t-keyed schedule rather than
#: one static mask, so it is armed differently from every other arm.
SCHEDULED_ARM = "fisher_sched"


def _schedule():
    """Arm E's bands as `(step_fraction, name)`. Only the NAMES are used here.

    The `t` a band starts at depends on `time_dist_shift`, which is a property of
    the loaded checkpoint, so it is resolved in `run_sample` where that is known.
    """
    from stage2.transport.blending import GAMMA_C_SCHEDULE_F
    return GAMMA_C_SCHEDULE_F


# ---------------------------------------------------------------------------
# Data: one split -> one 8-view sample
# ---------------------------------------------------------------------------


class SplitSamples:
    """The planned frames of one scene, grouped into `num_views`-view samples.

    Views are the frames session 6 cached masks for -- evenly strided across a
    split's render views -- taken in order, so the first `cond_num` are the
    reference half of the trajectory and the rest are generated. GT is used only
    for scoring.

    **What fills the reference half is a choice, not plumbing**, and both options
    are implemented:

    - `clean_refs_root=None` (the default, and what arm D and arms B/C/E ran):
      every view is an ARTIFACT render, so the references are degraded too. An
      earlier version of this docstring called that "the realistic refinement
      setting and the one session 7 targets". The second half was wrong --
      `docs/SESSION_7.md:132` asks for clean refs in `[0, cond_num)` -- and the
      first half is arguable, since the training photographs exist at test time
      by construction and both baselines we are tabled against consume them.
    - `clean_refs_root=<path>`: the reference half comes from
      `geofix.blend.clean_refs`, which crops the nearest TRAINING photographs
      through the same 504 box. The target half is untouched, so the frames being
      scored are identical to the degraded-reference run and the two are paired.

    **In clean-reference mode the reference views have different camera poses
    from anything in `gt`, so a full-frame metric over all `V` views is
    meaningless.** `masked_metrics(..., first_view=cond_num)` is the only
    admissible reduction there, and `main` refuses to emit the all-view numbers.
    That restriction is worth applying to the degraded-reference runs too: views
    below `cond_num` are never blended (`transport/blending.py:100-103`), so they
    decode identically in every arm and averaging them in halves every effect.
    """

    def __init__(
        self,
        artifact_root: pathlib.Path,
        gt_root: pathlib.Path,
        tokens: dict,
        num_views: int,
        cond_num: int,
        splits_per_scene: int | None = None,
        clean_refs_root: pathlib.Path | None = None,
    ):
        self.artifact_root = pathlib.Path(artifact_root)
        self.gt_root = pathlib.Path(gt_root)
        self.num_views = int(num_views)
        self.cond_num = int(cond_num)
        self.clean_refs_root = (
            pathlib.Path(clean_refs_root) if clean_refs_root else None)

        frame_ids = [str(f) for f in tokens["frame_id"]]
        by_split: dict[str, list[int]] = {}
        for i, fid in enumerate(frame_ids):
            level, split, _frame = fid.split("/")
            by_split.setdefault(f"{level}/{split}", []).append(i)

        self.samples = []
        for split, idx in sorted(by_split.items()):
            if len(idx) < self.num_views:
                continue  # a split with fewer cached frames than views cannot form a sample
            self.samples.append((split, idx[: self.num_views]))
        if splits_per_scene:
            self.samples = self.samples[:splits_per_scene]

        self.tokens = tokens
        self.frame_ids = frame_ids

    def __len__(self) -> int:
        return len(self.samples)

    def _frames(self, i: int) -> list[str]:
        _split, idx = self.samples[i]
        return [self.frame_ids[j] for j in idx]

    def __getitem__(self, i: int) -> dict:
        from PIL import Image

        split, idx = self.samples[i]
        level, split_name = split.split("/")

        imgs, gts, poses, intrinsics = [], [], [], []
        for j in idx:
            frame = self.frame_ids[j].split("/")[-1]
            a_dir = self.artifact_root / level / split_name
            g_dir = self.gt_root / level / split_name

            imgs.append(np.asarray(Image.open(a_dir / "images_4" / f"{frame}.png").convert("RGB")))
            gts.append(np.asarray(Image.open(g_dir / "images_4" / f"{frame}.png").convert("RGB")))
            cam = np.load(a_dir / "images_4" / f"{frame}.npz")
            poses.append(cam["pose"])
            intrinsics.append(cam["intrinsic"])

        if self.clean_refs_root is not None:
            # Overwrite ONLY the reference half. `gts` keeps the render views'
            # ground truth throughout, including for the slots just replaced --
            # those entries are now unpaired with `imgs`, which is why every
            # metric in this mode starts at `first_view=cond_num`.
            targets = [self.frame_ids[j].split("/")[-1] for j in idx[self.cond_num:]]
            self._swap_in_clean_refs(split, targets, imgs, poses, intrinsics)

        to_t = lambda a: torch.from_numpy(np.stack(a)).permute(0, 3, 1, 2).float() / 255.0  # noqa: E731
        return {
            "split": split,
            "frames": self._frames(i),
            "token_idx": np.asarray(idx),
            "image": to_t(imgs)[None],                                   # (1,V,3,H,W) artifact
            "gt": to_t(gts)[None],                                       # (1,V,3,H,W) clean
            "c2w": torch.from_numpy(np.stack(poses)).float()[None],      # (1,V,4,4)
            "intrinsic": torch.from_numpy(np.stack(intrinsics)).float()[None],
        }

    def _swap_in_clean_refs(self, split, targets, imgs, poses, intrinsics) -> None:
        """Replace views `[0, cond_num)` with the training photographs, in place.

        The pack is built by `geofix.blend.clean_refs` in our repo, where the
        camera code and `pycolmap` already live; everything that happens here is
        reading four PNGs and four poses. That split is deliberate (GeoFix hard
        rule 9 -- keep the fork diff surgical), but it means the sample grouping
        is stated in two places: `SplitSamples.__init__` above and
        `clean_refs.group_samples` there. The pack records the target frames it
        selected against, so the duplication is CHECKED here rather than trusted
        -- a drift between the two would otherwise silently pair each reference
        with a target it is not near, which looks like "clean references do not
        help" and not like a bug.
        """
        import json

        from PIL import Image   # imported per-call, as `__getitem__` does

        d = self.clean_refs_root / split
        man_path = d / "refs.json"
        if not man_path.is_file():
            raise SystemExit(
                f"{man_path} not found. Build the packs first:\n"
                "  PYTHONPATH=src python -m geofix.blend.clean_refs "
                "--config configs/data/dl3dv_sparse.yaml "
                "--eval-config configs/eval/blend_6_5.yaml "
                "--tokens eval/blend/tokens --out eval/blend/clean_refs")
        man = json.loads(man_path.read_text())

        if list(man["targets"]) != list(targets):
            raise SystemExit(
                f"{split}: the clean-reference pack was built against targets "
                f"{list(man['targets'])} but this sample's targets are {list(targets)}. "
                "The two sample groupings have drifted -- rebuild the packs with the "
                "same --eval-config this run uses.")
        if len(man["refs"]) != self.cond_num:
            raise SystemExit(
                f"{split}: pack has {len(man['refs'])} references for cond_num="
                f"{self.cond_num}.")

        for e in man["refs"]:
            j = int(e["slot"])
            stem = e["stem"]
            imgs[j] = np.asarray(Image.open(d / "images_4" / f"{stem}.png").convert("RGB"))
            cam = np.load(d / "images_4" / f"{stem}.npz")
            poses[j] = cam["pose"]
            intrinsics[j] = cam["intrinsic"]

    def masks_for(self, i: int, arm: str) -> torch.Tensor:
        """`M_edit` on the token grid for this sample's views, `(V,1,g,g)`."""
        _split, idx = self.samples[i]
        m = self.tokens[f"m__{arm}"][idx]
        return torch.from_numpy(np.asarray(m, dtype=np.float32))[:, None]

    def region_for(self, i: int) -> torch.Tensor:
        """The fixed oracle scoring region for this sample's views, `(V,1,g,g)`."""
        _split, idx = self.samples[i]
        r = self.tokens["region"][idx]
        return torch.from_numpy(np.asarray(r, dtype=np.float32))[:, None]

    def null_ok(self, i: int, threshold: float) -> bool:
        """Is every view's decoy sufficiently disjoint from the real damage?"""
        _split, idx = self.samples[i]
        return bool(np.all(self.tokens["null_overlap"][idx] <= threshold))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def score_views(pred: torch.Tensor, gt: torch.Tensor, region: torch.Tensor, ctx) -> dict:
    """Metrics on the generated half, plus the all-view numbers when they mean something.

    `*_tgt` (views `[cond_num, V)`) is always emitted and is the primary set: it
    is the only half any arm can change, and under `--clean-refs` it is the only
    half that has a ground truth at all.

    The unsuffixed all-view keys are emitted ONLY with degraded references, where
    they are well defined -- they are what arm D and arms B/C/E reported, so
    keeping them lets an old run and a new one be compared without re-running
    the old one. They are absent, rather than present-and-wrong, under
    `--clean-refs`: a reader who greps `psnr` out of a clean-reference row should
    get a `KeyError`, not a number computed against another camera's photograph.
    """
    out = {f"{k}_tgt": v for k, v in
           masked_metrics(pred, gt, region, ctx["lpips"], ctx["cond_num"]).items()}
    if not ctx["clean_refs"]:
        out.update(masked_metrics(pred, gt, region, ctx["lpips"]))
    return out


def upsample_region(region: torch.Tensor, size: int) -> torch.Tensor:
    """Token-grid region -> pixel mask, nearest, `(V,1,H,W)`."""
    return torch.nn.functional.interpolate(region, size=(size, size), mode="nearest")


def masked_metrics(pred: torch.Tensor, gt: torch.Tensor, region: torch.Tensor, lpips_fns,
                   first_view: int = 0) -> dict:
    """PSNR / SSIM / LPIPS full-frame and restricted to `region`.

    All three are reduced over the region rather than computed on a crop: SSIM
    and LPIPS both produce spatial maps, so the restriction is an average over
    the same pixels PSNR uses, and a crop would change their receptive fields.

    Every arm gets the SAME region -- the oracle's. See the module docstring.

    `first_view` drops the leading views before reducing. Pass `cond_num` to
    score only the generated half. Two independent reasons to want that:

    - Views below `cond_num` are never blended (`transport/blending.py:100-103`),
      so they decode to the same pixels in every arm. Averaging them in does not
      bias any comparison, but it halves every difference at `cond_num == V/2`,
      which is what the session-6.5 tables report.
    - Under `--clean-refs` those views are training photographs at training-view
      poses. They have no counterpart in `gt` at all, and a full-frame number
      there is not diluted but meaningless.
    """
    from utils.metrics import compute_ssim

    pred = pred[first_view:].clamp(0, 1)
    gt = gt[first_view:].clamp(0, 1)
    region = region[first_view:]
    out = {}

    se = (pred - gt) ** 2
    out["psnr"] = float((-10.0 * torch.log10(se.mean(dim=(1, 2, 3)).clamp_min(1e-12))).mean())

    w = region.expand_as(se)
    denom = w.sum(dim=(1, 2, 3)).clamp_min(1.0)
    mse_m = (se * w).sum(dim=(1, 2, 3)) / denom
    valid = region.sum(dim=(1, 2, 3)) > 0
    if valid.any():
        out["psnr_masked"] = float((-10.0 * torch.log10(mse_m[valid].clamp_min(1e-12))).mean())
    else:
        out["psnr_masked"] = float("nan")

    out["ssim"] = float(compute_ssim(gt, pred).mean())

    # `spatial` is a CONSTRUCTOR argument of lpips.LPIPS, not a forward kwarg, so
    # the scalar and the map need two instances. Both are alexnet and cheap; the
    # alternative -- taking the mean of the spatial map as the scalar -- is only
    # approximately the canonical LPIPS, because the spatial branch upsamples
    # each layer's map before summing instead of averaging it first.
    scalar_fn, spatial_fn = lpips_fns
    with torch.no_grad():
        out["lpips"] = float(scalar_fn.forward(gt * 2 - 1, pred * 2 - 1).mean())
        d_sp = spatial_fn.forward(gt * 2 - 1, pred * 2 - 1)
        rs = torch.nn.functional.interpolate(region, size=d_sp.shape[-2:], mode="nearest")
        out["lpips_masked"] = float((d_sp * rs).sum() / rs.sum().clamp_min(1.0))

    # SSIM restricted: the map, averaged over the region.
    ssim_map = _ssim_map(pred, gt)
    rs = torch.nn.functional.interpolate(region, size=ssim_map.shape[-2:], mode="nearest")
    out["ssim_masked"] = float((ssim_map * rs).sum() / rs.sum().clamp_min(1.0))
    out["region_frac"] = float(region.mean())
    return out


def _ssim_map(pred: torch.Tensor, gt: torch.Tensor, win: int = 11, sigma: float = 1.5):
    """Per-pixel SSIM, Gaussian-windowed. Averaged it equals the scalar SSIM."""
    c = pred.shape[1]
    coords = torch.arange(win, dtype=pred.dtype, device=pred.device) - (win - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    k = (g[:, None] @ g[None, :]).expand(c, 1, win, win).contiguous()
    f = lambda x: torch.nn.functional.conv2d(x, k, padding=win // 2, groups=c)  # noqa: E731

    mu1, mu2 = f(pred), f(gt)
    mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = f(pred * pred) - mu1s
    s2 = f(gt * gt) - mu2s
    s12 = f(pred * gt) - mu12
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1s + mu2s + c1) * (s1 + s2 + c2))).mean(1, keepdim=True)


# ---------------------------------------------------------------------------
# The pipeline for one sample, one arm
# ---------------------------------------------------------------------------


def encode_artifact(rae, images, level, stat_path, device):
    """Artifact frames -> `[latent_norm]` features at `level`, that level's stats.

    This is `F_artifact`, the thing the blend composites toward. It has to be
    encoded under the SAME normalisation the sampler operates in, or the
    composite mixes two different spaces and produces something plausible and
    wrong -- which is the failure mode this whole session is built to catch.
    """
    rae.level = level
    rae._init_normalization(stat_path=stat_path[level])
    images_norm = (images - rae.encoder_mean[None].to(device)) / rae.encoder_std[None].to(device)
    b, v, c, h, w = images_norm.shape
    with torch.no_grad():
        return rae.encode(images_norm.reshape(b * v, c, h, w), level=level)


def run_sample(ctx, sample, arm: str, blend_at: str, seed: int) -> dict:
    """One sample through the two-stage pipeline. Returns decoded RGB + timings.

    `blend_at` is "none", "l1", "l0" or "both". The hook object is armed for
    exactly the stages named and disarmed otherwise, so a run that reports
    `n_calls == 0` on a stage did not blend there whatever its metrics say.
    """
    from stage2.transport.blending import LatentBlend, gamma_c_schedule
    from utils.da3_validation_metric import get_denoised_features
    from eval_gld_metric import get_cascade_features

    device = ctx["device"]
    rae, stat_path = ctx["rae"], ctx["stat_path"]
    v, cond = ctx["num_views"], ctx["cond_num"]

    batch = {
        "image": sample["image"].to(device),
        "c2w": sample["c2w"].to(device),
        "intrinsic": sample["intrinsic"].to(device),
    }

    blend1 = LatentBlend(total_view=v, cond_num=cond)
    blend0 = LatentBlend(total_view=v, cond_num=cond)

    def _arm(hook, level):
        f_art = encode_artifact(rae, batch["image"], level, stat_path, device)
        if arm == SCHEDULED_ARM:
            bands = [(t_low, sample["mask_bands"][name].to(device), name)
                     for t_low, name in gamma_c_schedule(ctx["time_dist_shift"])]
            return hook.arm_schedule(f_art, bands)
        return hook.arm(f_art, sample["mask"].to(device))

    if blend_at in ("l1", "both"):
        _arm(blend1, 1)
    if blend_at in ("l0", "both"):
        _arm(blend0, 0)

    sample_fn_1 = ctx["make_sample_fn"](blend1 if blend1.enabled else None)
    sample_fn_0 = ctx["make_sample_fn"](blend0 if blend0.enabled else None)

    feat, feat_denorm = {}, {}

    # --- stage 1: noise -> L1 -------------------------------------------
    rae.level = 1
    rae._init_normalization(stat_path=stat_path[1])
    torch.manual_seed(seed)          # identical xT across arms; the paired unit
    torch.cuda.manual_seed_all(seed)
    t0 = time.time()
    feat[1] = get_denoised_features(
        rae=rae, model=ctx["model1"], transport=ctx["transport"], sampler=sample_fn_1,
        loader=None, device=device, total_view=v, cond_num=cond,
        val_num_batches=1, use_prope=ctx["use_prope"], rank=0, world_size=1,
        prope_image_size=ctx["hw"], predict_cls=False, joint_ode=False,
        ref_view_sampling=ctx["ref_view_sampling"], camera_mode=ctx["camera_mode"],
        is_concat_mode=True, pag_scale=None, pag_layer_idx=None,
        cfg_scale=ctx["cfg_scale"], use_camera_drop=ctx["use_camera_drop"],
        cfg_uncond_mode=ctx["cfg_uncond_mode"], batch=batch,
    )
    feat_denorm[1] = rae._denormalize(feat[1])
    t_l1 = time.time() - t0

    # --- stage 2: L1 -> L0 (learned cascade) -----------------------------
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 1)
    t0 = time.time()
    feat[0] = get_cascade_features(
        rae=rae, cascade_model=ctx["cascade_model"], sampler=sample_fn_0,
        source_features=feat[1], source_stat_path=stat_path[1], target_stat_path=stat_path[0],
        batch=batch, device=device, total_view=v, cond_num=cond,
        camera_mode=ctx["camera_mode"], use_prope=ctx["use_prope"],
        cfg_scale=ctx["cfg_scale_cascade"], use_camera_drop=ctx["use_camera_drop"],
        cfg_uncond_mode=ctx["cfg_uncond_mode"], noise_tau=0.0,
        prope_image_size=ctx["hw"], eval_mode="cascade",
    )
    rae.level = 0
    rae._init_normalization(stat_path=stat_path[0])
    feat_denorm[0] = rae._denormalize(feat[0])
    t_l0 = time.time() - t0

    # --- decode -----------------------------------------------------------
    from utils.da3_validation_metric import decode_into_images

    out = decode_into_images(
        rae=rae, features=feat, feat_latent_denorm=feat_denorm, level=1,
        total_view=v, cond_num=cond, device=device, batch=batch,
        stat_path=stat_path, sample_idx=0,
    )
    # `decode` returns (B, V, 3, H, W) with B=1 here. Everything downstream --
    # the metrics, the region mask, the image dump -- indexes on the VIEW axis,
    # so the batch axis is dropped once, here, rather than guessed at each use.
    # Getting this wrong does not crash in the obvious place: it broadcasts, and
    # `se.mean(dim=(1,2,3))` silently returns (1, 504) instead of (V,).
    rgb = out["rgb"]
    if rgb.ndim == 5:
        if rgb.shape[0] != 1:
            raise RuntimeError(f"expected batch 1, got {rgb.shape[0]}")
        rgb = rgb[0]
    if rgb.shape[0] != v:
        raise RuntimeError(f"decoded {rgb.shape[0]} views, expected {v}")
    return {
        "rgb": rgb,
        "n_calls_l1": blend1.n_calls,
        "n_calls_l0": blend0.n_calls,
        "bands_l1": dict(blend1.band_calls),
        "bands_l0": dict(blend0.band_calls),
        "t_l1": t_l1,
        "t_l0": t_l0,
    }


# ---------------------------------------------------------------------------
# Hook validation — runs before any arm
# ---------------------------------------------------------------------------


def validate_hook(ctx, sample, seed: int, tol_db: float) -> dict:
    """The two free tests. If either fails, the caller stops.

    `M_edit == 1` -> blend is the identity -> BIT-IDENTICAL to arm A. Catches a
    hook that fires when it should not, and catches an inverted sign.

    `M_edit == 0` -> reproduces `decode(F_artifact)` to autoencoder tolerance.
    Catches a hook that never fires -- which otherwise looks exactly like "the
    masks do nothing", the single most expensive way for this session to be
    wrong.

    Note these constants are the reverse of the ones in the first draft of the
    session-6.5 prompt. `docs/HOOKS_6_5.md` carries the derivation: the equation,
    the on-disk polarity and both of the prompt's own stated purposes agree on
    the assignment used here.

    Neither test needs a mask file or a GT frame.
    """
    v = ctx["num_views"]
    g = ctx["token_grid"]
    dev = ctx["device"]
    res = {}

    base = run_sample(ctx, {**sample, "mask": None}, "none", "none", seed)

    ones = torch.ones(v, 1, g, g, device=dev)
    ident = run_sample(ctx, {**sample, "mask": ones}, "identity", "both", seed)
    same = torch.equal(base["rgb"], ident["rgb"])
    res["identity_bit_identical"] = bool(same)
    res["identity_max_abs_diff"] = float((base["rgb"] - ident["rgb"]).abs().max())
    res["identity_n_calls_l1"] = ident["n_calls_l1"]
    res["identity_n_calls_l0"] = ident["n_calls_l0"]

    zeros = torch.zeros(v, 1, g, g, device=dev)
    art = run_sample(ctx, {**sample, "mask": zeros}, "artifact", "both", seed)
    ref = _decode_artifact(ctx, sample)
    psnr = _psnr(art["rgb"].clamp(0, 1), ref.clamp(0, 1))
    res["artifact_psnr_vs_encoded"] = psnr
    res["artifact_reproduced"] = bool(psnr >= tol_db)
    res["artifact_n_calls_l1"] = art["n_calls_l1"]
    res["artifact_n_calls_l0"] = art["n_calls_l0"]

    res["passed"] = bool(res["identity_bit_identical"] and res["artifact_reproduced"])
    return res


def _decode_artifact(ctx, sample) -> torch.Tensor:
    """`decode(F_artifact)` — the autoencoder round-trip, no diffusion involved.

    This is the reference the `M_edit == 0` test compares against, and it must be
    the round trip rather than the artifact PNG: the blend can only ever reach
    what the encoder-decoder pair can represent, so comparing to the raw image
    would fold the autoencoder's own reconstruction error into the tolerance.
    """
    from utils.da3_validation_metric import decode_into_images

    device, rae, stat_path = ctx["device"], ctx["rae"], ctx["stat_path"]
    batch = {
        "image": sample["image"].to(device),
        "c2w": sample["c2w"].to(device),
        "intrinsic": sample["intrinsic"].to(device),
    }
    feat, feat_denorm = {}, {}
    for lvl in (0, 1):
        feat[lvl] = encode_artifact(rae, batch["image"], lvl, stat_path, device)
        rae.level = lvl
        rae._init_normalization(stat_path=stat_path[lvl])
        feat_denorm[lvl] = rae._denormalize(feat[lvl])
    out = decode_into_images(
        rae=rae, features=feat, feat_latent_denorm=feat_denorm, level=1,
        total_view=ctx["num_views"], cond_num=ctx["cond_num"], device=device,
        batch=batch, stat_path=stat_path, sample_idx=0,
    )
    return out["rgb"]


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a - b) ** 2).mean()
    return float(-10.0 * torch.log10(mse.clamp_min(1e-12)))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def _payload_for(data, i: int, sample: dict, arm: str, device):
    """The sample dict `run_sample` wants for one arm, plus the mask to report.

    Shared by the arm loop and `--smoke` so the smoke test cannot exercise a
    different construction from the run it is gating.

    Returns `(payload, mask)`. `mask` is what goes in the row's `mask_mean`; for
    arm E that is the middle band, since the hook swaps masks per `t` and no
    single number describes the area it edited.
    """
    if arm == SCHEDULED_ARM:
        # Arm E carries all three gamma_c channels; the hook picks per t.
        bands = {name: data.masks_for(i, name).to(device) for _t, name in _schedule()}
        mask = bands["fisher_g1"]
        return {**sample, "mask": None, "mask_bands": bands}, mask
    mask = None if arm == "none" else data.masks_for(i, arm).to(device)
    return {**sample, "mask": mask}, mask


def _dump(img_dir, i, arm, blend_at, rgb, sample, cond_num):
    """Write one decoded target view per arm, plus the artifact and GT once.

    The FIRST TARGET view (index `cond_num`), not view 0: views below `cond_num`
    are references the blend never touches, so dumping one of those would show
    every arm producing an identical picture and prove nothing.
    """
    from PIL import Image

    def _save(name, t):
        a = (t.clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(a).save(img_dir / name)

    _save(f"s{i:03d}__{arm}_{blend_at}.png", rgb[cond_num])
    art = img_dir / f"s{i:03d}__aa_artifact.png"
    if not art.exists():
        _save(art.name, sample["image"][0][cond_num])
        _save(f"s{i:03d}__ab_gt.png", sample["gt"][0][cond_num])


def build_context(args, device) -> dict:
    from omegaconf import OmegaConf
    from stage1 import RAE_DA3  # noqa: F401  (registers the target)
    from stage2.transport import Sampler, create_transport
    from utils.config_utils import init_config_defaults
    from utils.model_utils import instantiate_from_config

    # Three configs, three different jobs -- conflating them is how the wrong
    # cond_num gets used. `eval_cfg` owns the view protocol, `model_cfg` owns
    # the model, transport, sampler and guidance.
    eval_cfg = OmegaConf.load(args.eval_config)
    model_cfg = OmegaConf.load(args.model_config_level1)

    sampling_cfg = eval_cfg.sampling
    num_views = args.num_views or int(sampling_cfg.num_views)
    cond_num = args.cond_num or int(sampling_cfg.cond_num)

    rae_config = OmegaConf.create(OmegaConf.to_container(model_cfg.stage_1, resolve=True))
    stage2_config = OmegaConf.create(OmegaConf.to_container(model_cfg.stage_2, resolve=True))
    misc_config = model_cfg.get("misc") or OmegaConf.create({})
    if "params" not in rae_config:
        rae_config.params = OmegaConf.create({})
    rae_config.params.level = 1
    init_config_defaults(rae_config, stage2_config, misc_config, patch_size=PATCH, is_da3=True)

    print("[6.5] loading RAE_DA3 ...", flush=True)
    rae = instantiate_from_config(rae_config).to(device).eval()

    def instantiate_model(path):
        """Mirror of `eval_gld_metric.instantiate_model` -- including the switch.

        `architecture_mode: "old"` means a DIFFERENT CLASS, `DDT_old`, not just a
        different branch inside `DDT`. The released cascade checkpoint is an old
        model (`DA3_cascade.yaml:33`) and the level-1 one is new
        (`DA3_level1.yaml:34`), so the two stages take different paths here.

        Instantiating `DDT` with `mode="old"` "works" -- it builds single
        embedders and aliases `_ref`/`_tgt` onto them, so the weights load into
        shared storage and the 8 missing keys are cosmetic. It is still the wrong
        class, and running an eval down an unsupported path to save one branch is
        not a trade worth making.
        """
        cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(path).stage_2, resolve=True))
        if cfg.params.get("architecture_mode", "old") == "old":
            cfg.target = "stage2.models.DDT_old.DiTwDDTHead"
            cfg.params.pop("architecture_mode", None)
            cfg.params.pop("cfg_mode", None)
        return instantiate_from_config(cfg).to(device)

    def load_ckpt(model, path, tag):
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("ema", ckpt.get("model", ckpt))
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        # DROP EXACT ALIAS DUPLICATES BEFORE LOADING -- and only those.
        #
        # `DDT_old` aliases `s_embedder_ref = s_embedder_tgt = s_embedder` (and
        # likewise for x), so `named_parameters()` dedups to 378 tensors while a
        # checkpoint SAVED from the training-side `DDT` class lists the same
        # tensors under all three names, 386 in total. Loading such a checkpoint
        # here therefore reports 8 UNEXPECTED keys -- the exact mirror of the 8
        # MISSING keys the docstring above describes for the opposite direction,
        # and equally cosmetic.
        #
        # This is NOT a relaxation of the check below. A key is dropped only when
        # its alias target is present in the same state_dict AND the two tensors
        # are bitwise equal, i.e. when dropping it provably cannot change what is
        # loaded. Anything else -- a `_ref` that differs from its base, or one
        # whose base is absent -- survives to the unexpected list and still
        # raises, because there it would mean a genuinely different architecture.
        # First needed by the level-0 cascade finetune (2026-08-28), which is the
        # first checkpoint this project has trained for an "old"-mode config.
        _alias = re.compile(r"^(?P<stem>[sx]_embedder)_(?:ref|tgt)\.(?P<rest>.+)$")
        _dropped = []
        for k in list(sd):
            m = _alias.match(k)
            if not m:
                continue
            base = f"{m.group('stem')}.{m.group('rest')}"
            if base in sd and sd[base].shape == sd[k].shape and torch.equal(sd[base], sd[k]):
                del sd[k]
                _dropped.append(k)
        if _dropped:
            print(f"[6.5] {tag}: dropped {len(_dropped)} exact alias duplicates "
                  f"(e.g. {_dropped[0]}); every one was bitwise equal to its base tensor",
                  flush=True)
        res = model.load_state_dict(sd, strict=False)
        print(f"[6.5] {tag}: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}",
              flush=True)
        # Fatal, not a warning. A released checkpoint must load completely; a
        # silently-uninitialised submodule produces a plausible image and no
        # error, which is the same failure class this whole session exists to
        # catch. The first smoke run hit exactly this -- eight missing cascade
        # embedders, from instantiating the wrong class for an "old" config.
        if res.missing_keys or res.unexpected_keys:
            for k in res.missing_keys:
                print(f"[6.5]   missing: {k}", flush=True)
            for k in res.unexpected_keys[:16]:
                print(f"[6.5]   unexpected: {k}", flush=True)
            raise RuntimeError(
                f"{tag}: checkpoint did not load cleanly "
                f"({len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected). "
                f"This is a config/checkpoint mismatch -- fix it rather than running on "
                f"partially random weights."
            )
        return model.eval()

    model1 = load_ckpt(instantiate_model(args.model_config_level1), args.checkpoint_level1, "level1")
    cascade_model = load_ckpt(
        instantiate_model(args.model_config_cascade), args.checkpoint_cascade, "cascade")

    stat_path = {i: str(pathlib.Path(args.stats_dir) / f"normalization_stats_level{i}.pt")
                 for i in range(4)}

    transport_cfg = OmegaConf.to_container(model_cfg.get("transport", {}), resolve=True)
    import math
    h = w = int(args.image_size)

    # time_dist_shift warps the ODE time grid and MUST match training, so this
    # reproduces eval_gld_metric.py:628-653 exactly rather than deriving
    # something equivalent-looking. The latent size is (C, h_lat, w_lat) --
    # channels INCLUDED -- and h_lat comes from the RAE's `encoder_input_size`,
    # not from the image size we happen to be feeding.
    #
    # THE `* num_views` BELOW IS DEAD CODE, AND THAT IS THE POINT. Both this and
    # train_multiview_da3.py:396 read `time_dist_shift_dim` with a
    # `prod(latent_size) * total_view` fallback -- but `init_config_defaults`
    # (config_utils.py:134) has already populated that key with
    # `in_channels * latent_h * latent_w`, NO view multiply, so the fallback never
    # fires in either place. Training therefore used sqrt(1536*36*36/4096) =
    # 22.045, not the 62.354 the fallback expression suggests. The fallback is
    # kept verbatim so this stays a line-for-line mirror of both call sites; do
    # not "fix" it to match the comment above it.
    #
    # This also makes the RAE config load-bearing: `encoder_input_size` 504
    # (DA3_level1.yaml) gives h_lat=36 and 22.045, while 252
    # (DA3_common_eval.yaml) gives h_lat=18 and 31.18. We take the level-1
    # training config, because matching training is the whole requirement.
    enc_in = rae_config.params.get("encoder_input_size", 504)
    enc_in = int(enc_in[0] if isinstance(enc_in, (list, tuple)) else enc_in)
    h_lat = w_lat = enc_in // PATCH
    in_channels = int(stage2_config.get("params", {}).get("in_channels", 1536))
    latent_size = (in_channels, h_lat, w_lat)

    misc = OmegaConf.to_container(misc_config, resolve=True) or {}
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size) * num_views)
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    print(f"[6.5] latent_size={latent_size} (encoder_input_size={enc_in}) "
          f"time_dist_shift = sqrt({shift_dim}/{shift_base}) = {time_dist_shift:.4f}",
          flush=True)
    if enc_in != h:
        print(f"[6.5] NOTE: encoder_input_size {enc_in} != render size {h}; the shift "
              f"follows the checkpoint's training config, which is what it must match.",
              flush=True)

    transport = create_transport(**transport_cfg.get("params", {}), time_dist_shift=time_dist_shift)
    sampler = Sampler(transport)
    sampler_params = OmegaConf.to_container(model_cfg.get("sampler", {}), resolve=True).get("params", {})

    def make_sample_fn(blend_fn):
        return sampler.sample_ode(**sampler_params, blend_fn=blend_fn)

    from lpips import LPIPS
    lpips_fns = (LPIPS(net="alex").to(device).eval(),
                 LPIPS(net="alex", spatial=True).to(device).eval())

    # Guidance comes from the model config, exactly as eval_gld_metric.py:724-736
    # reads it, so the arms run under the checkpoint's own eval settings unless
    # explicitly overridden on the command line.
    guidance = model_cfg.get("validation", {}).get("guidance", {}) or {}
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else guidance.get("scale", None)
    cfg_cascade = (args.cfg_scale_cascade if args.cfg_scale_cascade is not None else cfg_scale)

    stage2_params = stage2_config.get("params", {})
    if not bool(stage2_params.get("is_concat_mode", False)):
        raise RuntimeError(
            "is_concat_mode is false in the level-1 config. The blend assumes the "
            "[ref_cond | x_t] channel layout; with a plain state the C-channel "
            "split in LatentBlend would silently address the wrong half."
        )

    ctx_num_views = num_views
    print(f"[6.5] views={ctx_num_views} cond={cond_num} cfg={cfg_scale} "
          f"cascade_cfg={cfg_cascade}", flush=True)

    return {
        "device": device,
        "rae": rae,
        "model1": model1,
        "cascade_model": cascade_model,
        "transport": transport,
        "make_sample_fn": make_sample_fn,
        "stat_path": stat_path,
        "num_views": ctx_num_views,
        "cond_num": cond_num,
        # Arm E's band boundaries are step fractions and have to be warped
        # through this to become the `t` the hook compares against.
        "time_dist_shift": time_dist_shift,
        "hw": (h, w),
        "token_grid": h // PATCH,
        "use_prope": bool(stage2_params.get("use_prope", False)),
        "camera_mode": model_cfg.get("dataset", {}).get("camera_mode", "plucker"),
        "ref_view_sampling": sampling_cfg.get("ref_view_sampling", "prefix"),
        "cfg_scale": cfg_scale,
        # GeoFix mask-modulated guidance. `getattr` because build_context is also
        # called from latent_blending's own main() and from the masknet drivers,
        # whose argument namespaces predate these two fields; the default is the
        # strict no-op, so those callers are unaffected.
        "cfg_mask_mode": getattr(args, "cfg_mask_mode", "none"),
        "cfg_mask_gain": float(getattr(args, "cfg_mask_gain", 0.0)),
        "cfg_scale_cascade": cfg_cascade,
        "use_camera_drop": bool(guidance.get("use_camera_drop", True)),
        "cfg_uncond_mode": guidance.get("cfg_l1_uncond_mode", "keep"),
        "lpips": lpips_fns,
        "eval_cfg": eval_cfg,
        "clean_refs": bool(args.clean_refs),
    }


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-config", required=True)
    ap.add_argument("--model-config-level1", required=True)
    ap.add_argument("--model-config-cascade", required=True)
    ap.add_argument("--checkpoint-level1", required=True)
    ap.add_argument("--checkpoint-cascade", required=True)
    ap.add_argument("--stats-dir", required=True)
    ap.add_argument("--artifact-root", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--clean-refs", default=None, metavar="DIR",
                    help="fill the reference half [0, cond_num) from the TRAINING "
                         "photographs in this pack (geofix.blend.clean_refs) instead "
                         "of from artifact renders. Only `*_tgt` metrics are emitted, "
                         "since the reference views then have no counterpart in GT.")
    ap.add_argument("--tokens", required=True, help="eval/blend/tokens/<scene>.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms",
                default="oracle_bin,oracle_abs,fisher_g0,fisher_g1,fisher_g2,"
                        "loso_logistic,fisher_sched,null_decoy")
    ap.add_argument("--blend-levels", default="l1,l0")
    ap.add_argument("--oracle-both", action="store_true",
                    help="also run the oracle arm with both levels blended")
    ap.add_argument("--num-views", type=int, default=None)
    ap.add_argument("--cond-num", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=504)
    ap.add_argument("--splits-per-scene", type=int, default=None)
    ap.add_argument("--cfg-scale", type=float, default=None)
    ap.add_argument("--cfg-scale-cascade", type=float, default=None)
    ap.add_argument("--null-overlap-max", type=float, default=0.05)
    ap.add_argument("--hook-tol-db", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-images", type=int, default=0,
                    help="write decoded RGB for the first N samples, per arm, so "
                         "geofix.blend.visualise can build the side-by-side sheet. "
                         "A number that moves without a visible change is worth "
                         "distrusting, and vice versa.")
    ap.add_argument("--smoke", action="store_true",
                    help="hook validation on one sample, then stop")
    ap.add_argument("--skip-hook-validation", action="store_true",
                    help=argparse.SUPPRESS)  # for reruns of an already-validated build
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.chdir(pathlib.Path(__file__).resolve().parents[1])  # configs carry repo-relative paths

    tokens = dict(np.load(args.tokens, allow_pickle=False))
    scene = pathlib.Path(args.tokens).stem
    ctx = build_context(args, device)

    # SESSION_7.md step 3 asks for TWO assertions, not one. The ordering itself is
    # structural here -- `SplitSamples` builds the reference half as the prefix and
    # `_swap_in_clean_refs` writes into slots `[0, cond_num)`. What is not
    # structural is that the model AGREES the prefix is the reference half:
    # `_get_view_order` puts references first in every mode, but `random` and
    # `interpolate` also CHOOSE which views are references, so under either of
    # them the training photographs would land in generated slots and be scored
    # against render-view GT. This checks a config value a copied eval yaml would
    # silently change, and it is the more important of the two.
    if args.clean_refs and ctx["ref_view_sampling"] != "prefix":
        raise SystemExit(
            f"--clean-refs needs ref_view_sampling='prefix', got "
            f"{ctx['ref_view_sampling']!r}. Under {ctx['ref_view_sampling']!r} the "
            "loader also chooses which views are references, so the clean views "
            "would not occupy the reference slots they were built for.")

    data = SplitSamples(
        args.artifact_root, args.gt_root, tokens,
        ctx["num_views"], ctx["cond_num"], args.splits_per_scene,
        clean_refs_root=args.clean_refs,
    )
    print(f"[6.5] scene {scene[:12]}: {len(data)} samples of {ctx['num_views']} views", flush=True)
    if len(data) == 0:
        raise SystemExit(f"no samples for {scene}: fewer cached frames than num_views")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images" / scene
    if args.dump_images:
        img_dir.mkdir(parents=True, exist_ok=True)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    levels = [b.strip() for b in args.blend_levels.split(",") if b.strip()]

    # --- step 4: the two hook-validation tests, before any arm ----------
    # `clean_refs` goes in the report because it changes what the metric MEANS,
    # not just its value: a row set without it carries all-view keys and one with
    # it does not. A merge of the two that lost this field would silently compare
    # 8-view numbers against 4-view ones.
    report = {"scene": scene, "n_samples": len(data),
              "clean_refs": args.clean_refs, "cond_num": ctx["cond_num"],
              "scored_views": f"[{ctx['cond_num']}, {ctx['num_views']})"}
    if not args.skip_hook_validation:
        s0 = data[0]
        print("[6.5] hook validation ...", flush=True)
        hv = validate_hook(ctx, s0, args.seed, args.hook_tol_db)
        report["hook_validation"] = hv
        print(json.dumps(hv, indent=2), flush=True)
        (out_dir / f"{scene}.hook.json").write_text(json.dumps(hv, indent=2))
        if not hv["passed"]:
            raise SystemExit(
                "HOOK VALIDATION FAILED -- stopping before any arm.\n"
                f"  identity bit-identical: {hv['identity_bit_identical']} "
                f"(max abs diff {hv['identity_max_abs_diff']:.3e})\n"
                f"  artifact reproduced:    {hv['artifact_reproduced']} "
                f"({hv['artifact_psnr_vs_encoded']:.2f} dB vs {args.hook_tol_db} required)"
            )
        print("[6.5] hook validation PASSED", flush=True)
    if args.smoke:
        # Run ONE blended arm all the way through the metrics before stopping.
        # Hook validation alone does not touch `masked_metrics`, and the first
        # array submission died there on a shape it had never been handed --
        # a smoke test that skips the scoring path is not a smoke test.
        # The FIRST REQUESTED arm, not a fixed one: each submission brings arms
        # whose mask construction differs (arm E assembles three gamma_c bands
        # and hands the hook a schedule instead of a mask), and a smoke test
        # pinned to `oracle_bin` would never enter the path the batch is about
        # to run 20 times.
        smoke_arm = arms[0]
        smoke_level = levels[0]
        s0 = data[0]
        region = upsample_region(data.region_for(0), args.image_size).to(device)
        payload, _mask = _payload_for(data, 0, s0, smoke_arm, device)
        res = run_sample(ctx, payload, smoke_arm, smoke_level, args.seed)
        m = score_views(res["rgb"].to(device), s0["gt"][0].to(device), region, ctx)
        report["smoke_metrics"] = {k: round(v, 6) for k, v in m.items()}
        report["smoke_arm"] = f"{smoke_arm}/{smoke_level}"
        report["smoke_bands"] = res["bands_l1"] or res["bands_l0"]
        print(f"[6.5] smoke metrics ({smoke_arm}/{smoke_level}): "
              + json.dumps(report["smoke_metrics"]), flush=True)
        print(f"[6.5] smoke calls={res['n_calls_l1']}/{res['n_calls_l0']} "
              f"bands={json.dumps(report['smoke_bands'])}", flush=True)
        (out_dir / f"{scene}.smoke.json").write_text(json.dumps(report, indent=2))
        return 0

    # --- the arms. D first; it is the load-bearing one. ------------------
    plan = [("none", "none")]
    for arm in arms:
        for lv in levels:
            plan.append((arm, lv))
        if arm in ("oracle_abs", "oracle_bin") and args.oracle_both:
            plan.append((arm, "both"))

    rows = []
    for i in range(len(data)):
        sample = data[i]
        region = upsample_region(data.region_for(i), args.image_size).to(device)
        gt = sample["gt"][0].to(device)
        seed = args.seed + 1000 * i

        for arm, blend_at in plan:
            if arm == "null_decoy" and not data.null_ok(i, args.null_overlap_max):
                continue  # a decoy overlapping the real damage is not a null
            payload, mask = _payload_for(data, i, sample, arm, device)
            t0 = time.time()
            res = run_sample(ctx, payload, arm, blend_at, seed)
            m = score_views(res["rgb"].to(device), gt, region, ctx)
            if i < args.dump_images:
                _dump(img_dir, i, arm, blend_at, res["rgb"], sample, ctx["cond_num"])
            rows.append({
                "scene": scene, "split": sample["split"], "sample": i,
                "arm": arm, "blend_at": blend_at,
                # Per ROW, not just per report: these jsonl files get merged
                # (`report --in A --in B`), and a clean-reference row and a
                # degraded-reference row are not the same measurement.
                "clean_refs": bool(args.clean_refs),
                # How much of the frame this arm actually generates over. Not a
                # metric -- a confound, and its SIGN is counter-intuitive. Arm A
                # (pure generation, M == 1) scores BELOW the artifact renders
                # (15.605 vs 16.189 dB), so a LOW mask mean preserves more
                # artifact and scores higher for free. `null_decoy` max-pools to
                # 0.707 against `oracle_bin`'s 0.370: arm C generates over twice
                # the area, which pushes it DOWN, so C is a weaker control than
                # it looks -- the risk is "C loses for free", not "C wins". By
                # the same token `fisher_g2` (0.062) barely edits and will look
                # good on raw delta-vs-A for no other reason. `chord_excess` in
                # geofix.blend.report is what removes this; rank arms there.
                # Recorded per row so the caveat travels with the number.
                "mask_mean": None if mask is None else round(float(mask.mean()), 6),
                "n_calls_l1": res["n_calls_l1"], "n_calls_l0": res["n_calls_l0"],
                "bands_l1": res["bands_l1"], "bands_l0": res["bands_l0"],
                "wall_s": round(time.time() - t0, 2),
                **{k: round(v, 6) for k, v in m.items()},
            })
            print(f"[6.5] {scene[:8]} s{i:03d} {arm:>14s}/{blend_at:<4s} "
                  f"psnr_tgt={m['psnr_tgt']:.3f} masked_tgt={m['psnr_masked_tgt']:.3f} "
                  f"lpips_tgt={m['lpips_tgt']:.4f} "
                  f"calls={res['n_calls_l1']}/{res['n_calls_l0']}",
                  flush=True)

        with (out_dir / f"{scene}.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    report["rows"] = len(rows)
    (out_dir / f"{scene}.json").write_text(json.dumps(report, indent=2))
    print(f"[6.5] {scene[:12]} done: {len(rows)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
