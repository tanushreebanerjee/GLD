"""GeoFix pair dataset: clean references, degraded targets, and an uncertainty mask.

Session 7 steps 3 and 4. The counterpart to `geofix.train_manifest` in the GeoFix
repo, which decides *which* samples exist; this reads the manifest it wrote and
does nothing but load files. That split is GeoFix hard rule 9 -- the fork's diff
stays reviewable and rebasable, and no path, holdout rule or reference-selection
policy is stated inside upstream code.

Emits CUT3R's format (a list of per-view dicts) so `cut3r_adapter.convert_cut3r_batch`
consumes it unchanged, plus two keys of our own.

## The view ordering (step 3), and why it is asserted twice

    [0, cond_num)          CLEAN training photographs
    [cond_num, num_views)  DEGRADED artifact renders -- the views that are scored

`ref` and `tgt` are not semantic roles in GLD; they are view *indices*
(`docs/ARCH_NOTES.md`). Nothing in the model knows what a reference slot
contains, so this ordering is the entire mechanism by which clean photographs
become conditioning -- and nothing downstream would complain if it were wrong.

**`ref_view_sampling` must be `prefix`.** `_get_view_order` places references
first in every mode, but `random` and `interpolate` also *choose* which views
become references, so under either one this ordering is reshuffled and degraded
views land in reference slots. The manifest checks it against the eval config at
build time and `assert_view_config` checks it again here against the value
actually in force -- not redundant, because a copied eval yaml can change it
after the manifest was written.

## The mask (step 4)

- **MAX-pooled to the token grid, never MEAN** (GeoFix hard rule 6). Session 6
  measured this: a thin floater mean-pooled over a 14x14 patch washes out to
  near-clean. `sanity/mask_grid.png` shows it directly.
- **Polarity `edit1`**: on disk `1.0 = degraded, edit here` (hard rule 7). Read
  off the npz's own `polarity` field and checked, never inferred from the
  filename.
- **Reference views get a zero mask.** They are clean photographs; there is
  nothing to edit. Zero is not a neutral filler here, it is the correct value --
  and it is also provably inert, which is what the session-7 inertness gate
  established.

The cache stores full 504x504 uint8 planes, so pooling happens here rather than
at export: the pooling mode stays an ablation knob instead of a re-export.

## Contrast: `gamma`, then the fixed S-curve (`contrast_soft`)

Two knobs, both applied at LOAD time to the POOLED mask, in this order:

    pooled  ->  m ** gamma  ->  sigmoid(soft * (m - 0.5))  ->  the network

Order matters for the resulting AREA and not for anything else. `gamma` is the
area knob and the S-curve is centred on the FIXED midpoint 0.5, so shaping the
area first and then applying contrast about 0.5 is the composition where each
knob keeps its meaning; the reverse order makes `gamma` act on an already
S-shaped field and the recorded area figures below stop applying.

Both maps are strictly monotone on [0,1]. Three consequences worth stating
because each has bitten this project once:

- Neither can reorder tokens, so neither changes an AUROC or any rank-based
  score, and a trained network can undo either one.
- Both COMMUTE WITH MAX POOLING -- `max(f(x)) == f(max(x))` for monotone
  increasing `f` -- which is why applying them here rather than at export is
  exactly equivalent to re-exporting the cache, and why `MASK_CODE_VERSION`
  does not move. They do NOT commute with MEAN pooling, so `pooling='mean'`
  plus a contrast knob is a different mask from the one any of these numbers
  describes. `geofix/tests/test_mask_contrast.py` pins both halves.
- Neither is a per-image normalisation, and that is deliberate. A per-image
  percentile stretch would make any mask look punchy and would drive FRAME-LEVEL
  severity to exactly zero -- `geofix.masks.oracle.SpatialLPIPS` already fixes
  its squash scale for that reason ("a uniformly destroyed frame does not
  normalise to look exactly like a near-clean one"), and CLAUDE.md records
  near-severity-blindness as session 7's actionable gap. Both knobs here are
  scene-INDEPENDENT constants, so they buy contrast without spending severity.

**Why a contrast knob exists at all: area is not the problem `oracle_lpips`
has.** Measured over 144 frames spanning all six K levels
(`geofix/eval/contrast/transform_sweep.json`, 2026-08-27), MAX-pooled to 36x36:

| plane | area | std | frac >0.9 | frac <0.1 |
|---|---|---|---|---|
| `oracle_lpips` | 0.563 | 0.168 | 0.011 | 0.034 |
| `oracle_abs` | 0.266 | 0.150 | 0.008 | 0.305 |

`oracle_lpips` is a mid-grey field: 1.1% of tokens saturated and 3.4% near zero.
And SHARPENING IT IS NOT AVAILABLE -- 0.969 of its full-resolution variance is
already explained by the 36x36 block mean, because three of the five AlexNet
difference terms inside the installed `lpips` package are computed at 30x30,
coarser than our token grid, and bilinearly upsampled (`lpips/lpips.py:17-19`,
`:128`). There is nothing sub-token there to recover with a filter. The same
figure is 0.629 for `oracle_abs`, which is why the plane choice does more for
structure than any transform on this list can. See
`geofix/eval/contrast/README.md`.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

#: Matches `RAE_DA3`'s encoder buffers. The dataset emits ImageNet-normalised
#: images because that is what CUT3R emits and what `convert_cut3r_batch`
#: denormalises back to [0,1]; entering the pipeline in a different convention
#: would silently shift every image by one normalisation.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MASK_SUFFIX = ".edit1.npz"

#: How a 504x504 plane is reduced to the token grid.
#:
#:   max               hard rule 6, and the only mode any DEPLOYABLE mask may use
#:   mean              the ablation row that lets hard rule 6 be measured
#:   rms               sqrt(mean(x^2)) -- the token's own RMSE. ORACLES ONLY: it is
#:                     the aggregation an L2 criterion implies, where max is an
#:                     L-infinity one. Measured to change the selected token set far
#:                     more than the L1-vs-L2 norm does (Jaccard 0.73 vs 0.99).
#:   max_arearef_rms   THE CONTROL FOR `rms`. Max pooling, then a per-frame monotone
#:                     exponent that squashes it to exactly the area `rms` gives.
#:                     Same area, max's ranking -- so an rms-vs-this comparison is
#:                     placement, with area held fixed. Never deploy it either; it
#:                     needs the oracle plane twice.
POOLING_MODES = ("max", "mean", "rms", "max_arearef_rms")



def pool_mask(plane: np.ndarray, grid: int, mode: str = "max") -> torch.Tensor:
    """`(h, w)` or `(c, h, w)` uint8 in [0,255] -> `(c, grid, grid)` float in [0,1].

    MAX, never MEAN, in production (GeoFix hard rule 6): a thin floater
    mean-pooled over a 14x14 patch washes out to near-clean. `mode="mean"` exists
    only so that rule can be an ablation row instead of an assumption.

    `adaptive_*_pool2d` rather than a reshape so a plane whose side is not an
    exact multiple of `grid` still pools instead of raising -- at 504 and grid 36
    it is exact (14x14 per token, DA3's patch size), and the generality is free.
    """
    a = plane[None] if plane.ndim == 2 else plane
    t = torch.from_numpy(np.ascontiguousarray(a)).float().div_(255.0)
    if mode == "rms":
        # The token's own RMSE. See `geofix.masks.pooling`; ORACLES ONLY.
        return F.adaptive_avg_pool2d(t.pow(2), (grid, grid)).sqrt()
    pool = F.adaptive_max_pool2d if mode == "max" else F.adaptive_avg_pool2d
    return pool(t, (grid, grid))


#: Bisection bounds for `match_area_gamma`; mirrors `geofix.masks.pooling`.
_GAMMA_LO, _GAMMA_HI = 1e-4, 1e4


def match_area_gamma(m: torch.Tensor, target_area: float,
                     tol: float = 1e-6, iters: int = 80) -> tuple[float, bool]:
    """Solve `mean(m**g) == target_area` for `g`. Returns `(g, exact)`.

    Torch mirror of `geofix.masks.pooling.match_area_gamma` -- the fork cannot
    import the geofix package, the same reason `pool_mask` is duplicated. Keep
    the two in step; `tests/test_masks_pooling.py` checks they agree.

    THE AREA CONTROL for the rms-pooling arm. `rms <= max` pointwise, so an
    rms-pooled mask edits strictly less, and this project has measured that area
    alone moves PSNR by more than any placement effect it has ever found. A
    monotone exponent cannot reorder tokens, so squashing the MAX-pooled mask to
    the rms mask's area isolates placement from area exactly.
    """
    a = m.detach().to(torch.float64).clamp(0.0, 1.0)
    target = float(target_area)
    lo_area = float((a >= 1.0).to(torch.float64).mean())   # g -> inf
    hi_area = float((a > 0.0).to(torch.float64).mean())    # g -> 0+
    if not (lo_area < target < hi_area):
        return (_GAMMA_HI if target <= lo_area else _GAMMA_LO), False
    lo, hi = _GAMMA_LO, _GAMMA_HI
    for _ in range(iters):
        g = 0.5 * (lo + hi)
        area = float(a.pow(g).mean())
        if abs(area - target) <= tol:
            return g, True
        if area > target:      # area is DEcreasing in g
            lo = g
        else:
            hi = g
    return 0.5 * (lo + hi), True


# --- contrast: the fixed S-curve --------------------------------------------
#
# FreeFix's own curve, `third_party/freefix/recon/trainer.py:52-53`, used at
# `recon/refiner.py:279` as `soft_sigmoid(certainties - 0.5, soft=10.0)` on the
# rasterised certainty map before it reaches `fuse_latents`. Adopted rather than
# invented because it is SCENE-INDEPENDENT: the same map is applied to every
# frame, so it adds contrast without spending the frame-level severity that a
# per-image percentile stretch would destroy outright (see the module docstring,
# and `geofix.masks.oracle.SpatialLPIPS.SCALE` for the same argument made about
# the LPIPS squash).


def soft_sigmoid(x: torch.Tensor, soft: float) -> torch.Tensor:
    """FreeFix's `1 / (1 + exp(-soft * x))`, in its numerically stable form.

    `torch.sigmoid(soft * x)` is the same function; the literal expression
    overflows to `inf` in float32 once `soft * x` passes about -88, which for
    `x` in [-0.5, 0.5] happens at `soft` around 177. `1/(1+inf)` happens to
    evaluate to 0.0 rather than NaN, so the literal form would have been wrong
    only silently and only at large `soft`. `tests/test_mask_contrast.py` pins
    the two against each other over the range we would ever configure.
    """
    return torch.sigmoid(float(soft) * x)


#: Refuse steeper than this. The masks are float32, and a float32 sigmoid
#: saturates: measured on a 4001-point sweep of [0,1], `contrast_curve` has 0
#: tied neighbours at soft <= 20 and 132 at soft = 25, 605 at soft = 35 (where it
#: reaches exactly 1.0). Ties are a SILENT loss of ranking -- two tokens with
#: different damage become the same number, no shape changes, no metric reports
#: it, and the loss of ordering is invisible in every downstream score. In
#: float64 the same sweep stays strictly increasing until soft = 100, so this
#: ceiling is about the dtype the loader actually emits, not about the algebra.
#: Binarisation may well be worth trying; it should be asked for explicitly and
#: not arrive as an overflow.
MAX_CONTRAST_SOFT = 20.0


def contrast_curve(m: torch.Tensor, soft: float) -> torch.Tensor:
    """`sigmoid(soft * (m - 0.5))`: [0,1] -> [0,1], strictly increasing.

    Fixed points and limits, all of which the tests pin:

      m = 0.5      -> 0.5 exactly, for every `soft`. The midpoint is the pivot.
      soft -> 0    -> a constant 0.5 field. Every token is "half edit", which is
                      the degenerate end and the reason `contrast_soft` is
                      rejected at <= 0 rather than treated as "off".
      soft -> inf  -> a hard step at 0.5, i.e. binarisation -- but see
                      MAX_CONTRAST_SOFT: float32 gets there by SATURATING, which
                      ties tokens together rather than ordering them.

    THE ENDPOINTS ARE COMPRESSED, and this is the one behavioural surprise. The
    curve maps 0 -> sigmoid(-soft/2) and 1 -> sigmoid(soft/2), so a perfectly
    clean token no longer reads exactly 0.0: at soft=10 it reads 0.0067, at
    soft=4, 0.119. Two things follow.

    - A frame with NO mask on disk still gets exact zeros -- `_mask` returns
      `torch.zeros` before any transform runs -- while a frame WITH a mask that
      happens to be clean everywhere gets a floor of sigmoid(-soft/2). Those two
      cases are no longer identical to the network. That is a real asymmetry and
      it is why the training manifests are built with `--coverage masked`: under
      that policy every supervised target view carries a real mask, so the
      mask-absent branch is not exercised in training at all.
    - The session-7 inertness gate is untouched. It is about REFERENCE slots,
      which `__getitem__` fills with `torch.zeros(...)` directly and which never
      pass through `_mask`.
    """
    return soft_sigmoid(m - 0.5, soft)


def assert_view_config(ref_view_sampling: str, cond_num: int, manifest: dict) -> None:
    """The two step-3 assertions, against the values actually in force.

    Raises rather than warns. Both failures are silent otherwise: the model would
    train happily on degraded references and the only symptom would be a
    disappointing number.
    """
    if ref_view_sampling != "prefix":
        raise ValueError(
            f"ref_view_sampling={ref_view_sampling!r}, but this dataset puts clean "
            f"training photographs in slots [0, {cond_num}). Under 'random' or "
            "'interpolate' the loader also CHOOSES which views are references, so "
            "those photographs would not land in reference slots at all. Pin it to "
            "'prefix' (GeoFix docs/SESSION_7.md step 3).")
    if cond_num != manifest["cond_num"]:
        raise ValueError(
            f"cond_num={cond_num} but the manifest was built with "
            f"{manifest['cond_num']}. The split between clean and degraded views is "
            "baked into the manifest's per-sample `refs`/`targets` lists, so these "
            "must agree or references and targets interleave.")


class GeoFixPairs(Dataset):
    """One sample = `cond_num` clean photographs then `num_views - cond_num` renders.

    Args:
        manifest: path to `geofix.train_manifest` output.
        mask_types: which planes to stack, in order. Defaults to the manifest's
            own list, which is also what its `n_mask` was derived from -- pass a
            different one only with a matching `n_mask` on the model.
        token_grid: 504 / 14 = 36 for DA3 at our render size.
        return_gt: also load the clean ground truth of the target views. Needed
            for a training target and for scoring; never for conditioning, and
            never placed in a reference slot (that would be the oracle leak
            `geofix.blend.clean_refs` exists to rule out).
        gamma: area exponent on the pooled mask. 1.0 = untouched.
        contrast_soft: steepness of the fixed S-curve applied after `gamma`.
            `None` = no curve at all, which is NOT the same as `soft -> 0` (that
            limit is a constant 0.5 field). See `contrast_curve` and the module
            docstring for the order of operations.
    """

    def __init__(self, manifest: str | pathlib.Path, *, mask_types=None,
                 token_grid: int = 36, return_gt: bool = True,
                 gamma: float = 1.0, pooling: str = "max",
                 contrast_soft: float | None = None):
        self.manifest_path = pathlib.Path(manifest)
        m = json.loads(self.manifest_path.read_text())
        self.m = m
        self.samples = m["samples"]
        self.num_views = int(m["num_views"])
        self.cond_num = int(m["cond_num"])
        self.token_grid = int(token_grid)
        self.return_gt = bool(return_gt)
        self.mask_types = list(mask_types or m["mask_types"])
        self.gamma = float(gamma)
        # `None` means "no curve", and a non-positive `soft` is REFUSED rather
        # than read as "off": `soft = 0` is a well-defined and catastrophic
        # setting -- a constant 0.5 field, every token half-edited, the mask
        # carrying no information at all -- and a negative one inverts the
        # polarity, which hard rule 7 exists to prevent. Neither should be
        # reachable by a config typo that reads like a disable.
        if contrast_soft is not None and float(contrast_soft) <= 0.0:
            raise ValueError(
                f"contrast_soft={contrast_soft!r}: must be > 0, or None to "
                "disable. soft=0 is a CONSTANT 0.5 mask, not a no-op, and a "
                "negative soft flips the polarity (GeoFix hard rule 7).")
        if contrast_soft is not None and float(contrast_soft) > MAX_CONTRAST_SOFT:
            raise ValueError(
                f"contrast_soft={contrast_soft!r} exceeds MAX_CONTRAST_SOFT="
                f"{MAX_CONTRAST_SOFT}: in float32 the curve starts TYING tokens "
                "there (0 tied neighbours at 20, 132 at 25), which loses ranking "
                "silently. See MAX_CONTRAST_SOFT.")
        self.contrast_soft = None if contrast_soft is None else float(contrast_soft)
        if pooling not in POOLING_MODES:
            # Hard rule 6 in production. `mean` exists so the rule can be an
            # ablation row rather than an assumption; it is never the default.
            raise ValueError(
                f"pooling={pooling!r}: expected one of {sorted(POOLING_MODES)}.")
        self.pooling = pooling
        if self.contrast_soft is not None and pooling == "max_arearef_rms":
            # The area control matches `max`'s area to `rms`'s EXACTLY, per
            # frame. The S-curve is nonlinear, so applying it to both arms
            # afterwards does not preserve that equality -- the control would
            # silently stop being area-matched, and an unmatched control is not
            # a control. Refuse the combination rather than emit one.
            raise ValueError(
                "contrast_soft cannot be combined with pooling="
                "'max_arearef_rms': the S-curve is nonlinear, so it breaks the "
                "exact per-frame area match that arm's control depends on. Run "
                "the contrast arms at pooling='max'.")
        self.arearef_unmatched = 0
        self.artifact_root = pathlib.Path(m["artifact_root"])
        self.gt_root = pathlib.Path(m["gt_root"])
        self.clean_refs_root = pathlib.Path(m["clean_refs_root"])

        if m.get("mask_polarity", "").split()[0] != "edit1":
            raise ValueError(
                f"manifest polarity {m.get('mask_polarity')!r} is not edit1. This "
                "dataset assumes 1.0 = degraded, edit here (GeoFix hard rule 7).")

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def n_mask(self) -> int:
        """Channels the mask occupies. Must equal the model's `n_mask`."""
        return int(self.m["n_mask"])

    # -- loading ----------------------------------------------------------

    def _img(self, path: pathlib.Path) -> torch.Tensor:
        a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(a).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (t - mean) / std

    def _cam(self, path: pathlib.Path):
        z = np.load(path)
        pose = torch.from_numpy(np.asarray(z["pose"], dtype=np.float32))
        K = torch.from_numpy(np.asarray(z["intrinsic"], dtype=np.float32))
        return pose, K

    def _mask(self, split: str, frame: str, present: bool) -> torch.Tensor:
        """`(n_mask, g, g)` in [0,1], `edit1`. All zeros when no mask exists.

        A zero mask is the `zero_filled` coverage policy, and it is sound because
        the session-7 inertness gate proved zero mask channels reproduce the
        unmodified checkpoint bit-identically. It is not a neutral placeholder
        chosen for convenience; it is a measured no-op.
        """
        g = self.token_grid
        if not present:
            return torch.zeros(self.n_mask, g, g)
        z = np.load(self.artifact_root / split / "masks" / f"{frame}{MASK_SUFFIX}",
                    allow_pickle=True)
        pol = str(z["polarity"])
        if pol != "edit1":
            raise ValueError(
                f"{split}/{frame}: polarity {pol!r}, not 'edit1'. A flipped mask "
                "trains the model to repair the parts that are already correct "
                "(GeoFix hard rule 7).")
        if self.pooling == "max_arearef_rms":
            return self._mask_arearef(z, g)
        m = torch.cat([pool_mask(z[t], g, self.pooling) for t in self.mask_types],
                      dim=0)
        # Applied to the POOLED mask, matching how session 6.5 measured it.
        # Monotone, so it cannot reorder tokens -- only rescale the area.
        if self.gamma != 1.0:
            m = m.clamp(min=0).pow(self.gamma)
        # ... then contrast, about the fixed midpoint 0.5. Order documented in
        # the module docstring; both maps are monotone, so the token RANKING out
        # of here is the pooled mask's ranking whatever these two are set to.
        if self.contrast_soft is not None:
            m = contrast_curve(m, self.contrast_soft)
        return m

    def _mask_arearef(self, z, g: int) -> torch.Tensor:
        """MAX pooling squashed per-plane to the area `rms` pooling would give.

        The area control for the rms arm. Matching is done on the FINAL mask --
        after the config `gamma` -- because that is what the rms arm actually
        feeds the network; matching the pre-gamma planes and then applying gamma
        to both would leave a residual area gap, since `(m**a)**g` and `m_rms**g`
        do not stay matched under a further exponent.

        Per plane, not per stacked tensor: two planes with different areas would
        otherwise be matched only in aggregate.
        """
        out = []
        for t in self.mask_types:
            m_max = pool_mask(z[t], g, "max")
            m_rms = pool_mask(z[t], g, "rms")
            if self.gamma != 1.0:
                m_rms = m_rms.clamp(min=0).pow(self.gamma)
            gam, exact = match_area_gamma(m_max, float(m_rms.mean()))
            if not exact:
                self.arearef_unmatched += 1
            out.append(m_max.clamp(min=0).pow(gam))
        return torch.cat(out, dim=0)

    def __getitem__(self, i: int) -> list[dict]:
        s = self.samples[i]
        split = s["split"]
        g = self.token_grid
        views: list[dict] = []

        # Slots [0, cond_num): CLEAN training photographs, zero mask.
        ref_dir = self.clean_refs_root / split / "images_4"
        for e in s["refs"]:
            pose, K = self._cam(ref_dir / f"{e['stem']}.npz")
            v = {
                "img": self._img(ref_dir / f"{e['stem']}.png"),
                "camera_pose": pose,
                "camera_intrinsics": K,
                "idx": (i, 0, int(e["slot"])),
                "mask": torch.zeros(self.n_mask, g, g),
                "is_ref": torch.tensor(True),
            }
            if self.return_gt:
                # A clean photograph is its own ground truth. It is never scored
                # -- every GeoFix metric starts at `first_view=cond_num` -- but
                # the key has to exist for collation to stack.
                v["gt"] = v["img"]
            views.append(v)

        # Slots [cond_num, num_views): DEGRADED renders, real mask, scored.
        a_dir = self.artifact_root / split / "images_4"
        for k, (frame, has) in enumerate(zip(s["targets"], s["has_mask"])):
            pose, K = self._cam(a_dir / f"{frame}.npz")
            v = {
                "img": self._img(a_dir / f"{frame}.png"),
                "camera_pose": pose,
                "camera_intrinsics": K,
                "idx": (i, 0, self.cond_num + k),
                "mask": self._mask(split, frame, bool(has)),
                "is_ref": torch.tensor(False),
            }
            if self.return_gt:
                v["gt"] = self._img(self.gt_root / split / "images_4" / f"{frame}.png")
            views.append(v)

        if len(views) != self.num_views:
            raise RuntimeError(
                f"sample {i} ({split}) built {len(views)} views, expected "
                f"{self.num_views}: {len(s['refs'])} refs + {len(s['targets'])} targets.")
        return views

    def __getitem__(self, i: int) -> list[dict]:
        s = self.samples[i]
        split = s["split"]
        g = self.token_grid
        views: list[dict] = []

        # Slots [0, cond_num): CLEAN training photographs, zero mask.
        ref_dir = self.clean_refs_root / split / "images_4"
        for e in s["refs"]:
            pose, K = self._cam(ref_dir / f"{e['stem']}.npz")
            v = {
                "img": self._img(ref_dir / f"{e['stem']}.png"),
                "camera_pose": pose,
                "camera_intrinsics": K,
                "idx": (i, 0, int(e["slot"])),
                "mask": torch.zeros(self.n_mask, g, g),
                "is_ref": torch.tensor(True),
            }
            if self.return_gt:
                # A clean photograph is its own ground truth. It is never scored
                # -- every GeoFix metric starts at `first_view=cond_num` -- but
                # the key has to exist for collation to stack.
                v["gt"] = v["img"]
            views.append(v)

        # Slots [cond_num, num_views): DEGRADED renders, real mask, scored.
        a_dir = self.artifact_root / split / "images_4"
        for k, (frame, has) in enumerate(zip(s["targets"], s["has_mask"])):
            pose, K = self._cam(a_dir / f"{frame}.npz")
            v = {
                "img": self._img(a_dir / f"{frame}.png"),
                "camera_pose": pose,
                "camera_intrinsics": K,
                "idx": (i, 0, self.cond_num + k),
                "mask": self._mask(split, frame, bool(has)),
                "is_ref": torch.tensor(False),
            }
            if self.return_gt:
                v["gt"] = self._img(self.gt_root / split / "images_4" / f"{frame}.png")
            views.append(v)

        if len(views) != self.num_views:
            raise RuntimeError(
                f"sample {i} ({split}) built {len(views)} views, expected "
                f"{self.num_views}: {len(s['refs'])} refs + {len(s['targets'])} targets.")
        return views


def collate(batch: list[list[dict]]) -> list[dict]:
    """`[sample][view] -> [view]` with a batch dim, which is CUT3R's layout.

    `convert_cut3r_batch` indexes `cut3r_batch[view]['img']` and expects
    `(B, C, H, W)`, so the view axis stays a Python list and only the batch axis
    becomes a tensor dim.
    """
    V = len(batch[0])
    if any(len(s) != V for s in batch):
        raise RuntimeError("samples in a batch have different view counts.")
    out = []
    for v in range(V):
        d: dict = {}
        for key in batch[0][v]:
            vals = [s[v][key] for s in batch]
            d[key] = (torch.stack(vals) if torch.is_tensor(vals[0]) else vals)
        out.append(d)
    return out
