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
    pool = F.adaptive_max_pool2d if mode == "max" else F.adaptive_avg_pool2d
    return pool(t, (grid, grid))


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
    """

    def __init__(self, manifest: str | pathlib.Path, *, mask_types=None,
                 token_grid: int = 36, return_gt: bool = True,
                 gamma: float = 1.0, pooling: str = "max"):
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
        if pooling != "max":
            # Hard rule 6 in production. `mean` exists so the rule can be an
            # ablation row rather than an assumption; it is never the default.
            if pooling != "mean":
                raise ValueError(f"pooling={pooling!r}: expected 'max' or 'mean'.")
        self.pooling = pooling
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
        m = torch.cat([pool_mask(z[t], g, self.pooling) for t in self.mask_types],
                      dim=0)
        # Applied to the POOLED mask, matching how session 6.5 measured it.
        # Monotone, so it cannot reorder tokens -- only rescale the area.
        return m if self.gamma == 1.0 else m.clamp(min=0).pow(self.gamma)

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
