#!/usr/bin/env python
"""INERTNESS GATE for the LEVEL-0 (cascade) GeoFix conditioning slots.

    PYTHONPATH=.:src python scripts/cascade_mask_inertness_gate.py

Runs on CPU in seconds. No GPU, no checkpoint, no dataset -- see "What it stubs
and why that is sound" below for why that is a feature and not a shortcut.

## What this is the analogue of

Session 7 shipped `scripts/mask_inertness_gate.py`, which proved that widening
the level-1 embedders from `2C` to `2C + n_mask` left the released checkpoint
BIT-IDENTICAL (max_abs 0.0, 0 of 15,925,248 elements differing) so that no
session-8 number could be confounded with "my channel surgery perturbed a
175k-iteration checkpoint". This is the same argument one level down.

`get_cascade_features` in `src/eval_gld_metric.py` now takes three GeoFix
arguments (`geofix_artifact_images`, `geofix_mask`, `geofix_gamma`). Every
existing caller passes none of them, and every number already in RESULTS.md that
went through that function must stay reproducible. The gate proves it, and --
more usefully -- proves the SPECIFIC inert setting: **a mask that is all-ones on
target views reproduces the constant 1.0 the released cascade has always seen,
so the new plumbing is a no-op at that value rather than merely at `None`.**

That second property is what makes the plumbing trustworthy. `None` being inert
only says the branch was skipped. All-ones being inert says the branch runs,
writes into camera channel 0, and lands on exactly the value that was there
before -- which is the claim that has to hold if a graded mask is to be read as
"the constant, modified" rather than as "a different input entirely".

## What it does NOT prove, stated plainly

**Slot 1 has no inert setting and is not covered.** Unlike the mask, there is no
value of the render that makes filling `cond_channel[:, cond_num:]` a no-op --
those slots are ZEROS in the released cascade, and the whole point of slot 1 is
to stop them being zeros. Slot 1 is inert only when it is OFF. Case E below
therefore asserts the OPPOSITE: that switching it on changes the network input.
Both directions matter, and a gate that only ever asserted equality would pass
just as happily on plumbing that was silently dead.

**This is not evidence that level-0 conditioning HELPS.** It is evidence that it
changes nothing until it is asked to. `third_party/gld/checkpoints/da3_cascade.pt`
has never seen either input; scoring it with these flags on measures a
distribution shift. The finetune is
`gld-session7/configs/training/DA3_geofix_cascade.yaml`.

## What it stubs and why that is sound

The RAE and the diffusion sampler are stubbed, and the gate compares the TENSORS
FED TO THE NETWORK (`camera_embedding` and the concatenated `[cond | xT]`)
bit-exactly, rather than comparing decoded images.

That is stronger than an image comparison, not weaker. Determinism from
identical inputs is already established on real hardware and does not need
re-establishing here: `eval/session8/det_same_arch_a6000` re-ran the step-29500
checkpoint on an A6000 with everything else fixed and measured **max|diff|
0.000000, 0 of 76,204,800 pixels differing over 100 frames**
(`eval/session8/scores/determinism_same_arch_a6000.json`). Given that, identical
inputs imply identical outputs, and comparing inputs isolates THIS change
instead of re-measuring the sampler. It also removes the confound that would
otherwise dominate: identical inputs on an A6000 and an H200 differ by up to
max|diff| 0.9255 with 96% of pixels differing, so an image-level gate would have
to pin the architecture to say anything at all.

The stub RAE additionally records which normalisation statistics were in force
at each `encode` call, which lets case G check the trap this project named in
advance: level 0 and level 1 have DIFFERENT channel statistics, and the render
must be encoded INSIDE the target-level window or its features land in a
different space from the reference features sitting beside them in the same
tensor. That failure would not raise -- it would decode to something plausible.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


# Small on purpose. 4x4 tokens at patch 14 is a 56x56 image, which exercises the
# same nearest-upsample path `grade_camera_mask` uses at 36x36 / 504x504 (both
# are an exact 14x integer factor) at 1/81 of the cost.
B, V, COND, G, PATCH = 1, 8, 4, 4, 14
H = W = G * PATCH
C = 16


class StubRAE:
    """Just enough RAE for `get_cascade_features`, plus a normalisation tattle.

    `encode` is a fixed, seeded linear map of 14x14-average-pooled pixels, then
    the module's CURRENT `latent_mean` / `latent_var` if `do_normalization` is
    set -- which is exactly the coupling the real `RAE_DA3._normalize` has, and
    the reason the artifact encode has to happen inside the target-stats window.
    Every call appends `(level, signature-of-the-stats-in-force)` to `.calls`, so
    case G can assert that the render was encoded under LEVEL-0 statistics rather
    than under whatever the caller left on the module.
    """

    def __init__(self):
        self.encoder_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.latent_mean = None
        self.latent_var = None
        self.do_normalization = False
        self.level = 1
        self.calls: list[tuple] = []
        g = torch.Generator().manual_seed(0)
        self.W = torch.randn(C, 3, generator=g, dtype=torch.float32)

    def eval(self):
        return self

    def encode(self, x, level=None, **kw):
        sig = (None if self.latent_mean is None
               else round(float(self.latent_mean.sum()), 6))
        self.calls.append((level, self.do_normalization, sig))
        b, v = x.shape[0], x.shape[1]
        t = F.avg_pool2d(x.reshape(b * v, 3, H, W), PATCH)          # (bv, 3, G, G)
        z = torch.einsum("cd,bdhw->bchw", self.W, t)                 # (bv, C, G, G)
        if self.do_normalization and self.latent_mean is not None:
            m = self.latent_mean.reshape(1, -1, 1, 1)
            var = self.latent_var.reshape(1, -1, 1, 1)
            z = (z - m) / torch.sqrt(var + 1e-5)
        return z


class CaptureSampler:
    """Records the two tensors the cascade DiT actually consumes.

    Returns a deterministic function of its own input so that a difference in
    the inputs also shows up in the return value -- otherwise a gate could pass
    on a sampler that ignored everything it was handed.
    """

    def __init__(self):
        self.x = None
        self.cam = None

    def __call__(self, sample_input_flat, model, **kw):
        self.x = sample_input_flat.detach().clone()
        self.cam = kw["camera_embedding"].detach().clone()
        self.ref_cond = kw["ref_cond"].detach().clone()
        return [torch.tanh(sample_input_flat)]


def make_batch():
    g = torch.Generator().manual_seed(7)
    image = torch.rand(B, V, 3, H, W, generator=g)                   # [0, 1], as the loader emits
    c2w = torch.eye(4).repeat(B, V, 1, 1)
    # Spread the views so the plucker embedding is not degenerate.
    c2w[..., :3, 3] = torch.linspace(0, 1, V).view(1, V, 1)
    fx = fy = torch.full((B, V), float(W))
    cx = cy = torch.full((B, V), W / 2.0)
    intrinsic = torch.stack([fx, fy, cx, cy], dim=-1)
    return {"image": image, "c2w": c2w, "intrinsic": intrinsic}


def make_mask(target_value: float) -> torch.Tensor:
    """(B, V, 1, G, G) `edit1` mask: ZERO on references, `target_value` on targets.

    Zero on `[0, cond_num)` is not cosmetic -- `grade_camera_mask` refuses a
    non-zero reference plane, and `GeoFixPairs` emits literal zeros there because
    those slots are clean photographs with nothing to repair.
    """
    m = torch.zeros(B, V, 1, G, G)
    m[:, COND:] = target_value
    return m


def stats_file(tmp: pathlib.Path, name: str, seed: int) -> str:
    g = torch.Generator().manual_seed(seed)
    p = tmp / name
    torch.save({"mean": torch.randn(C, generator=g),
                "var": torch.rand(C, generator=g) + 0.5}, p)
    return str(p)


def run(get_cascade_features, batch, src, stat0, **geofix):
    rae, sampler = StubRAE(), CaptureSampler()
    torch.manual_seed(1234)   # sample_xT is drawn inside; hold it fixed across cases
    out = get_cascade_features(
        rae=rae, cascade_model=torch.nn.Identity(), sampler=sampler,
        source_features=src, source_stat_path=None, target_stat_path=stat0,
        batch=batch, device=torch.device("cpu"), total_view=V, cond_num=COND,
        camera_mode="plucker", use_prope=False, cfg_scale=None,
        prope_image_size=(H, W), eval_mode="cascade", **geofix)
    return out, sampler, rae


def main() -> int:
    from eval_gld_metric import get_cascade_features

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="cascade_inert_"))
    stat0 = stats_file(tmp, "normalization_stats_level0.pt", seed=11)
    batch = make_batch()
    src = torch.randn(B * V, C, G, G, generator=torch.Generator().manual_seed(3))

    results: list[tuple[str, bool, str]] = []

    def same(a, b):
        return bool(torch.equal(a, b))

    def maxabs(a, b):
        return float((a.float() - b.float()).abs().max())

    # ---- A: the baseline. Every GeoFix argument at its default. ------------
    outA, sA, raeA = run(get_cascade_features, batch, src, stat0)

    # ---- B: mask all-ones on targets. THE INERTNESS CLAIM. -----------------
    # 1.0 is exactly the constant the released cascade has always seen on target
    # views, so writing it must land on the tensor that was already there.
    outB, sB, _ = run(get_cascade_features, batch, src, stat0,
                      geofix_mask=make_mask(1.0))
    ok = same(sA.cam, sB.cam) and same(sA.x, sB.x) and same(outA, outB)
    results.append(("B  mask=1.0 is INERT (camera_embedding, [cond|xT], output)",
                    ok, f"max|d cam| {maxabs(sA.cam, sB.cam):.3e}"))

    # ---- C: same, through a non-unit gamma. --------------------------------
    # 1.0 ** gamma == 1.0 for every gamma, so a per-level contrast exponent must
    # not be able to break inertness at the inert value. This is the case that
    # would catch a gamma applied to the WHOLE plane instead of to the mask.
    outC, sC, _ = run(get_cascade_features, batch, src, stat0,
                      geofix_mask=make_mask(1.0), geofix_gamma=2.0)
    ok = same(sA.cam, sC.cam) and same(sA.x, sC.x) and same(outA, outC)
    results.append(("C  mask=1.0 INERT under gamma=2.0",
                    ok, f"max|d cam| {maxabs(sA.cam, sC.cam):.3e}"))

    # ---- D: NEGATIVE CONTROL -- a graded mask must CHANGE the input. -------
    # Without this, cases B and C would pass on plumbing that was silently dead,
    # which is the failure mode session 7 recorded as "the absence of output is
    # the measurement, which reads exactly like an absence of data".
    outD, sD, _ = run(get_cascade_features, batch, src, stat0,
                      geofix_mask=make_mask(0.5))
    ok = (not same(sA.cam, sD.cam)) and same(sA.x, sD.x)
    results.append(("D  mask=0.5 CHANGES camera_embedding and only that",
                    ok, f"max|d cam| {maxabs(sA.cam, sD.cam):.3e}, "
                        f"max|d x| {maxabs(sA.x, sD.x):.3e}"))

    # ---- E: NEGATIVE CONTROL -- slot 1 must CHANGE the condition channel. --
    # Slot 1 has NO inert value: `cond_channel[:, cond_num:]` is zeros in the
    # released cascade and filling it is the entire point. So the assertion here
    # is the opposite of B's, and the reference half must be untouched.
    outE, sE, _ = run(get_cascade_features, batch, src, stat0,
                      geofix_artifact_images=batch["image"])
    cA = sA.ref_cond.reshape(B, V, C, G, G)
    cE = sE.ref_cond.reshape(B, V, C, G, G)
    ok = (same(cA[:, :COND], cE[:, :COND])          # clean photographs preserved
          and not same(cA[:, COND:], cE[:, COND:])  # target half filled
          and float(cA[:, COND:].abs().max()) == 0.0)  # ...and it WAS zeros
    results.append(("E  cond_artifact_l0 fills targets, preserves references",
                    ok, f"ref max|d| {maxabs(cA[:, :COND], cE[:, :COND]):.3e}, "
                        f"tgt max|d| {maxabs(cA[:, COND:], cE[:, COND:]):.3e}"))

    # ---- F: gamma must be WIRED, not decorative. ---------------------------
    outF, sF, _ = run(get_cascade_features, batch, src, stat0,
                      geofix_mask=make_mask(0.5), geofix_gamma=2.0)
    ok = not same(sD.cam, sF.cam)
    results.append(("F  gamma changes a graded mask (0.5 vs 0.5**2)",
                    ok, f"max|d cam| {maxabs(sD.cam, sF.cam):.3e}"))

    # ---- G: the render is encoded under LEVEL-0 statistics. ----------------
    # The named trap. `StubRAE` records the stats in force at each encode; both
    # the reference encode and the artifact encode must report the SAME level-0
    # signature, and the module must have been restored afterwards.
    _, _, raeE = run(get_cascade_features, batch, src, stat0,
                     geofix_artifact_images=batch["image"])
    calls = raeE.calls
    ok = (len(calls) == 2
          and calls[0][0] == 0 and calls[1][0] == 0        # both at level 0
          and calls[0][1] and calls[1][1]                  # normalisation ON
          and calls[0][2] == calls[1][2]                   # the SAME stats
          and raeE.do_normalization is False)              # restored afterwards
    results.append(("G  render encoded at level 0 under the level-0 stats, "
                    "module restored", ok, f"encode calls {calls}"))

    # ---- H: references keep their ZERO in camera channel 0. ----------------
    # `random_masks[:, :cond_num] = 0` marks a view as a REFERENCE. Grading a
    # reference plane would tell the network a clean photograph is a target.
    cam = sD.cam.reshape(B, V, -1, H, W)
    ok = float(cam[:, :COND, 0].abs().max()) == 0.0
    results.append(("H  reference views keep camera channel 0 == 0",
                    ok, f"max |ch0| on refs {float(cam[:, :COND, 0].abs().max()):.3e}"))

    print()
    print("=" * 78)
    print("LEVEL-0 (cascade) GeoFix conditioning -- inertness gate")
    print("=" * 78)
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"         {detail}")
    passed = all(ok for _, ok, _ in results)
    print("-" * 78)
    print(f"  {'GATE PASS' if passed else 'GATE FAIL'}  "
          f"({sum(ok for _, ok, _ in results)}/{len(results)} checks)")
    print()
    print("  Reminder: this proves the plumbing is INERT until asked, not that")
    print("  level-0 conditioning helps. da3_cascade.pt has never seen either")
    print("  input; finetune with DA3_geofix_cascade.yaml before scoring.")
    print("=" * 78)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
