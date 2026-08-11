"""Session 7 step 6, THE DELIVERABLE: the widened model must be inert at zero-init.

With `n_mask` extra input channels added and their weights zeroed, the model must
reproduce the unmodified released checkpoint on the same input. If it does not,
every later ablation is confounded: a "conditioning helps" result could be the
surgery, not the mask.

## What is actually asserted, and why it is stronger than the spec asks

SESSION_7.md asks for equality when the mask channels are zero. This feeds a
**uniform-random mask in [0, 1)** instead, because the weights are zero -- so
*any* mask value must give the identical answer. That distinction is not
cosmetic. A zeros-mask passes even if the padding went to the FRONT of the input
channel dim, which would shift every pre-trained channel by `n_mask` and silently
destroy the model; a random mask does not. The failure this catches is exactly
the one `mask_conditioning`'s docstring warns about.

Three assertions per model:

  1. WEIGHTS   -- pre-trained channels survive `expand_state_dict` bit-exactly,
                  mask channels are exactly zero, and the load reports no
                  missing/unexpected keys.
  2. FORWARD   -- base(x) == widened(cat([x, random_mask])), where the tolerance
                  actually observed is recorded rather than chosen in advance.
  3. CAMERA    -- `camera_embedder` is untouched (7 in-channels, bit-identical
                  weights) and its token output matches. The camera path is a
                  separate embedder that widening must not reach, and
                  SESSION_7.md calls it out by name.

Determinism per the spec: `torch.use_deterministic_algorithms(True)`,
`cudnn.benchmark = False`, fixed seed, `torch.no_grad`, `.eval()`. The two
forwards run in the same process on the same device so nothing but the surgery
differs.

## The two models are different CLASSES

    level-1   configs/training/DA3_level1.yaml   architecture_mode "new"   DDT.DiTwDDTHead
    cascade   configs/training/DA3_cascade.yaml  architecture_mode "old"   DDT_old.DiTwDDTHead

`latent_blending.py:707` rewrites an "old" config's target, so the cascade does
NOT run DDT.py's own "old" branch. This script mirrors that switch instead of
reimplementing it -- running the gate down a path eval never takes would prove
nothing about eval.

Usage:
    python scripts/mask_inertness_gate.py --checkpoint-dir <dir> [--n-mask 1] [--device cuda]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from stage2.models import mask_conditioning as mc  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402


# Mirrors configs/eval/blend_6_5.yaml -- the protocol session 6.5 ran and session
# 7 inherits. Not arbitrary: cond_num=4 exercises the ref/tgt view split, which is
# the branch the "new" architecture's four embedders live behind.
TOTAL_VIEW = 8
COND_NUM = 4
IMAGE_SIZE = 504
PATCH = 14
GRID = IMAGE_SIZE // PATCH          # 36
N_TOKENS = GRID * GRID              # 1296


def build(config_path: str, n_mask: int, device: str):
    """Instantiate a stage-2 model, mirroring `latent_blending.instantiate_model`.

    The `architecture_mode: "old"` -> `DDT_old` rewrite is copied deliberately,
    including popping the two keys that class does not accept.
    """
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(config_path).stage_2, resolve=True))
    if cfg.params.get("architecture_mode", "old") == "old":
        cfg.target = "stage2.models.DDT_old.DiTwDDTHead"
        cfg.params.pop("architecture_mode", None)
        cfg.params.pop("cfg_mode", None)
    if n_mask:
        cfg.params.n_mask = n_mask
    return instantiate_from_config(cfg).to(device).eval()


def load_ckpt(model, path, *, expand: bool):
    """Load a released checkpoint, optionally zero-padding it into a widened model.

    Fatal on any missing/unexpected key, matching `latent_blending.load_ckpt`. A
    partially-initialised model is the exact failure this gate exists to rule out,
    so it must not be reachable from inside the gate either.
    """
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("ema", ckpt.get("model", ckpt))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    if expand:
        sd = mc.expand_state_dict(sd, model)
    res = model.load_state_dict(sd, strict=False)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(
            f"{path}: missing={res.missing_keys[:8]} unexpected={res.unexpected_keys[:8]}"
        )
    return model


def make_cameras(device, dtype, *, n_views, image_size, radius=2.0, arc=0.6):
    """`viewmats` (w2c, B V 4 4) and `Ks` (B V 3 3) for ProPE, as a small camera arc.

    Both shipped configs set `use_prope: true`, and ProPE's attention asserts these
    are present (`model_utils.py:454`) -- a forward without them does not run.

    The arc is DELIBERATELY not degenerate. Identical (or identity) poses across
    views would satisfy the assert while collapsing ProPE's relative-pose term to a
    constant, which is precisely the sort of geometry that could mask a bug in the
    view-split path. Cameras look inward from a radius-2 arc, matching the scale
    training normalizes to.
    """
    idx = torch.arange(n_views, dtype=torch.float64)
    theta = (idx - (n_views - 1) / 2) * (arc / max(n_views - 1, 1))
    c2w = torch.eye(4, dtype=torch.float64).repeat(n_views, 1, 1)
    cos, sin = torch.cos(theta), torch.sin(theta)
    c2w[:, 0, 0], c2w[:, 0, 2] = cos, sin        # rotation about +Y
    c2w[:, 2, 0], c2w[:, 2, 2] = -sin, cos
    c2w[:, 0, 3], c2w[:, 2, 3] = radius * sin, radius * cos
    viewmats = torch.linalg.inv(c2w).unsqueeze(0)

    f = 0.7 * image_size
    K = torch.eye(3, dtype=torch.float64)
    K[0, 0] = K[1, 1] = f
    K[0, 2] = K[1, 2] = image_size / 2
    Ks = K.repeat(n_views, 1, 1).unsqueeze(0)

    # ALWAYS fp32, never the model dtype. `prope.py:306` does
    # `einsum(_lift_K(Ks_norm), viewmats)` against fp32 internals and raises
    # "expected scalar type Float but found BFloat16" on anything else. This
    # matches training and eval, which build poses from the dataset in fp32 and
    # never cast them -- poses are geometry, not activations.
    del dtype
    return (viewmats.to(device=device, dtype=torch.float32),
            Ks.to(device=device, dtype=torch.float32))


def make_inputs(device, dtype, *, seed, n_mask, source_condition):
    """Synthetic inputs at the real shapes. Seeded so base and widened see the same x."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    BV, C = TOTAL_VIEW, 1536

    def r(*shape):
        return torch.randn(*shape, generator=g).to(device=device, dtype=dtype)

    # Sequence format (BV, 2C, N, 1): [condition | noisy], which is what the
    # transport hands the model in concat mode.
    x = r(BV, 2 * C, N_TOKENS, 1)
    # Mask channels: uniform in [0, 1), the range a real M_edit occupies -- and
    # NON-ZERO, which is the whole point of this gate.
    m = torch.rand(BV, n_mask, N_TOKENS, 1, generator=g).to(device=device, dtype=dtype)
    viewmats, Ks = make_cameras(device, dtype, n_views=TOTAL_VIEW, image_size=IMAGE_SIZE)
    kw = dict(
        t=torch.rand(BV, generator=g).to(device=device, dtype=dtype),
        camera_embedding=r(BV, 7, IMAGE_SIZE, IMAGE_SIZE),
        total_view=TOTAL_VIEW,
        cond_num=COND_NUM,
        prope_image_size=IMAGE_SIZE,
        viewmats=viewmats,
        Ks=Ks,
    )
    if source_condition:
        kw["source_condition"] = r(BV, C, GRID, GRID)
    return x, m, kw


def report_diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Max abs/rel difference, plus WHERE it lives.

    SESSION_7.md: a structureless bounded diff is acceptable, one 'concentrated in
    a channel block or a spatial region' is a bug at any magnitude. So report the
    concentration, not just the magnitude -- a max-abs alone cannot tell the two
    apart.
    """
    d = (a.float() - b.float()).abs()
    scale = b.float().abs().max().clamp_min(1e-12)
    out = {
        "max_abs": d.max().item(),
        "max_rel": (d.max() / scale).item(),
        "n_differing": int(torch.count_nonzero(d).item()),
        "n_total": d.numel(),
        "bitwise_identical": bool(torch.equal(a, b)),
    }
    if out["max_abs"] > 0:
        # Fraction of the total error mass in the worst channel. Uniform noise
        # over C channels gives ~1/C; a channel-block bug gives ~1.
        per_ch = d.sum(dim=[i for i in range(d.ndim) if i != 1])
        out["worst_channel_share"] = (per_ch.max() / per_ch.sum().clamp_min(1e-30)).item()
        out["uniform_share_would_be"] = 1.0 / per_ch.numel()
    return out


def gate_one(name, config, ckpt, *, n_mask, device, dtype, seed, source_condition):
    print(f"\n=== {name}  (n_mask={n_mask}, dtype={dtype}, device={device})", flush=True)

    base = load_ckpt(build(config, 0, device), ckpt, expand=False).to(dtype)
    wide = load_ckpt(build(config, n_mask, device), ckpt, expand=True).to(dtype)
    print(f"  shapes: {mc.describe(wide)}", flush=True)

    # --- 1. WEIGHTS ---------------------------------------------------------
    assert mc.mask_channels_are_zero(wide), "mask channels are not zero after expand_state_dict"
    base_sd, wide_sd = base.state_dict(), wide.state_dict()
    widened_keys = set(mc.embedder_weight_keys(wide))
    preserved, n_widened = True, 0
    for k, v in base_sd.items():
        w = wide_sd[k]
        if k in widened_keys and w.shape != v.shape:
            n_widened += 1
            preserved &= torch.equal(w[:, : v.shape[1]], v)   # pre-trained block
            preserved &= bool(torch.count_nonzero(w[:, v.shape[1]:]) == 0)  # mask block
        else:
            preserved &= torch.equal(w, v)
    assert n_widened > 0, "no embedder was actually widened -- the gate would be vacuous"
    print(f"  [1] WEIGHTS: {n_widened} embedder weights widened, "
          f"all {len(base_sd)} tensors preserved bit-exactly: {preserved}", flush=True)
    assert preserved

    # --- 3. CAMERA (checked before the forward: it is a precondition) -------
    cam_w_base, cam_w_wide = base.camera_embedder.proj.weight, wide.camera_embedder.proj.weight
    assert cam_w_base.shape[1] == 7, f"camera in_channels moved: {cam_w_base.shape[1]}"
    assert torch.equal(cam_w_base, cam_w_wide), "camera_embedder weights differ"
    print(f"  [3] CAMERA: in_chans={cam_w_base.shape[1]}, weights bit-identical", flush=True)

    # --- 2. FORWARD ---------------------------------------------------------
    x, m, kw = make_inputs(device, dtype, seed=seed, n_mask=n_mask,
                           source_condition=source_condition)
    assert torch.count_nonzero(m) == m.numel(), "mask must be non-zero for this gate to bite"
    with torch.no_grad():
        out_base = base(x, **kw)
        out_wide = wide(torch.cat([x, m], dim=1), **kw)
    assert out_base.shape == out_wide.shape, f"{out_base.shape} vs {out_wide.shape}"

    d = report_diff(out_base, out_wide)
    verdict = "BIT-IDENTICAL" if d["bitwise_identical"] else "DIFFERS"
    print(f"  [2] FORWARD: {verdict}  out={tuple(out_base.shape)}", flush=True)
    for k, v in d.items():
        print(f"        {k}: {v}", flush=True)

    # Camera token output, through the shared embedder.
    with torch.no_grad():
        cam_ok = torch.equal(base.camera_embedder(kw["camera_embedding"]),
                             wide.camera_embedder(kw["camera_embedding"]))
    print(f"  [3] CAMERA tokens identical: {cam_ok}", flush=True)
    assert cam_ok

    del base, wide
    if device == "cuda":
        torch.cuda.empty_cache()
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--config-dir", default=str(pathlib.Path(__file__).resolve().parents[1]
                                               / "configs" / "training"))
    p.add_argument("--n-mask", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", default="level1,cascade")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"torch {torch.__version__} | deterministic=True benchmark=False tf32=False "
          f"seed={args.seed}", flush=True)

    cd, kd = pathlib.Path(args.config_dir), pathlib.Path(args.checkpoint_dir)
    specs = {
        "level1":  (cd / "DA3_level1.yaml",  kd / "da3_level1.pt",  False),
        "cascade": (cd / "DA3_cascade.yaml", kd / "da3_cascade.pt", True),
    }
    dtype = getattr(torch, args.dtype)

    results = {}
    for name in args.models.split(","):
        cfg, ck, sc = specs[name.strip()]
        results[name.strip()] = gate_one(name.strip(), str(cfg), str(ck), n_mask=args.n_mask,
                                         device=args.device, dtype=dtype, seed=args.seed,
                                         source_condition=sc)

    print("\n=== GATE SUMMARY", flush=True)
    all_exact = True
    for name, d in results.items():
        all_exact &= d["bitwise_identical"]
        print(f"  {name:8s} bit-identical={d['bitwise_identical']} "
              f"max_abs={d['max_abs']:.3e} max_rel={d['max_rel']:.3e} "
              f"differing={d['n_differing']}/{d['n_total']}", flush=True)
    print(f"\nINERTNESS GATE: {'PASS (bit-identical)' if all_exact else 'SEE TOLERANCE ABOVE'}",
          flush=True)
    return 0 if all_exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
