"""Tiny smoke of the gate's plumbing: shapes, all three classes/branches, l1_as_cond.

Not the gate -- a small-model rehearsal (seconds, ~1 MB of weights) so the real
run does not discover a TypeError after loading 6 GB of checkpoints. Real widths,
real checkpoints and the recorded tolerance are `mask_inertness_gate.py`'s job.

REQUIRES A GPU despite its size. `model_utils.py:602` routes attention through
xformers `memory_efficient_attention`, which has no CPU kernel -- and note it
downcasts q/k/v to bfloat16 even when the model is fp32, so "fp32" is only ever
fp32 outside attention. That is the precision ceiling any tolerance here lives
under.

Covers three paths, because they are three different pieces of code:
  level1/new  DDT.DiTwDDTHead, architecture_mode "new"  -- 4 embedders, what eval runs
  cascade     DDT_old.DiTwDDTHead + l1_as_cond          -- 2 embedders, what eval runs
  DDT/old     DDT.DiTwDDTHead, architecture_mode "old"  -- aliased, NOT what eval runs
"""
import sys, pathlib, torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from mask_inertness_gate import make_cameras  # noqa: E402  (shared, so both agree)

from stage2.models.DDT import DiTwDDTHead as New
from stage2.models.DDT_old import DiTwDDTHead as Old
from stage2.models import mask_conditioning as mc

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
C, N_MASK = 64, 1
GRID, PATCH, IMG = 4, 14, 56           # 56/14 = 4
N = GRID * GRID
BV, TV, CN = 4, 4, 2

# use_prope=True MATTERS: both shipped configs set it, and ProPE's attention
# asserts `viewmats`/`Ks` are present (model_utils.py:454). An earlier version of
# this test ran with use_prope=False, passed, and let the real gate discover the
# missing kwargs on a GPU node -- a rehearsal that skips the deployed branch is
# not a rehearsal.
COMMON = dict(patch_size=1, in_channels=C, hidden_size=[64, 96], depth=[2, 1],
              num_heads=[2, 2], use_rope=False, use_pos_embed=False, use_prope=True,
              use_rmsnorm=True, level=1, predict_cls=False, is_concat_mode=True,
              cam_input_size=IMG, cam_patch_size=PATCH, cam_in_channels=7)


def run(tag, cls, extra, n_mask, x, m, kw):
    torch.manual_seed(0)
    base = cls(**COMMON, **extra, n_mask=0).to(DEV).eval()
    torch.manual_seed(0)
    wide = cls(**COMMON, **extra, n_mask=n_mask).to(DEV).eval()
    sd = mc.expand_state_dict(base.state_dict(), wide)
    res = wide.load_state_dict(sd, strict=False)
    assert not res.missing_keys and not res.unexpected_keys, res
    assert mc.mask_channels_are_zero(wide)
    with torch.no_grad():
        a = base(x, **kw)
        b = wide(torch.cat([x, m], dim=1), **kw)
    same = torch.equal(a, b)
    print(f"{tag:10s} modules={len(mc.mask_embedders(wide))} keys={len(mc.embedder_weight_keys(wide))} "
          f"out={tuple(a.shape)} bit-identical={same} maxabs={(a-b).abs().max().item():.3e}")
    assert a.shape[1] == C, f"output width moved: {a.shape}"
    return same


x = torch.randn(BV, 2 * C, N, 1, device=DEV)
m = torch.rand(BV, N_MASK, N, 1, device=DEV)
viewmats, Ks = make_cameras(DEV, torch.float32, n_views=TV, image_size=IMG)
kw = dict(t=torch.rand(BV, device=DEV), camera_embedding=torch.randn(BV, 7, IMG, IMG, device=DEV),
          total_view=TV, cond_num=CN, prope_image_size=IMG, viewmats=viewmats, Ks=Ks)

ok = run("level1/new", New, dict(architecture_mode="new", cfg_mode="new"), N_MASK, x, m, kw)
# The cascade: DDT_old, l1_as_cond -- the branch that rebuilds encoder_input.
kw_c = dict(kw, source_condition=torch.randn(BV, C, GRID, GRID, device=DEV))
ok &= run("cascade", Old, dict(source_condition_mode="l1_as_cond"), N_MASK, x, m, kw_c)
# DDT.py's own "old" branch: not what eval runs, but it is reachable code.
ok &= run("DDT/old", New, dict(architecture_mode="old", cfg_mode="old"), N_MASK, x, m, kw)

print("\nTINY SMOKE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
