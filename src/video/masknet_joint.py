"""In-graph MaskNet: predict the conditioning mask instead of loading it from disk.

GeoFix joint training. Stock GLD and every frozen-mask arm read the mask from an
`.npz` pack, so it is a CONSTANT and nothing flows back to the predictor. This
module puts the predictor inside the graph so the diffusion loss can train it.

## Why this is expected to help, and where the argument stops

The frozen head minimises L2 against `oracle_l2` over all 1296 tokens. MSE
converges to the conditional MEAN per token, and the sum is dominated by the ~1000
easy clean tokens -- so capacity goes where the error is cheap. Measured, that
produces rho 0.65 overall but only **0.328 overlap on the oracle's worst 5% of
tokens**, with the extremes systematically softened (most-damaged decile predicted
at 0.69x). Joint training replaces "match the plane everywhere" with "produce
whatever mask lowers the diffusion loss", which weights a token by how much it
changes the OUTPUT rather than by how much it changes an L2 against a proxy.

That is one hypothesis about why the frozen head buys +0.027 dB where the oracle
buys +0.413, and it addresses only the OBJECTIVE. Two other causes it cannot
touch:

  - **Input resolution.** MaskNet sees DA3 level-1 at one vector per 14x14 block.
    `geofix.masknet.net` measures 36.9% of the oracle's variance as genuinely
    sub-token, so no objective recovers it.
  - **Capacity.** The head explains 38.8% against a token-resolution ceiling of
    63.1% -- 24 points of headroom that is features and width, not loss.

So joint training is worth running and is not obviously sufficient. Whether the
objective is the binding constraint is exactly what `split-top_oracle` vs
`split-bot_oracle` is measuring.

## Why the LEVEL-1 head and not `sf633_l0`, which scores higher

`sf633_l0` needs level-0 features alongside level-1. They ARE obtainable --
`RAE_DA3.encode(mode='all')` returns all four levels in ONE encoder forward -- but
that path returns **raw** features in `(B*V, N, C)` sequence layout, where
`mode='single'` returns **latent_norm** in `(B*V, C, 36, 36)`. Feeding a head
trained on latent_norm with raw features produces something plausible and wrong,
which is the failure mode this project has hit before. The level-1 head consumes
`latents_art` exactly as the trainer already computes it, with no conversion and
no chance of a silent space mismatch, at a cost of rho 0.704 against 0.722.
Add L0 after the mechanism is shown to work, not before.

## The normalisation actually matches -- checked, not assumed

`geofix.masknet.train.DA3FeatureSource` encodes through `encode_artifact`, which is
`(images - rae.encoder_mean)/rae.encoder_std` then `rae.encode(..., level=level)`
in `mode='single'`. `prepare_data` computes `latents_art` with the identical two
lines, and both point at the `dl3dv_art_504__imagenet` stats leg. Same space.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class JointMaskNet(nn.Module):
    """Wraps a pretrained `geofix.masknet.net.MaskNet` for in-graph use.

    Call it with the artifact's level-1 features `(B*V, C, g, g)` and it returns a
    mask shaped `(B, V, 1, g, g)` -- the layout `prepare_data` expects.
    """

    def __init__(self, ckpt_path: str, device, dtype=None):
        super().__init__()
        # geofix IS importable here: it is pip-installed in the `gld` env
        # (v0.0.1). The note in `video/geofix_pairs.py` saying the fork cannot
        # import it is stale, and is why `pool_mask` was duplicated.
        from geofix.masknet.net import MaskNet

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = (ck.get("config") or {}).get("model", {}) or {}
        if cfg.get("aux_l0") or cfg.get("aux_refs") or cfg.get("aux_render"):
            raise ValueError(
                f"{ckpt_path}: head needs auxiliary inputs {sorted(k for k in ('aux_l0','aux_refs','aux_render') if cfg.get(k))}, "
                "which are not wired in-graph. Use a level-1-only head (see this "
                "module's docstring for why L0 is not a drop-in).")
        if cfg.get("out_res", "token") != "token":
            raise ValueError(
                f"{ckpt_path}: out_res={cfg.get('out_res')!r}. The camera channel "
                "consumes the 36x36 token grid; a pixel head would be pooled "
                "straight back down.")
        self.net = MaskNet(base=int(cfg.get("base", 64)),
                           out_res="token",
                           init_bias=float(cfg.get("init_bias", 0.0)))
        self.net.load_state_dict(ck["model"], strict=True)
        self.net.to(device=device, dtype=dtype)
        self.step_trained_from = int(ck.get("step", -1))
        self.ckpt_path = ckpt_path

    def forward(self, feats_bv: torch.Tensor, B: int, V: int, cond_views: int) -> torch.Tensor:
        """`(B*V, C, g, g)` -> `(B, V, 1, g, g)`, reference slots forced to 0.

        The reference half is zeroed because `train_multiview_da3` asserts it: a
        non-zero mask on a conditioning view would claim there is damage to repair
        in a clean training photograph. The assert is the reason, not a formality
        -- see the `geofix_mask_tokens[:, :cond_num]` check in the training loop.
        """
        m = self.net(feats_bv)                      # (B*V, 1, g, g), already in [0,1]
        g = m.shape[-1]
        m = m.reshape(B, V, 1, g, g)
        if cond_views > 0:
            # index_fill rather than in-place slice assignment: the latter is an
            # autograd error on a tensor that requires grad.
            keep = torch.zeros_like(m[:, :cond_views])
            m = torch.cat([keep, m[:, cond_views:]], dim=1)
        return m
