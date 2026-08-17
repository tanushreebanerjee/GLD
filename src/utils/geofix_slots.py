"""GeoFix session 8: the two conditioning slots, in ONE place.

Three separate code paths in this repo build the same conditioning from scratch:

    prepare_data                    (train_multiview_da3.py)   -- training
    validate_da3_multiview          (da3_validation.py)        -- in-loop validation
    get_denoised_features           (da3_validation_metric.py) -- offline inference

They must agree exactly. If inference fills a slot differently from training, the
model is evaluated on inputs it never saw and the number is not wrong-looking, just
wrong -- which is how session 8 already lost one set of validation metrics (they
scored the model against the 3DGS render instead of the clean photograph, because
the validation path never got the loader's image-role swap).

So the slot semantics live here and the three sites call in. Two functions, because
the two slots are independently switchable and the ablation ladder is exactly those
two booleans.

## Slot 1 -- `fill_cond_artifact`

Stock GLD leaves `latents_cond[:, cond_num:]` as ZEROS: the target half of the
concat-mode condition channel is never written. Filling it with features of the
3DGS render is route (a) conditioning through a pathway that already exists at full
C width, with no embedder surgery. In session 6.5 the render never entered the
network at all -- it arrived only as a sampling-time overwrite of `x_t`.

## Slot 2 -- `grade_camera_mask`

Camera-embedding channel 0 is a per-view spatial plane, patch-embedded at 504/14
onto our 36x36 token grid, preserved by CFG. Its polarity already agrees with
`M_edit` (channel 0 = 1 means "target, generate it"; M_edit = 1 means "repair
here"), so this is the one mask in the project that enters with NO sign flip --
hard rule 7 is satisfied by construction, and a flip added "to be safe" would be
the bug.

> **MEASURED 2026-08-17: it starts far off-distribution and then converges.** Jobs
> 7264903 (this slot on) vs 7264904 (off), one config, `--geofix-no-mask` the only
> difference, `clip_grad: 1.0`, linear warmup over 100 steps. Gradient norms in the
> mask arm, binned by 10 steps as (count above 10, mean):
>
>     0-9   10/10  269.5      40-49  10/10   67.1
>     10-19 10/10  212.9      50-59   9/10   30.3
>     20-29 10/10  187.9      60-69   4/10   15.6
>     30-39 10/10  128.1      70-79   1/10   12.0
>
> Train loss: 1.72 against slot 1's 0.44 at step 20; **0.3413 against 0.3184 at step
> 130.** The gap closes and the warmup is load-bearing.
>
> **Do not read this curve before the warmup ends.** Steps 0-26 look like divergence
> -- I called it that -- and they are not. Norms are only logged above
> `clip_grad * 10`, so the warning *going quiet* is the pass signal: the absence of
> output is the measurement, which reads exactly like an absence of data.
>
> **Why it starts that far out**, and this is where `docs/ARCH_NOTES.md` is
> incomplete. ARCH_NOTES says channel 0 "saw a strict {0, 1} for all 175k
> iterations" -- true and insufficient. The channel was also spatially **CONSTANT
> WITHIN EACH VIEW**: one value per frame, distinguishing reference from target. A
> mask makes it spatially varying inside a view for the first time, so
> `camera_embedder.proj`'s 14x14 convolution sees structure in a channel that was
> always flat. That accounts for the initial magnitude without implying the route is
> unusable.
>
> The zero-init widening route (`stage2/models/mask_conditioning.py`, inertness gate
> PASS 2026-08-11) remains the fallback for an arm that genuinely fails to absorb.
> It was NOT needed here.
"""

from __future__ import annotations

import torch


def fill_cond_artifact(latents_cond_5d: torch.Tensor,
                       latents_art: torch.Tensor,
                       cond_num: int) -> torch.Tensor:
    """Slot 1: write render features into the target half of the condition channel.

    `latents_cond_5d` is (B, V, C, h, w) and is modified IN PLACE and returned.
    `latents_art` may be (B*V, C, h, w) or (B, V, C, h, w) -- the encoders in this
    repo return the flat form, the callers hold the 5-d form.

    Reference slots `[0, cond_num)` are left exactly as the caller set them: they
    carry the clean photographs, and overwriting them with renders would destroy
    the reference signal the whole method depends on.
    """
    B, V, C, h, w = latents_cond_5d.shape
    if latents_art.ndim == 4:
        if latents_art.shape[0] != B * V:
            raise ValueError(
                f"artifact latents have batch {latents_art.shape[0]}, expected "
                f"B*V = {B}*{V} = {B * V}.")
        latents_art = latents_art.reshape(B, V, *latents_art.shape[1:])
    if latents_art.shape != (B, V, C, h, w):
        raise ValueError(
            f"artifact latents {tuple(latents_art.shape)} must match the condition "
            f"channel {(B, V, C, h, w)}.")
    latents_cond_5d[:, cond_num:] = latents_art[:, cond_num:].to(
        device=latents_cond_5d.device, dtype=latents_cond_5d.dtype)
    return latents_cond_5d


def grade_camera_mask(view_plane_5d: torch.Tensor,
                      mask_tokens: torch.Tensor,
                      cond_num: int,
                      size: tuple[int, int]) -> torch.Tensor:
    """Slot 2: replace the constant 1.0 on the target half with M_edit.

    `view_plane_5d` is (B, V, 1, H, W) -- camera channel 0 before concatenation,
    already 0 on the reference half. Modified IN PLACE and returned.
    `mask_tokens` is (B, V, 1, g, g) on the token grid, `edit1` polarity.

    Upsampled NEAREST, so one token maps to exactly one 14x14 patch. Bilinear
    would blur values across patch boundaries, which would make the MAX pooling
    that produced these tokens (hard rule 6) pointless -- the whole reason for MAX
    is that a thin floater must not be averaged away, and a bilinear upsample
    re-averages it on the way back out.

    NO SIGN FLIP. See the module docstring: this plane and `M_edit` already agree.
    """
    B, V, one, H, W = view_plane_5d.shape
    if one != 1:
        raise ValueError(f"view plane must have 1 channel, got {one}.")
    if mask_tokens.ndim != 5 or mask_tokens.shape[:3] != (B, V, 1):
        raise ValueError(
            f"mask must be (B={B}, V={V}, 1, g, g), got {tuple(mask_tokens.shape)}. "
            "A multi-plane mask has to be reduced to one plane by the caller, which "
            "is a modelling decision and not this function's to make.")
    if size != (H, W):
        raise ValueError(f"size {size} must match the view plane {(H, W)}.")
    g_h, g_w = mask_tokens.shape[3], mask_tokens.shape[4]
    up = torch.nn.functional.interpolate(
        mask_tokens.reshape(B * V, 1, g_h, g_w).to(dtype=view_plane_5d.dtype),
        size=(H, W), mode="nearest").reshape(B, V, 1, H, W)
    view_plane_5d[:, cond_num:] = up[:, cond_num:].to(view_plane_5d.device)
    return view_plane_5d
