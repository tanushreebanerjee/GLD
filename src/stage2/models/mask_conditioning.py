"""Widen the patch embedders for a mask channel, and load a narrow checkpoint into them.

Session 7 step 2. New file plus one small hook in `DDT.py` (GeoFix hard rule 9:
new files and small hooks, never rewrites -- we may need to rebase onto upstream).

## What widening means here

`DDT.__init__` computes

    embed_in_channels = in_channels * 2 if is_concat_mode else in_channels

and feeds it to the patch embedders. With `n_mask` it becomes `2C + n_mask`, so
the embedder input is

    [ condition (C) | noisy (C) | mask (n_mask) ]
      ^-- 0            ^-- C      ^-- 2C

**The mask occupies the LAST `n_mask` channels.** Every function here asserts
that rather than assuming it, because the failure is silent in the worst way: pad
a checkpoint tensor at the FRONT instead of the end and every pre-trained channel
shifts by `n_mask`. Shapes still match, the model still runs, and the output is
garbage that looks like "conditioning hurt". This is hard rule 7's shape --
a sign/offset convention that has to be checked where it is consumed, not
inferred from the concat order at some call site.

What must NOT widen: `final_layer` and `special_output_proj` emit `C = 1536`
(`DDT.py:282` computes `self.x_channel_per_token` from `in_channels`, *not*
doubled and not widened). The mask is an input, never a prediction.
`camera_embedder` is a separate path on `cam_in_channels = 7` and is untouched --
it is deliberately absent from `_EMBEDDER_ATTRS`.

## The two architecture modes are NOT symmetric

    level-1 model   DA3_level1.yaml:34    architecture_mode "new"   4 embedders
    cascade model   DA3_cascade.yaml:33   architecture_mode "old"   2 embedders

In `"old"` mode `DDT.py:273-277` aliases `_ref`/`_tgt` onto the same module, so a
loop over the four attribute names visits each module **twice**. Everything here
deduplicates on `id(module)`. `SESSION_7.md` step 2 said "the four PatchEmbeds in
both models"; that is wrong for the cascade and is corrected in
`docs/ARCH_NOTES.md` § CORRECTION 2026-08-10.

**`named_modules()` and `state_dict()` disagree about aliases, and the difference
is load-bearing.** `named_modules()` memoizes on identity and yields an aliased
module **once**; `state_dict()` recurses through `_modules` and emits it under
**every** registered name. So the cascade's 2 distinct embedders produce **6**
state-dict keys (`s_embedder`, `x_embedder`, and the four `_ref`/`_tgt`
aliases). Measured, not assumed: an adapter that padded only the deduplicated
names left 4 of 6 keys narrow and the load still failed with 4 size mismatches.

Hence the two-function split below, which is the whole subtlety of this file:

- `mask_embedders` — deduplicated **by identity**, for operations on *modules*
  (zeroing, describing, width checks). Visiting an aliased module twice would
  double-count it.
- `embedder_weight_keys` — **every** attribute name that exists, for operations
  on a *state dict*. Padding only the canonical name is the bug above.

## Why an explicit adapter is required at all

Every load path in the repo passes `strict=False`
(`eval_gld_metric.py:573,591,616`, `latent_blending.py:717`,
`train_multiview_da3.py:597`), which raises the obvious fear that a widened
embedder would silently keep its random init. **It would not.** Verified on
torch 2.10.0+cu128: `strict` governs only *missing* and *unexpected* keys, while
a size mismatch is accumulated into `error_msgs` and raised regardless:

    RuntimeError: size mismatch for weight: copying a param with shape
    torch.Size([768, 3072, 1, 1]) ... current model is torch.Size([768, 3073, 1, 1])

So the surgery cannot run on unloaded weights -- it simply cannot load without
`expand_state_dict`. The zero padding that adapter applies **is** the
zero-initialization the session calls for; there is no separate init step, and
therefore no way to get a loaded-but-not-zeroed model.
"""

from __future__ import annotations

import torch

#: Attribute names that may hold an embedder consuming `embed_in_channels`.
#: Both architecture modes are listed; `_mask_embedders` filters to what exists
#: and deduplicates by identity, so "old" mode yields 2 and "new" yields 4.
#: `camera_embedder` is deliberately absent -- it is a separate 7-channel path.
_EMBEDDER_ATTRS = (
    "x_embedder_ref", "x_embedder_tgt",
    "s_embedder_ref", "s_embedder_tgt",
    "x_embedder", "s_embedder",
)


def mask_embedders(model) -> list[tuple[str, torch.nn.Module]]:
    """The patch embedders that consume `embed_in_channels`, DEDUPLICATED BY IDENTITY.

    Returns `[(attr_name, module), ...]` -- 4 entries in "new" mode, 2 in "old".
    For module-level operations only. To touch a *state dict*, use
    `embedder_weight_keys`: state_dict does not deduplicate aliases and this does.
    """
    out: list[tuple[str, torch.nn.Module]] = []
    seen: set[int] = set()
    for attr in _EMBEDDER_ATTRS:
        mod = getattr(model, attr, None)
        if mod is None or id(mod) in seen:
            continue
        seen.add(id(mod))
        out.append((attr, mod))
    return out


def embedder_weight_keys(model) -> list[str]:
    """EVERY `<embedder>.proj.weight` key `state_dict()` will emit. Not deduplicated.

    In "old" mode one module is registered under three names, and `state_dict()`
    emits all three, so this returns 6 keys for the cascade's 2 modules against
    `mask_embedders`' 2. Padding only the deduplicated names leaves the alias
    keys narrow and the load fails -- which is exactly the bug this exists to
    prevent.
    """
    return [
        f"{attr}.proj.weight"
        for attr in _EMBEDDER_ATTRS
        if getattr(model, attr, None) is not None
    ]


def mask_channel_slice(model) -> slice:
    """The input-channel slice the mask occupies: the LAST `n_mask` channels.

    Raises if the model was not built with `n_mask > 0`, so callers cannot
    quietly treat an unwidened model as widened.
    """
    n_mask = int(getattr(model, "n_mask", 0))
    if n_mask <= 0:
        raise ValueError(
            "model was built with n_mask=0; it has no mask channels to address. "
            "Pass n_mask through the model config before conditioning."
        )
    total = _embed_in_channels(model)
    return slice(total - n_mask, total)


def _embed_in_channels(model) -> int:
    """Input width the embedders were actually constructed with.

    Read off a live module rather than recomputed from config -- recomputing is
    how a widening silently disagrees with the thing it widened.
    """
    named = mask_embedders(model)
    if not named:
        raise ValueError("model exposes no patch embedders to widen")
    widths = {m.proj.weight.shape[1] for _, m in named}
    if len(widths) != 1:
        raise ValueError(f"embedders disagree on input width: {sorted(widths)}")
    return widths.pop()


def expand_state_dict(state_dict: dict, model, *, strict_shapes: bool = True) -> dict:
    """Zero-pad a narrow checkpoint's embedder weights to the widened model.

    Returns a NEW dict; `state_dict` is not mutated. Only `<embedder>.proj.weight`
    is touched -- `.proj.bias` is per-output-channel and unchanged by widening.

    The padding goes at the END of the input-channel dim, matching
    `[condition | noisy | mask]`. Asserted against `model.n_mask`, so a checkpoint
    whose width differs by anything other than exactly `n_mask` raises instead of
    being quietly reshaped.
    """
    n_mask = int(getattr(model, "n_mask", 0))
    if n_mask <= 0:
        return dict(state_dict)

    out = dict(state_dict)
    model_sd = model.state_dict()
    for key in embedder_weight_keys(model):
        if key not in out:
            # A genuinely absent embedder is the caller's problem to notice via
            # missing_keys; silently inventing a tensor here would hide it.
            continue
        w_old = out[key]
        want = model_sd[key].shape
        if tuple(w_old.shape) == tuple(want):
            continue  # already widened (e.g. resuming our own checkpoint)
        gap = want[1] - w_old.shape[1]
        if strict_shapes and gap != n_mask:
            raise ValueError(
                f"{key}: checkpoint has {w_old.shape[1]} input channels, model "
                f"wants {want[1]} -- a gap of {gap}, but n_mask is {n_mask}. "
                "Refusing to pad: this is not the widening this model describes."
            )
        w_new = torch.zeros(want, dtype=w_old.dtype, device=w_old.device)
        w_new[:, : w_old.shape[1]] = w_old
        out[key] = w_new
    return out


@torch.no_grad()
def zero_mask_channels_(model) -> int:
    """Force the mask channels' weights to zero in place. Returns modules touched.

    `expand_state_dict` already produces zeros, so this is for the other two
    paths: a model built fresh without loading, and a paranoia check before the
    inertness gate. Idempotent.
    """
    sl = mask_channel_slice(model)
    n = 0
    for _, mod in mask_embedders(model):
        mod.proj.weight[:, sl].zero_()
        n += 1
    return n


@torch.no_grad()
def mask_channels_are_zero(model) -> bool:
    """True iff every embedder's mask channels are exactly zero.

    The precondition of the inertness gate: with these zero, the mask cannot
    reach any activation, so the widened model must reproduce the original.
    """
    sl = mask_channel_slice(model)
    return all(
        torch.count_nonzero(mod.proj.weight[:, sl]).item() == 0
        for _, mod in mask_embedders(model)
    )


def describe(model) -> dict:
    """Shape summary for logging into the gate's provenance."""
    named = mask_embedders(model)
    return {
        "architecture_mode": getattr(model, "architecture_mode", None),
        "n_mask": int(getattr(model, "n_mask", 0)),
        "embed_in_channels": _embed_in_channels(model) if named else None,
        "out_channels": int(getattr(model, "out_channels", -1)),
        "n_embedders": len(named),
        "embedders": {n: tuple(m.proj.weight.shape) for n, m in named},
    }
