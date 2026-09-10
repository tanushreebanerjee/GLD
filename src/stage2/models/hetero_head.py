"""A heteroscedastic (mu, log sigma^2) head for the DDT, and the NLL that trains it.

## What this is for

Design B finetunes the refiner so its flow TARGET is the L2 error map itself, then reads
a mask off the decoded output. It predicts a point estimate under GLD's own
flow-matching MSE, and MSE converges to the CONDITIONAL MEAN -- it averages over every
token the render cannot determine, and spends the same capacity on those as on the ones
it can.

The heteroscedastic NLL replaces that with

    L = mean[ (v_hat - ut)^2 * exp(-logvar) + logvar ]

whose `1/sigma^2` reweighting DOWN-WEIGHTS the tokens the input cannot determine, so the
mean head stops averaging over irreducible ambiguity. That mechanism was worth +0.020
rho in MaskNet's decomposition, where it is the only difference between two otherwise
identical arms.

## Why a separate head instead of widening `final_layer.linear`

`DDTFinalLayer.linear` is `Linear(2048, 1536)` and every released checkpoint carries it
at that shape. Widening it to 3072 would make `load_state_dict` fail on the pretrained
file -- exactly the `[2048, 3074] vs [2048, 3072]` class of breakage this project has
already hit once. An additive parallel head loads beside the pretrained weights and
leaves them untouched (CLAUDE.md hard rule 9: new files plus small hooks, not rewrites).

## Why the factor of 2, which looks wrong and is deliberate

The Gaussian NLL is `0.5 * [ (y-mu)^2/sigma^2 + log sigma^2 ]`. This drops the 0.5.

That constant cannot move the optimum -- `dL/dlogvar = 0` still gives
`sigma^2 = (y-mu)^2`, and the `mu` gradient is reweighted by `exp(-logvar)` either way --
but it buys an exact ablation. The head is ZERO-initialised, so at step 0 `logvar = 0`,
`exp(-logvar) = 1`, and the loss is

    mean[ (v_hat - ut)^2 * 1 + 0 ]  ==  mean_flat((model_output - ut) ** 2)

which is `transport.training_losses`' existing VELOCITY branch, character for character.
With the 0.5 it would be half of it, and "half the loss" is a second change riding along
with the one being tested (hard rule 14: an ablation whose arms differ in more than the
knob is not an ablation). Step 0 is now literally the current objective, and the arm
differs from its control in the variance head alone.

## What sigma means here

The head predicts the variance of the VELOCITY, `ut = x0 - x1`. Under
`bridge_x0: artifact` the `x0` end is the artifact features, which are GIVEN at
deployment -- so uncertainty in `ut` is uncertainty in `x1`, the error-map latent, up to
a sign. That is the quantity a downstream conditioner wants, and it is why the head can
sit on the velocity without a change of variable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: `logvar` is exponentiated twice per step (as `exp(-logvar)` in the weight and
#: implicitly through the gradient), so an unclamped head can produce `inf` on one bad
#: batch and never recover. +-10 spans sigma in [6.7e-3, 1.5e2], far wider than any
#: plausible residual on normalised DA3 features, whose |z| is O(1).
LOGVAR_MIN, LOGVAR_MAX = -10.0, 10.0


class LogVarHead(nn.Module):
    """`decoder_hidden -> out_channels` log-variance, mirroring `DDTFinalLayer.linear`.

    Deliberately NOT adaLN-modulated, unlike the mean head. The mean is a function of
    the timestep and must be; the variance is a statement about how determined a TOKEN
    is by the conditioning, which is a property of the input rather than of where on the
    path we are. Keeping it unconditioned also keeps the added parameter count to one
    linear layer, so the arm's capacity gain over its control is negligible and the
    result cannot be read as "the bigger model won".
    """

    def __init__(self, hidden_size: int, out_channels: int, use_rmsnorm: bool = False):
        super().__init__()
        if use_rmsnorm:
            from stage2.models.DDT import RMSNorm
            self.norm = RMSNorm(hidden_size)
        else:
            self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        # ZERO INIT, and it is the whole reason step 0 reproduces the MSE objective.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x)).clamp(LOGVAR_MIN, LOGVAR_MAX)


def hetero_nll(model_output: torch.Tensor, ut: torch.Tensor,
               logvar: torch.Tensor) -> torch.Tensor:
    """`mean[(v_hat - ut)^2 * exp(-logvar) + logvar]`, per sample.

    Reduced over every axis but the batch, matching `transport.mean_flat`, so it drops
    into `terms['loss']` with no change to the caller.

    NEVER draws a sample. `E[(y - mu - sigma*eps)^2] = (y-mu)^2 + sigma^2` is minimised
    at `sigma = 0`, so a sampled estimator would collapse the variance head -- and the
    Gaussian lets the expectation be written in closed form anyway, which is the same
    gradient with strictly less noise.
    """
    if logvar.shape != model_output.shape:
        raise ValueError(
            f"logvar {tuple(logvar.shape)} does not match the model output "
            f"{tuple(model_output.shape)}. The head must emit one variance per predicted "
            "channel; a broadcast here would silently share one variance across 1536 "
            "channels and still train.")
    se = (model_output - ut) ** 2
    per = se * torch.exp(-logvar) + logvar
    return per.reshape(per.shape[0], -1).mean(dim=-1)
