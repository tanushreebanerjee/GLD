"""Mask-weighted latent compositing during sampling — GeoFix session 6.5.

A DIAGNOSTIC, not a method. It answers two questions without any training and
without touching the model: do GeoFix's uncertainty masks carry information GLD
can use, and can a blend at level 1 reach appearance at all?

New file. The only edits outside it are a `blend_fn` keyword threaded through
`Sampler.sample_ode` and `integrators.ode`, both defaulting to `None` so the
unblended path is byte-for-byte unchanged (GeoFix CLAUDE.md hard rule 9).


THE COMPOSITE
-------------
At every sampling step the generated features are composited with the ENCODED
ARTIFACT features under an edit-polarity mask:

    F_blended = (1 - M_edit) * F_artifact + M_edit * F_generated

`M_edit = 1` means "this token is degraded, generate here"; `M_edit = 0` means
"this token is fine, keep what the 3DGS render already had". That is the
polarity on disk (`geofix.masks.common.POLARITY == "edit1"`), and it is the
OPPOSITE of FreeFix's convention, whose `rasterize_splats_w_certainty` returns
`1 - certainties` despite the name. GeoFix session 6 settled the orientation by
measurement rather than by argument: `fisher_g2` scores AUROC 0.572 against
degraded tokens and its reflection scores 0.428, below chance. The sign is not
reopened here; it is re-checked by the two tests below.


WHY THIS ACTS ON THE VELOCITY AND NOT ON THE STATE
--------------------------------------------------
`Sampler.sample_ode` hands the loop to `torchdiffeq.odeint`, which owns the
integration and exposes no hook that can modify state between steps. Replacing
it with a hand-written Euler loop would work, but it would also make the
`M_edit == 1` identity test compare our loop against `odeint`'s rather than
testing the blend, and it is a much larger diff.

The path makes that unnecessary. With `path_type: 'Linear'` and
`prediction: 'velocity'` (`path.ICPlan`),

    x_t = (1 - t) * F + t * eps          alpha_t = 1 - t,  sigma_t = t
    v   = dx_t/dt = eps - F

so the data-prediction is closed-form, `F_hat = x_t - t * v`, and is AFFINE in
`v` at fixed `x_t`. Compositing `F_hat` under a mask is therefore identical to
compositing `v` against the artifact's own velocity:

    v_artifact = (x_t - F_artifact) / t
    v_blended  = M_edit * v_generated + (1 - M_edit) * v_artifact

Two properties follow, and both are load-bearing:

  * `M_edit == 1` returns `v_generated` bit-for-bit. The identity test holds by
    construction, not by luck.

  * `M_edit == 0` is EXACT under Euler, not merely convergent. Writing
    `y = x - F_artifact`, the artifact velocity gives
    `y_{n+1} = y_n * t_{n+1} / t_n`, which telescopes to `y_N = y_0 * t_N / t_0`
    with no accumulated solver error. `Transport.check_interval` yields
    `t0 = 0, t1 = 0.999`, so the grid descends to `t_N = 0.001` and a fully
    preserved region lands within `~1e-3 * (eps - F_artifact)` of `F_artifact`.
    A bounded number, not a hope.

This is the flow-matching equivalent of what FreeFix does in pixel-space
diffusion — they composite `pred_original_sample`, the x0-prediction, inside a
vendored DDIM scheduler. Same mechanism, different parameterisation; no code is
shared, and none could be.


THE CHANNEL SPLIT IS THE TRAP
------------------------------
Under `is_concat_mode` the tensor the solver integrates is `(BV, 2C, h, w)`,
laid out `[ref_cond | x_t]`. `velocity_ode` already returns zero drift for the
first `C` channels so the conditioning stays frozen. The blend must address the
SECOND half only. Writing into the first half corrupts what the model is
conditioned on and would not crash.


WHAT IS NOT DONE HERE
---------------------
No `* M_alpha` multiply. Opacity saturates on this data — `alpha_mean` 0.9964,
AUROC 0.501, i.e. a multiply by ~1 carrying no localisation signal (GeoFix
RESULTS.md, session 5 and session 6). Adding it would be inert, and a null
result must not be read as evidence about it either way.
"""

from __future__ import annotations

import torch as th


class LatentBlend:
    """Composites artifact features into the sampling trajectory under a mask.

    One instance is created per sampling stage (level 1, then the L1 -> L0
    cascade) and installed via `Sampler.sample_ode(blend_fn=...)`. The caller
    sets `f_artifact` and `mask` before each batch with `arm()`.

    Args:
        total_view: views per sample, `V`. The flat batch is `(B*V, ...)`.
        cond_num: reference views, `[0, cond_num)`. Views below this index are
            never blended -- they are the conditioning the model is given, and
            steering them would change what arm A conditions on rather than what
            it generates.
        eps_t: floor on `t` in the `1/t` of the artifact velocity. The eval grid
            bottoms out at `t = 1e-3` so this never binds, but a division that
            can produce inf on a schedule change is not worth leaving open.
    """

    def __init__(self, total_view: int, cond_num: int, eps_t: float = 1e-6):
        self.total_view = int(total_view)
        self.cond_num = int(cond_num)
        self.eps_t = float(eps_t)

        self.f_artifact = None   # (B*V, C, h, w) encoded artifact, latent_norm
        #: Ordered `[(t_low, mask, label), ...]`, first entry whose `t > t_low`
        #: wins. A single unscheduled mask is just one band with `t_low = -inf`.
        self.schedule: list[tuple[float, object, str]] = []
        self.enabled = False

        #: Incremented on every step the blend actually modifies the velocity.
        #: A run that reports 0 here did not blend, whatever its metrics say --
        #: which is precisely the failure "the masks do nothing" imitates.
        self.n_calls = 0

        #: Steps served per band, by label. For arm E this is the assertion that
        #: the schedule's boundaries were actually crossed: a band with 0 calls
        #: means that gamma_c never ran, and the arm silently measured a
        #: different schedule than the one it reports.
        self.band_calls: dict[str, int] = {}

    # -- setup ------------------------------------------------------------

    def arm(self, f_artifact: th.Tensor, mask: th.Tensor) -> "LatentBlend":
        """Load the artifact features and mask for the batch about to be sampled.

        `mask` is `M_edit` already pooled to the token grid with MAX (GeoFix
        hard rule 6 -- a thin floater mean-pooled over a 14x14 patch washes out
        to near-clean, and a training-free blend composites VALUES, so the
        session-6 finding that AUROC is indifferent to the reducer does not
        transfer here).
        """
        if f_artifact.shape[-2:] != mask.shape[-2:]:
            raise ValueError(
                f"mask grid {tuple(mask.shape[-2:])} != feature grid "
                f"{tuple(f_artifact.shape[-2:])}; pool the mask to the token "
                f"grid before arming"
            )
        if mask.shape[0] != f_artifact.shape[0]:
            raise ValueError(
                f"mask batch {mask.shape[0]} != feature batch {f_artifact.shape[0]}"
            )
        if mask.min() < 0.0 or mask.max() > 1.0:
            raise ValueError(
                f"M_edit out of [0,1]: [{float(mask.min())}, {float(mask.max())}]"
            )

        self.f_artifact = f_artifact
        self.schedule = [(float("-inf"), mask, "static")]
        self.enabled = True
        self.n_calls = 0
        self.band_calls = {"static": 0}
        return self

    def arm_schedule(self, f_artifact, bands) -> "LatentBlend":
        """Arm with a gamma_c SCHEDULE over the sampling trajectory (arm E).

        `bands` is `[(t_low, mask, label), ...]`, highest `t_low` first; the
        first band whose `t > t_low` serves the step.

        Session 6 stacked the three gamma_c channels instead of scheduling them
        for one stated reason -- there was no denoising schedule to hang a
        schedule on. A sampler hook IS that schedule, so the constraint has
        lifted, and arm B would otherwise carry a workaround into the very
        session that removes its cause.

        FreeFix's schedule (`ours/pipelines/sdxl_pipeline.py:1466-1469`, with
        `c_scheduler: [0.3, 0.9, 1.0]` over 50 steps) is loosest first and
        steepest last: broad graded edits while the sample is still mostly
        noise, tightening toward a near-binary preserve as it resolves.

        **Their step index runs the opposite way from our `t`.** GLD integrates a
        DESCENDING `t` (1 -> ~0.001), so their step fraction `f` maps to
        `t = 1 - f`. What transfers is the ORDERING by dynamic range, not the
        numerical cutoffs -- their gamma_c is a linear pre-rasterisation scale
        and ours an exponent applied after median-normalised accumulation
        (`BASELINES.md` § The four divergences, #1).
        """
        bands = list(bands)
        if not bands:
            raise ValueError("arm_schedule needs at least one band")
        if any(bands[i][0] <= bands[i + 1][0] for i in range(len(bands) - 1)):
            raise ValueError(
                f"bands must be strictly descending in t_low, got "
                f"{[b[0] for b in bands]}"
            )
        for t_low, mask, label in bands:
            if mask.shape[-2:] != f_artifact.shape[-2:]:
                raise ValueError(f"band {label!r}: pool the mask to the token grid")
            if mask.min() < 0.0 or mask.max() > 1.0:
                raise ValueError(f"band {label!r}: M_edit out of [0,1]")

        self.f_artifact = f_artifact
        self.schedule = bands
        self.enabled = True
        self.n_calls = 0
        self.band_calls = {label: 0 for _t, _m, label in bands}
        return self

    def disarm(self) -> "LatentBlend":
        self.f_artifact = None
        self.schedule = []
        self.enabled = False
        return self

    def _band_for(self, t: float):
        for t_low, mask, label in self.schedule:
            if t > t_low:
                return mask, label
        return self.schedule[-1][1], self.schedule[-1][2]

    # -- the hook ---------------------------------------------------------

    def __call__(self, x: th.Tensor, t: th.Tensor, drift: th.Tensor) -> th.Tensor:
        """Return the blended velocity for state `x` at time `t`.

        `drift` is what `Sampler`'s drift function just produced: under
        `is_concat_mode` its first `C` channels are already zero and must stay
        zero.
        """
        if not self.enabled:
            return drift

        c = self.f_artifact.shape[1]
        concat = x.shape[1] == 2 * c
        if not concat and x.shape[1] != c:
            raise ValueError(
                f"state has {x.shape[1]} channels, expected {c} or {2 * c}"
            )

        x_noisy = x[:, c:] if concat else x
        v_gen = drift[:, c:] if concat else drift

        # t arrives as (B*V,); broadcast against (B*V, C, h, w).
        t_b = t.view(-1, *([1] * (x_noisy.dim() - 1))).to(x_noisy.dtype)
        t_b = t_b.clamp_min(self.eps_t)

        # The artifact's own velocity along the same linear path. Exact, not an
        # approximation: x_t = (1-t) F + t eps has dx/dt = (x_t - F) / t.
        v_art = (x_noisy - self.f_artifact) / t_b

        mask, label = self._band_for(float(t.reshape(-1)[0]))
        m = self._view_gated_mask(mask, v_gen)
        v_blended = m * v_gen + (1.0 - m) * v_art
        self.n_calls += 1
        self.band_calls[label] = self.band_calls.get(label, 0) + 1

        if concat:
            return th.cat([drift[:, :c], v_blended], dim=1)
        return v_blended

    def _view_gated_mask(self, mask: th.Tensor, like: th.Tensor) -> th.Tensor:
        """`M_edit`, forced to 1.0 on the reference views.

        `M_edit = 1` is "generate here", so 1.0 on the refs means the blend is
        the identity there and those views follow plain GLD exactly.
        """
        m = mask.to(dtype=like.dtype, device=like.device)
        if self.cond_num <= 0:
            return m

        bv = m.shape[0]
        if bv % self.total_view != 0:
            raise ValueError(
                f"batch {bv} is not a multiple of total_view {self.total_view}"
            )
        b = bv // self.total_view
        m = m.reshape(b, self.total_view, *m.shape[1:]).clone()
        m[:, : self.cond_num] = 1.0
        return m.reshape(bv, *m.shape[2:])


#: Arm E's gamma_c schedule, in FreeFix's units: STEP FRACTION, not `t`.
#:
#: FreeFix's `c_scheduler: [0.3, 0.9, 1.0]` over 50 steps puts steps 0-14 on
#: gamma_c=0.001, 15-44 on 0.01 and 45-49 on 0.1 -- loosest first, steepest last.
#: Ordering by dynamic range is what transfers, not the cutoffs: their gamma_c is
#: a linear pre-rasterisation scale, ours an exponent after median-normalised
#: accumulation.
#:
#: Kept in step fraction because that is the only place the boundaries are
#: stable. `gamma_c_schedule` converts them to the `t` the hook sees.
GAMMA_C_SCHEDULE_F = ((0.3, "fisher_g0"), (0.9, "fisher_g1"), (1.0, "fisher_g2"))


def gamma_c_schedule(time_dist_shift: float):
    """Step-fraction bands -> the descending-`t` bands `arm_schedule` wants.

    GLD does not step linearly in `t`. The sampler builds its grid as

        t = s*u / (1 + (s - 1)*u),   u = 1 - f,   s = time_dist_shift

    and `s` is 22.0454 at 504x504, which crushes the grid toward `t = 1`: of the
    49 drift evaluations, 45 land above `t = 0.7` and one below `t = 0.1`.

    An earlier version of this schedule mapped FreeFix's `f` straight through as
    `t = 1 - f`, which is only correct when `s = 1`. Measured, it put 45/4/0
    calls in the three bands -- arm E silently degenerating into arm B's
    `fisher_g0` while still looking like a schedule. Applying the same warp the
    sampler applies restores FreeFix's own 15/30/5 split.

    `band_calls` on the hook is what asserts this, per run. It is the reason
    that counter exists.
    """
    s = float(time_dist_shift)
    bands = []
    for f, label in GAMMA_C_SCHEDULE_F:
        u = 1.0 - float(f)
        t = s * u / (1.0 + (s - 1.0) * u)
        bands.append((t, label))
    # The last band must catch everything left, whatever rounding did to it.
    bands[-1] = (float("-inf"), bands[-1][1])
    return tuple(bands)


def constant_mask(like: th.Tensor, value: float) -> th.Tensor:
    """`M_edit` filled with `value`, shaped for `LatentBlend.arm`.

    The two hook-validation tests of session 6.5 run through this:

      * `value = 1.0` -> blend is the identity -> output must be BIT-IDENTICAL
        to the unblended arm. Catches a hook that fires when it should not, and
        catches an inverted sign (which would return the artifact instead).

      * `value = 0.0` -> output must reproduce `decode(F_artifact)` to
        autoencoder tolerance. Catches a hook that never fires -- a failure mode
        that otherwise looks exactly like "the masks do nothing".

    Note these constants are the reverse of the ones in the first draft of the
    session-6.5 prompt, which transposed them; the equation at the top of this
    module, the on-disk polarity and the two stated purposes all agree on the
    assignment above. `docs/HOOKS_6_5.md` carries the derivation.
    """
    return th.full(
        (like.shape[0], 1, *like.shape[2:]),
        float(value),
        dtype=like.dtype,
        device=like.device,
    )
