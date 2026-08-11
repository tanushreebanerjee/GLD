import numpy as np
import torch as th
import torch.nn as nn
from torchdiffeq import odeint
from functools import partial
from tqdm import tqdm

class sde:
    """SDE solver class"""
    def __init__(
        self, 
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
        time_dist_shift,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = 1 - th.linspace(t0, t1, num_steps)
        self.t = time_dist_shift * self.t / (1 + (time_dist_shift - 1) * self.t)
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type
        self.time_dist_shift = time_dist_shift
        print(self.t)
    def __Euler_Maruyama_step(self, x, mean_x, t_curr, t_next, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t_curr
        dw = w_cur * th.sqrt(t_curr - t_next)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x - drift * (t_curr - t_next)
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x
    
    def __Heun_step(self, x, _, t_curr, t_next, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(t_curr - t_next)
        diffusion = self.diffusion(x, th.ones(x.size(0)).to(x) * t_curr)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(
            xhat, th.ones(x.size(0)).to(x) * t_curr, model, **model_kwargs
        )
        xp = xhat - (t_curr - t_next) * K1
        K2 = self.drift(
            xp, th.ones(x.size(0)).to(x) * t_next, model, **model_kwargs
        )
        return xhat - 0.5 * (t_curr - t_next) * (K1 + K2), xhat # at last time point we do not perform the heun step

    def __forward_fn(self):
        """TODO: generalize here by adding all private functions ending with steps to it"""
        sampler_dict = {
            "euler": self.__Euler_Maruyama_step,
            "heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Smapler type not implemented.")
    
        return sampler

    def sample(self, init, model, **model_kwargs) -> tuple[th.Tensor]:
        """forward loop of sde"""
        x = init
        mean_x = init 
        samples = []
        sampler = self.__forward_fn()
        for t_curr, t_next in zip(self.t[:-1], self.t[1:]):
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, t_curr, t_next, model, **model_kwargs)
                samples.append(x)

        return samples

class ode:
    """ODE solver class"""
    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        time_dist_shift,
        blend_fn=None,
        init_t=None,
        init_steps=None,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        # GeoFix session 6.5: mask-weighted latent compositing. None -> the call
        # path below is byte-for-byte the original. See transport/blending.py.
        self.blend_fn = blend_fn

        def _warp(u):
            return time_dist_shift * u / (1 + (time_dist_shift - 1) * u)

        # self.t = th.linspace(t0, t1, num_steps)
        self.t = _warp(1 - th.linspace(t0, t1, num_steps))

        # GeoFix: the img2img arm, for comparability with FreeFix (an img2img
        # refiner at `strength: 0.5`). It starts the trajectory from the encoded
        # ARTIFACT noised to `init_t` rather than from pure noise.
        #
        # This CANNOT be done by truncating the grid above, and the reason is
        # `time_dist_shift` (22.05 for this checkpoint). That warp is extremely
        # convex: 47 of the 50 default steps sit above t = 0.5, and the grid then
        # falls 0.665 -> 0.022 in its last few entries. So the two readings of
        # "strength 0.5" come apart completely here, where in SDXL they roughly
        # agree:
        #
        #   keep the last 50% of STEPS  -> starts at t = 0.955, i.e. 95% noise.
        #                                  The artifact contributes 4.5% and the
        #                                  arm is indistinguishable from arm A.
        #   start at NOISE LEVEL t=0.5  -> only 3 steps remain. Undersampled by
        #                                  8x against FreeFix's 25.
        #
        # Neither is the comparison we want. So the grid is REBUILT on the
        # sub-interval [t_end, init_t] with `init_steps` points, using GLD's own
        # warp so the spacing philosophy is unchanged -- solve `_warp(u) = init_t`
        # for the linear coordinate and lay the points out there. That gives
        # FreeFix's step count AT a genuine noise level, which is the only
        # version of this arm that tests what it claims to.
        if init_t is not None:
            n = int(init_steps or num_steps)
            if not 0.0 < init_t <= 1.0:
                raise ValueError(f"init_t must be in (0, 1], got {init_t}")
            # inverse of `_warp`
            u_hi = init_t / (time_dist_shift - init_t * (time_dist_shift - 1))
            u_lo = 1 - t1        # the linear coordinate the full grid ends at
            self.t = _warp(th.linspace(u_hi, u_lo, n))

        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type
        print(self.t)
    def sample(self, x, model, **model_kwargs) -> tuple[th.Tensor]:
        
        device = x[0].device if isinstance(x, tuple) else x.device
        def _fn(t, x):
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            model_output = self.drift(x, t, model, **model_kwargs)
            if self.blend_fn is not None:
                model_output = self.blend_fn(x, t, model_output)
            return model_output
        
        t = self.t.to(device)
        
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(
            _fn,
            x,
            t,
            method=self.sampler_type,
            atol=atol,
            rtol=rtol
        )
        return samples
