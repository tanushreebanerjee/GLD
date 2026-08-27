# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
DA3 MVDiffusion training script - trains diffusion model on DA3 feature space.
Supports level-specific training (-1, -2, -3, -4) for multi-level feature reconstruction.
"""
import os
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import json
import argparse
import itertools
import logging
import sys

import math
from torch.cuda.amp import autocast
from omegaconf import OmegaConf
from stage1 import RAE_DA3
from stage2.models import Stage2ModelProtocol
from stage2.transport.transport import Transport, ModelType, WeightType, PathType, Sampler
from utils.train_utils import (
    parse_configs,
    create_transport,
    update_ema,
    requires_grad,
    cleanup,
    create_logger,
    center_crop_arr,
)
from utils.model_utils import instantiate_from_config
from utils import wandb_utils
from utils.optim_utils import build_optimizer, build_scheduler
from utils.config_utils import init_config_defaults, get_image_size_from_config

# added import
from video.video_load import DA3VideoDataset_Pose, create_multiview_dataloader
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
from einops import rearrange
from disc import (
    LPIPS
)
import torch.nn.functional as F
from utils.metrics import compute_psnr, compute_ssim, compute_lpips
import wandb

from utils.camera.camera import get_camera_embedding
from datetime import datetime

def _geofix_diag_zero():
    """Zeroed accumulators for the mask-split loss diagnostic (loss_edit /
    loss_keep). Observability only -- nothing here touches a gradient.

    `n` counts only the units (micro-batches, then optimizer steps) that
    actually carried a mask. Averaging by `n` rather than by the total keeps a
    mask-free unit from diluting the number, and `n == 0` means "emit nothing"
    rather than "0.0": a logged zero would read as "no error in the keep
    region", which is false, not merely unmeasured.
    """
    return {"edit": 0.0, "keep": 0.0, "frac": 0.0, "n": 0}


def prepare_data(
    rae, images, intrinsic, extrinsic, device,
    random_cond_num: int = 1,
    return_cls: bool = False,
    camera_mode: str = "camray",
    return_scale: bool = False,
    artifact_images=None,
    geofix_mask=None,
    cond_artifact: bool = True,
    return_artifact: bool = False,
):
    """
    Prepare data for training.
    For DA3: Image 518x518, camera embedding at 518x518, which gets patchified to 37x37 by camera_embedder.
    Args:
        camera_mode: "camray" for direction only (3ch), "plucker" for [d, o×d] (6ch)
        return_scale: Whether to return the translation normalization scale factor.

    GeoFix (both arguments optional; unset reproduces stock GLD exactly):

        artifact_images: (B, V, 3, H, W) 3DGS renders. Fills
            `latents_cond[:, cond_num:]`, which stock GLD leaves as ZEROS -- so
            without this the degraded render never enters the network at all and
            there is nothing to refine. `images` must then be the CLEAN frames,
            because `latents_all` is the flow TARGET. Two encodes, deliberately:
            the model is conditioned on the artifact and supervised toward clean.

        geofix_mask: (B, V, 1, g, g) `edit1` mask on the token grid, 1 = "repair
            here". Replaces the constant 1.0 that target views carry in camera
            channel 0. Nearest-upsampled to (H, W) so each token maps onto its own
            14x14 patch and survives the camera_embedder's patchify exactly.

        cond_artifact: whether `artifact_images` fills `latents_cond[:, cond_num:]`.
            Split out from `artifact_images` so LATENT BRIDGE MATCHING can ask for
            the artifact latents WITHOUT the conditioning slot -- a pure bridge
            (x0 = artifact features, nothing in slot 1) is the arm that says
            whether the bridge REPLACES conditioning or merely accompanies it.
            Default True, which is exactly the pre-bridge behaviour.

        return_artifact: append `latents_art` to the return tuple. It is encoded
            here, in the SAME normalised space as `latents_all` (that is the whole
            reason it is computed inside this function rather than at the call
            site), and a bridge whose x0 lives in a different space than its x1 is
            the failure mode this project keeps catching. Returning it is strictly
            cheaper and safer than re-encoding in the training loop.
    """
    # B, V, C, H, W = images.shape # Removed resize logic as per user request
    B, V, C, H, W = images.shape
    # Move inputs to device immediately
    images = images.to(device, non_blocking=True)
    intrinsic = intrinsic.to(device, non_blocking=True)
    extrinsic = extrinsic.to(device, non_blocking=True)
    # -------------------------------------------------------------------------
    # Encode Images
    # -------------------------------------------------------------------------
    # ImageNet Normalization is required for DA3 encoder
    # Using buffers from rae (1, 3, 1, 1) -> (1, 1, 3, 1, 1) for broadcasting
    images_norm = (images - rae.encoder_mean[None]) / rae.encoder_std[None]
    # 1. encode image - RAE_DA3 now accepts 5D input (B, V, C, H, W) directly
    with torch.no_grad():
        if random_cond_num is None:
            raise ValueError("random_cond_num (cond_num) must be provided explicitly.")
        cond_views = int(random_cond_num)
        if cond_views < 0:
            raise ValueError(f"cond_num must be >= 0, got {cond_views}")
        if cond_views > V:
            raise ValueError(
                f"cond_num exceeds number of views in batch: cond_num={cond_views}, V={V}. "
                "Fix your multiview config or dataloader."
            )
        
        # A. Encode ALL views together (to get Target GTs with cross-view info)
        if return_cls:
            latents_all, cls_all = rae.encode(images_norm, return_cls=True)
            # Reshape and concatenate CLS token: (BV, C, H, W) + (BV, C) -> (BV, C, N+1, 1)
            BV, C_lat, h_lat, w_lat = latents_all.shape
            latents_all = torch.cat([cls_all.reshape(BV, C_lat, 1, 1), 
                                    latents_all.reshape(BV, C_lat, h_lat * w_lat, 1)], dim=2)
        else:
            latents_all = rae.encode(images_norm)
            cls_all = None
            
        # B. Encode ONLY Reference views (to get leakage-free conditioning)
        if cond_views > 0:
            if return_cls:
                latents_ref, cls_ref = rae.encode(images_norm[:, :cond_views], return_cls=True)

                BrV = latents_ref.shape[0]
                latents_ref = torch.cat([cls_ref.reshape(BrV, C_lat, 1, 1), 
                                        latents_ref.reshape(BrV, C_lat, h_lat * w_lat, 1)], dim=2)
            else:
                latents_ref = rae.encode(images_norm[:, :cond_views])
        else:
            latents_ref = None

        # B2. GeoFix: encode the ARTIFACT renders, per view, for the target slots.
        # `mode='single'`-equivalent framing does not apply here: we deliberately
        # encode them the same way `latents_all` is encoded, so the conditioning
        # and the target live in the same normalized space.
        latents_art = None
        if artifact_images is not None:
            if artifact_images.shape != images.shape:
                raise ValueError(
                    f"artifact_images {tuple(artifact_images.shape)} must match "
                    f"images {tuple(images.shape)}.")
            art_norm = (artifact_images.to(device, non_blocking=True)
                        - rae.encoder_mean[None]) / rae.encoder_std[None]
            if return_cls:
                latents_art, cls_art = rae.encode(art_norm, return_cls=True)
                BVa, C_a, h_a, w_a = latents_art.shape
                latents_art = torch.cat(
                    [cls_art.reshape(BVa, C_a, 1, 1),
                     latents_art.reshape(BVa, C_a, h_a * w_a, 1)], dim=2)
            else:
                latents_art = rae.encode(art_norm)

        # C. Merge: Reference part from 'ref-only' pass, Target part from 'all' pass
    # latents_all is (B*V, C, ...)
    #
    # `latents_cond` USED TO BE BUILT HERE TOO, and that copy was dead: the block
    # near the end of this function ("Construct latents_cond") re-zeros and
    # reassigns it unconditionally in BOTH the sequence and the spatial branch,
    # so nothing between the two ever read the first result. Deleted 2026-08-27.
    #
    # It was not merely redundant. In the sequence branch the assignment sat
    # INSIDE `if latents_ref is not None:`, so `cond_views == 0` plus the packed
    # CLS format left `latents_cond` unbound and the next reader would have hit a
    # NameError rather than a shaped error -- a latent crash that only the
    # zero-reference configuration could trigger, which is exactly the
    # configuration nothing in session 8 runs. Removing the block removes the
    # trap; the surviving block zero-initialises before the `if`, so it is safe
    # at cond_views == 0.
    #
    # `BV` stays: the surviving block and the CLS reshapes below both read it.
    BV = latents_all.shape[0]

    # 2. get camera embedding at IMAGE resolution
    # The camera_embedder in DiT model will patchify this to match latent size
    if extrinsic.shape[-2:] == (4, 4):
        extrinsic = extrinsic[..., :3, :4]
    elif extrinsic.shape[-2:] != (3, 4):
        raise ValueError(f"Unexpected extrinsic shape {extrinsic.shape}.")

    if intrinsic.shape[-1] == 4:
        fx, fy, cx, cy = intrinsic.unbind(dim=-1)
        zeros = torch.zeros_like(fx)
        ones = torch.ones_like(fx)
        intrinsic_mat = torch.stack(
            (
                torch.stack((fx, zeros, cx), dim=-1),
                torch.stack((zeros, fy, cy), dim=-1),
                torch.stack((zeros, zeros, ones), dim=-1),
            ),
            dim=-2,
        )
    elif intrinsic.shape[-2:] == (3, 3):
        intrinsic_mat = intrinsic
    else:
        raise ValueError(f"Unexpected intrinsic shape {intrinsic.shape}.")

    extri_ = rearrange(extrinsic, "b f c1 c2 -> (b f) c1 c2", f=V)
    intri_ = rearrange(intrinsic_mat, "b f c1 c2 -> (b f) c1 c2", f=V)

    # Generate camera embedding at IMAGE resolution (H, W)
    # camera_embedder in DiT will patchify to latent resolution
    # camera_mode: "camray" = direction only (3ch), "plucker" = [d, o×d] (6ch)
    if return_scale:
        camera_embedding, scale = get_camera_embedding(intri_, extri_, B, V, H, W, mode=camera_mode, return_scale=True)
    else:
        camera_embedding = get_camera_embedding(intri_, extri_, B, V, H, W, mode=camera_mode)
        scale = None
        
    camera_embedding = rearrange(camera_embedding, "b f c h w -> (b f) c h w")

    # 3. make mask for camera distinguishing cond or tgt (at IMAGE resolution)
    if random_cond_num is None:
        raise ValueError("random_cond_num (cond_num) must be provided explicitly.")
    cond_views = int(random_cond_num)
    if cond_views < 0:
        raise ValueError(f"cond_num must be >= 0, got {cond_views}")
    if cond_views > V:
        raise ValueError(
            f"cond_num exceeds number of views in batch: cond_num={cond_views}, V={V}. "
            "Fix your multiview config or dataloader."
        )
    random_masks = torch.ones((B, V, 1, H, W), device=device, dtype=latents_all.dtype)
    random_masks[:, :cond_views] = 0
    # GeoFix slot 2: grade the target half by M_edit instead of a constant 1.0.
    # Polarity already agrees -- channel 0 = 1 means "this is a target, generate
    # it", M_edit = 1 means "repair here" -- so this enters with NO sign flip,
    # the only mask in the project that does (hard rule 7).
    #
    # NOTE the semantic overload this creates, because it bounds what slot 2 can
    # do alone: a target token driven to 0 now looks like a REFERENCE token, but
    # reference views also carry real content in `latents_cond` while target
    # views carry zeros unless slot 1 is on. Slot 2 alone therefore says
    # "preserve" while supplying nothing to preserve. The slots are
    # complementary, not alternatives, and `mask_only` exists to measure that
    # rather than to be deployed.
    if geofix_mask is not None:
        m = geofix_mask.to(device=device, dtype=latents_all.dtype)
        if m.ndim != 5 or m.shape[0] != B or m.shape[1] != V or m.shape[2] != 1:
            raise ValueError(
                f"geofix_mask must be (B={B}, V={V}, 1, g, g), got {tuple(m.shape)}.")
        # Nearest, so one token -> exactly one 14x14 patch. Bilinear would blur
        # values across patch boundaries and MAX pooling (hard rule 6) would then
        # have been pointless.
        m_img = torch.nn.functional.interpolate(
            m.reshape(B * V, 1, m.shape[3], m.shape[4]), size=(H, W), mode="nearest"
        ).reshape(B, V, 1, H, W)
        random_masks[:, cond_views:] = m_img[:, cond_views:]
    random_masks = random_masks.reshape(B * V, 1, H, W)

    camera_embedding = torch.cat([random_masks, camera_embedding], dim=1)  # (B*V, 7, H, W)

    # Return (latents_cond, latents_all, camera_embedding)
    # 1. latents_cond: [ref from ref-only pass | zeros for tgt]
    # 2. latents_all: all views from all-pass (GT for global features)
    # 3. camera_embedding: plucker/camray + mask

    # Construct latents_cond (handles both spatial and sequence/packed formats)
    if latents_all.ndim == 4 and latents_all.shape[3] == 1:
        # Sequence/packed format: (BV, C, K+N, 1)
        _, C_lat, seq_len, _ = latents_all.shape
        latents_cond_5d = torch.zeros(B, V, C_lat, seq_len, 1, device=device, dtype=latents_all.dtype)
        if latents_ref is not None:
            latents_ref_5d = latents_ref.reshape(B, cond_views, C_lat, seq_len, 1)
            latents_cond_5d[:, :cond_views] = latents_ref_5d
        # GeoFix slot 1: the target half stops being zeros and carries the render.
        if latents_art is not None and cond_artifact:
            latents_cond_5d[:, cond_views:] = latents_art.reshape(
                B, V, C_lat, seq_len, 1)[:, cond_views:]
        latents_cond = latents_cond_5d.reshape(BV, C_lat, seq_len, 1)
    else:
        # Spatial format: (BV, C, h, w)
        _, C_lat, h_lat, w_lat = latents_all.shape
        latents_cond_5d = torch.zeros(B, V, C_lat, h_lat, w_lat, device=device, dtype=latents_all.dtype)
        if latents_ref is not None:
            latents_ref_5d = latents_ref.reshape(B, cond_views, C_lat, h_lat, w_lat)
            latents_cond_5d[:, :cond_views] = latents_ref_5d
        # GeoFix slot 1: the target half stops being zeros and carries the render.
        if latents_art is not None and cond_artifact:
            latents_cond_5d[:, cond_views:] = latents_art.reshape(
                B, V, C_lat, h_lat, w_lat)[:, cond_views:]
        latents_cond = latents_cond_5d.reshape(BV, C_lat, h_lat, w_lat)

    if return_artifact:
        if latents_art is None:
            raise ValueError(
                "return_artifact=True but artifact_images was not given, so there "
                "are no artifact latents to bridge from. Returning None here would "
                "make the caller silently fall back to a noise x0 -- i.e. train a "
                "plain flow model under a bridge config.")
        return latents_cond, latents_all, camera_embedding, latents_art
    return latents_cond, latents_all, camera_embedding

def make_vis_multiview(gt, recon):
    """
    gt, recon: shape (2, C, H, W)
    returns numpy image: (H*2, W*2, C)
    """

    rows = []
    for i in range(2):    # two views
        # (C, H, W) → (H, W, C)
        gt_i = gt[i].permute(1, 2, 0)
        rc_i = recon[i].permute(1, 2, 0)
        
        # horizontal concat: (H, W*2, C)
        row = torch.cat([gt_i, rc_i], dim=1)
        rows.append(row)

    # vertical concat final output: (H*2, W*2, C)
    grid = torch.cat(rows, dim=0)

    return grid.cpu().numpy()


################################################################################
#                                  Training Loop                                #
################################################################################


def main(args):
    """Trains a new SiT model using config-driven hyperparameters."""
    if not torch.cuda.is_available():
        raise RuntimeError("Training currently requires at least one GPU.")
    cfg = OmegaConf.load(args.config)
    (
        rae_config,
        model_config,
        transport_config,
        sampler_config,
        guidance_config_top,
        pag_config_top,
        misc_config,
        training_config,
        validation_config,
        multiview_config,
        dataset_config,
    ) = parse_configs(args.config)
    # ------------------------------------------------------------------
    # SCENE-COUNT SCALING CURVE. Each point trains on a different SUBSET of the
    # same manifest, so the manifest is the one thing that must vary between arms
    # -- and copying the config per arm is the drift hazard the --geofix-no-mask
    # comment warns about: a second file lets anything else move with it.
    #
    # This is applied to `dataset_config` AFTER parse_configs, which is why a
    # generic `--set dataset.manifest=...` would not work: parse_configs re-reads
    # the config from disk, so an override on `cfg` never reaches this object.
    # `val_manifest` is deliberately NOT touched -- every point on the curve must
    # be validated against the identical held-out set or the curve is not a curve.
    # ------------------------------------------------------------------
    if getattr(args, "geofix_manifest", None):
        if dataset_config is None:
            raise ValueError("--geofix-manifest given but the config has no dataset block.")
        if not Path(args.geofix_manifest).is_file():
            raise ValueError(f"--geofix-manifest {args.geofix_manifest} does not exist.")
        print(f"[GeoFix] train manifest OVERRIDE: {dataset_config.get('manifest')} "
              f"-> {args.geofix_manifest}")
        dataset_config["manifest"] = args.geofix_manifest
    # Initialize derived config defaults (auto-compute input_size, cam_*, latent_size, etc.)
    # Supports 2D encoder sizes (H, W)
    # Note: init_config_defaults returns RAE encoder size. We use this as default for dataloader
    # unless dataset config explicitly overrides it.
    # Auto-detect backbone type from config target
    rae_target = rae_config.get('target', '')
    is_da3 = 'rae_da3' in rae_target.lower() or 'rae_vggt' in rae_target.lower()
    encoder_h, encoder_w = init_config_defaults(
        rae_config, model_config, misc_config, patch_size=14, is_da3=is_da3
    )

    # Dataset configuration with defaults for backward compatibility
    dataset_cfg = dataset_config or {}
    dataset_name = dataset_cfg.get("name", "re10k")
    train_video_path = dataset_cfg.get("train_video_path", "data/re10k/training_256")
    train_pose_path = dataset_cfg.get("train_pose_path", "data/re10k/training_poses")
    val_video_path = dataset_cfg.get("val_video_path", "data/re10k/test_256")
    val_pose_path = dataset_cfg.get("val_pose_path", "data/re10k/test_poses")

    # Dataset configuration with defaults for backward compatibility
    dataset_cfg = dataset_config or {}
    dataset_name = dataset_cfg.get("name", "re10k")
    train_video_path = dataset_cfg.get("train_video_path", "data/re10k/training_256")
    train_pose_path = dataset_cfg.get("train_pose_path", "data/re10k/training_poses")
    val_video_path = dataset_cfg.get("val_video_path", "data/re10k/test_256")
    val_pose_path = dataset_cfg.get("val_pose_path", "data/re10k/test_poses")

    # Determine Dataloader Image Size (STRICT MODE)
    # Priority:
    # 1. dataset.image_size (Supports int, [h,w], or list of candidates)
    # NO FALLBACKS allowed.
    if "image_size" not in dataset_cfg:
        raise ValueError(
            "Config validation failed: 'dataset.image_size' is missing. "
            "Please explicitly specify 'image_size' in the 'dataset' section of your config. "
            "It can be an integer, a [h, w] list, or a list of candidate resolutions."
        )
        
    dl_image_size = dataset_cfg["image_size"]
    if OmegaConf.is_config(dl_image_size):
        dl_image_size = OmegaConf.to_container(dl_image_size, resolve=True)

    # Prepare dataset kwargs for create_multiview_dataloader
    # Exclude 'image_size' as it's passed explicitly
    dataset_kwargs = OmegaConf.to_container(dataset_cfg, resolve=True)
    # Filter out keys that are explicitly passed to create_multiview_dataloader or not needed by dataset
    for key in ["name", "image_size", "num_views", "cond_num", "ref_view_sampling", "camera_mode", "normalize_pose"]:
        if key in dataset_kwargs:
            del dataset_kwargs[key]

    # Update args for logging (best effort)
    if isinstance(dl_image_size, (int, float)):
        args.image_size = int(dl_image_size)
        args.image_height = int(dl_image_size)
        args.image_width = int(dl_image_size)
    elif isinstance(dl_image_size, (list, tuple)) and len(dl_image_size) == 2 and isinstance(dl_image_size[0], (int, float)):
        # Single rectangular resolution
        args.image_height = int(dl_image_size[0])
        args.image_width = int(dl_image_size[1])
        args.image_size = args.image_height # Ambiguous but set to H
    else:
        # List of candidates or complex structure
        args.image_size = dl_image_size 
        args.image_height = -1 # Indicating dynamic/multiple
        args.image_width = -1

    if rae_config is None or model_config is None:
        raise ValueError("Config must provide both stage_1 and stage_2 sections.")

    def to_dict(cfg_section):
        if cfg_section is None:
            return {}
        if not OmegaConf.is_config(cfg_section):
            return cfg_section
        return OmegaConf.to_container(cfg_section, resolve=True)

    misc = to_dict(misc_config)
    transport_cfg = to_dict(transport_config)
    sampler_cfg = to_dict(sampler_config)
    training_cfg = to_dict(training_config)
    validation_cfg = to_dict(validation_config)
    multiview_cfg = to_dict(multiview_config)
    # Guidance and PAG (Prioritize validation section, then top-level)
    guidance_cfg = to_dict(validation_cfg.get("guidance", guidance_config_top))
    pag_cfg = to_dict(validation_cfg.get("pag", pag_config_top))
    #
    use_prope = False
    if 'use_prope' in model_config.get('params', {}):
        if model_config.params.use_prope:
            use_prope = True
            multiview_cfg['use_prope'] = True
            # logger.info("Enable ProPE from Config") # Logger not init yet
            print("Enable ProPE from Config")
    # Update args for consistency (optional but good for logging)
    args.use_prope = use_prope

    # training config parameters
    global_batch_size = int(training_cfg.get("global_batch_size", 256))
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    # Overridden from the command line so ONE config can run on both the 48G A6000
    # and the 141G H200 without a second copy that could drift in some other key.
    # These change only how the same effective batch is split across steps and
    # devices, never what the batch IS -- keep global_batch_size fixed across arms
    # or the arms are not comparable.
    if getattr(args, "global_batch_size", None) is not None:
        global_batch_size = int(args.global_batch_size)
    if getattr(args, "grad_accum_steps", None) is not None:
        grad_accum_steps = int(args.grad_accum_steps)
    num_epochs = int(training_cfg.get("epochs", 100))
    ema_decay = float(training_cfg.get("ema_decay", 0.9999))
    num_workers = int(training_cfg.get("num_workers", 4))
    log_every = int(training_cfg.get("log_every", 100))
    clip_grad = float(training_cfg.get("clip_grad", 1.0))
    default_seed = int(training_cfg.get("global_seed", 0))

    # validation config parameters
    ckpt_every = int(validation_cfg.get("ckpt_every", training_cfg.get("ckpt_every", 5000)))
    sample_every = int(validation_cfg.get("sample_every", training_cfg.get("sample_every", 100)))
    val_num_batches = validation_cfg.get("val_num_batches", None)
    val_mode = validation_cfg.get("validation_mode", "propagation") 
    # Overrides
    cfg_scale_override = training_cfg.get("cfg_scale", None) 
    # minkyung: multi-view config (can also be in dataset section)
    total_view = int(multiview_cfg.get("total_view", dataset_cfg.get("num_views", 2)))
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))


    #
    # SNR' = SNR / V 효과를 위해 shift_dim = (H*W*C) * V 로 설정
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size) * total_view)
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    # Dynamic cond_num: supports int or string range like "2-4"
    cond_num_raw = multiview_cfg.get("cond_num", dataset_cfg.get("cond_num", 2))
    if isinstance(cond_num_raw, str) and "-" in cond_num_raw:
        cond_num_min, cond_num_max = map(int, cond_num_raw.split("-"))
        cond_num = None  # Will be sampled per batch
    else:
        cond_num = int(cond_num_raw)
        cond_num_min, cond_num_max = cond_num, cond_num
    if "ref_view_sampling" not in multiview_cfg and "ref_view_sampling" not in dataset_cfg:
        raise ValueError("Config must set multiview.ref_view_sampling or dataset.ref_view_sampling explicitly (prefix|random|interpolate).")
    ref_view_sampling = multiview_cfg.get("ref_view_sampling", dataset_cfg.get("ref_view_sampling", "prefix"))
    if not isinstance(ref_view_sampling, str):
        raise TypeError(f"multiview.ref_view_sampling must be str, got {type(ref_view_sampling)}")
    ref_view_sampling = ref_view_sampling.lower()
    if ref_view_sampling not in {"prefix", "random", "interpolate"}:
        raise ValueError(
            f"Unknown multiview.ref_view_sampling={ref_view_sampling}. Use one of: prefix, random, interpolate."
        )
    use_prope = bool(multiview_cfg.get("use_prope", False))
    camera_mode = multiview_cfg.get("camera_mode", dataset_cfg.get("camera_mode", "camray")).lower()
    if camera_mode not in {"camray", "plucker"}:
        raise ValueError(f"Unknown camera_mode={camera_mode}. Use 'camray' or 'plucker'.")

    # ----------------------------------------------------------------------
    # GeoFix: the two conditioning slots that already exist in this checkpoint
    # (docs/ARCH_NOTES.md "two conditioning slots ... need no widening").
    # Both default OFF, so every stock GLD config trains exactly as before.
    #   cond_artifact  -> fill latents_cond[:, cond_num:] with render features
    #   mask_in_camera -> grade camera channel 0 by M_edit on the target half
    # The ablation ladder is these two booleans, one code path, so the rows are
    # comparable (docs/SESSION_8.md deliverable 2).
    # ----------------------------------------------------------------------
    geofix_cfg = cfg.get("geofix", {}) or {}
    geofix_cond_artifact = bool(geofix_cfg.get("cond_artifact", False))
    geofix_mask_in_camera = bool(geofix_cfg.get("mask_in_camera", False))
    # The ablation arm runs off the SAME config file with this flag, rather than a
    # second copy of it. Two files would let anything else drift between the arms,
    # and the one comparison that isolates the mask from the render is worthless if
    # a second thing moved with it. A generic `--set key=value` would NOT be safe
    # here: `parse_configs` re-loads the config from disk (line ~344), so an
    # override applied to `cfg` never reaches `dataset_config` or `model_config`.
    # This flag only touches a value read from `cfg` and nowhere else.
    if getattr(args, "geofix_no_mask", False):
        if not geofix_mask_in_camera:
            raise ValueError(
                "--geofix-no-mask given but geofix.mask_in_camera is already false "
                "in the config. The flag exists to run the ablation arm off the "
                "mask-ON config; passing it to a mask-OFF config means one of the "
                "two is not what you think it is.")
        geofix_mask_in_camera = False
    # Same contract for slot 1, added 2026-08-19 for the REDUNDANCY control.
    # The mask has been worth <= +0.039 dB at every checkpoint, and the leading
    # explanation is not that it is uninformative but that it is REDUNDANT: slot 1
    # already hands the model the artifact render, which contains the damage the
    # mask points at. Turning slot 1 off measures the mask's value when the render
    # is NOT supplied. If it is large there and ~0 with the render, the finding is
    # redundancy, which is a different -- and more interesting -- claim than
    # "the mask carries no information".
    if getattr(args, "geofix_no_cond_artifact", False):
        if not geofix_cond_artifact:
            raise ValueError(
                "--geofix-no-cond-artifact given but geofix.cond_artifact is already "
                "false in the config. The flag exists to run the control off the "
                "cond-ON config; passing it to a cond-OFF config means one of the "
                "two is not what you think it is.")
        geofix_cond_artifact = False
    # ----------------------------------------------------------------------
    # LATENT BRIDGE MATCHING. `geofix.bridge_x0: artifact` moves the x0 end of the
    # flow from `randn_like(x1)` to the ARTIFACT features, so the model learns
    # F_artifact -> F_clean instead of noise -> F_clean.
    #
    # Why this is a two-line change and not a new transport: `Transport.sample`
    # already takes `x0_override` and `training_multiview_losses` already takes
    # `x0=` (GLD uses them for its CROSS-LEVEL flow, `training.source_level`).
    # What is new here is same-level, different-image: x0 and x1 are both level-1
    # features, of the render and of the clean photograph respectively.
    #
    # NOISE. A noiseless bridge makes the map deterministic at t=1 and the flow
    # objective degenerates toward regression, so `geofix.bridge_noise_tau`
    # perturbs the start exactly the way GLD's own cross-level path does:
    # `x0 = (F_art + sigma*eps) / sqrt(1 + sigma^2)`, sigma ~ |N(0, tau^2)|. It is
    # a SEPARATE knob from `training.noise_tau_gt_feat` so a bridge run cannot
    # silently inherit a value tuned for the cross-level path.
    #
    # This is NOT the img2img trick that 2026-08-16 killed. That one anchored a
    # SAMPLER trained on `x_t = (1-t) F_clean + t eps` at artifact features it had
    # never seen, and lost ~1.2 dB the moment t left 1.0. Here the training path
    # and the sampling path agree by construction; whether that is worth anything
    # is the question, and it is why `CLAUDE.md` recorded bridge matching as set
    # aside for ABSENCE of evidence rather than ruled out.
    # ----------------------------------------------------------------------
    geofix_bridge_x0 = geofix_cfg.get("bridge_x0", None)
    if getattr(args, "geofix_bridge_x0", None) is not None:
        geofix_bridge_x0 = args.geofix_bridge_x0
    if getattr(args, "geofix_no_bridge_x0", False):
        if not geofix_bridge_x0:
            raise ValueError(
                "--geofix-no-bridge-x0 given but geofix.bridge_x0 is already unset. "
                "The flag exists to run the noise-x0 control off the BRIDGE config "
                "so nothing else can differ between the two arms.")
        geofix_bridge_x0 = None
    if geofix_bridge_x0 not in (None, "", "artifact"):
        raise ValueError(
            f"geofix.bridge_x0={geofix_bridge_x0!r}; the only implemented value is "
            "'artifact' (x0 = level-1 features of the 3DGS render).")
    geofix_bridge = geofix_bridge_x0 == "artifact"
    geofix_bridge_noise_tau = float(geofix_cfg.get("bridge_noise_tau", 0.0))
    # ------------------------------------------------------------------
    # THE MASK-MODULATED BRIDGE, and why it is a different experiment from
    # `mask_in_camera`.
    #
    # Slot 2 hands the mask to the network as an INPUT and lets it decide what to
    # do with it. Across four checkpoints that has been worth <= +0.065 dB, and
    # the leading explanation is redundancy: slot 1 already supplies the render,
    # which contains the damage the mask points at.
    #
    # A bridge gives the mask a job no input channel can do. The start of the
    # flow is now the render, so "how far does this token have to travel" is a
    # real per-token quantity -- and the mask is a guess at it. Scaling the START
    # NOISE by M_edit makes that guess structural rather than advisory:
    #
    #     sigma_i = tau * M_edit_i
    #     x0_i    = (F_art_i + sigma_i * eps_i) / sqrt(1 + sigma_i^2)
    #
    # M=0 ("clean, preserve here") gives a deterministic start the model can copy;
    # M=1 ("damaged, repair here") gives a noisy one it must generate through. It
    # is REPRODUCIBLE AT INFERENCE -- unlike anything that mixes in F_clean --
    # because the mask and the render are both available at test time, which is
    # the constraint that rules out the obvious alternatives.
    #
    # Independent of `mask_in_camera`: the 2x2 of {input, start-noise} is the
    # ablation, and collapsing them into one flag would make it unrunnable.
    # ------------------------------------------------------------------
    geofix_bridge_mask_noise = bool(geofix_cfg.get("bridge_mask_noise", False))
    if getattr(args, "geofix_bridge_mask_noise", False):
        geofix_bridge_mask_noise = True
    if getattr(args, "geofix_no_bridge_mask_noise", False):
        if not geofix_bridge_mask_noise:
            raise ValueError(
                "--geofix-no-bridge-mask-noise given but geofix.bridge_mask_noise is "
                "already false. The flag exists to run the control off the SAME "
                "config as the mask-modulated arm.")
        geofix_bridge_mask_noise = False
    # ------------------------------------------------------------------
    # LATENT BLENDING AT TRAINING TIME -- the third, and most literal, way the
    # mask can enter. Session 6.5's blend composites x_t toward F_artifact at
    # every solver step; there is no solver loop in training, but under a BRIDGE
    # the composite has an exact single-shot form, t_i = 1 - M_i (1 - t). The
    # derivation is in transport.training_multiview_losses, at the point of use.
    #
    # It needs a bridge for a reason that is not a technicality: with x0 ~ N(0, I)
    # the same algebra gives a spatially varying SNR, not a transport schedule.
    # ------------------------------------------------------------------
    # PHASE 3a: the mask-gated loss. lambda = 0 is bit-for-bit the unweighted
    # loss, so this is inert unless asked for.
    geofix_loss_keep_weight = float(geofix_cfg.get("loss_keep_weight", 0.0) or 0.0)
    if getattr(args, "geofix_loss_keep_weight", None) is not None:
        geofix_loss_keep_weight = float(args.geofix_loss_keep_weight)
    if geofix_loss_keep_weight < 0:
        raise ValueError(
            f"geofix.loss_keep_weight={geofix_loss_keep_weight} must be >= 0. A "
            "negative lambda would DOWNWEIGHT the preserve region -- the direction "
            "this project's own loss-split diagnostic says suppresses the "
            "behaviour we lack.")

    geofix_blend_train = bool(geofix_cfg.get("blend_train", False))
    if getattr(args, "geofix_blend_train", False):
        geofix_blend_train = True
    if getattr(args, "geofix_no_blend_train", False):
        if not geofix_blend_train:
            raise ValueError(
                "--geofix-no-blend-train given but geofix.blend_train is already "
                "false. The flag exists to run the control off the SAME config.")
        geofix_blend_train = False
    # It does NOT require a bridge. The composite targets the ARTIFACT latents --
    # passed to the transport as `geofix_blend_toward` rather than read off x0 --
    # which under a bridge happen to BE x0. Under the stock noise start it is
    # literally session 6.5's blend moved into the training loop, and that is the
    # direct test of why `geofix_blend_s29500` lost 0.457 dB: the model met the
    # composite for the first time at inference. Only the `t_i = 1 - M_i (1-t)`
    # reading is bridge-specific, not the operation.

    # ------------------------------------------------------------------
    # A MASK-CONSUMING ARM MAY NOT RUN ON A MASK-FREE MANIFEST.
    #
    # `geofix.train_manifest --coverage no_mask` builds sample lists over an
    # export tree that has no masks at all (the 100-scene scale-up,
    # `configs/data/dl3dv_scale100.yaml`), and records `masks_on_disk: false` to
    # say so. `GeoFixPairs._mask` returns ZEROS wherever `has_mask` is false, so
    # such a manifest is silently loadable by every mask arm -- and under `edit1`
    # an all-zero plane reads as "clean, preserve everywhere" on a frame the loss
    # then demands a repair of. That is contradictory supervision on 100% of
    # samples, and the only symptom would be a disappointing number that reads as
    # "the mask does not help at 100 scenes".
    #
    # Nothing else catches it: the manifest loads, the shapes are right, the
    # polarity field is right, and n_mask matches.
    # ------------------------------------------------------------------
    _mask_consumers = [n for n, on in (
        ("geofix.mask_in_camera", geofix_mask_in_camera),
        ("geofix.bridge_mask_noise", geofix_bridge_mask_noise),
        ("geofix.blend_train", geofix_blend_train),
        ("geofix.loss_keep_weight", geofix_loss_keep_weight != 0.0),
    ) if on]
    if _mask_consumers and (dataset_config or {}).get("name") == "geofix":
        _mpath = (dataset_config or {}).get("manifest")
        if _mpath and Path(_mpath).is_file():
            _m = json.loads(Path(_mpath).read_text())
            if _m.get("masks_on_disk", True) is False:
                raise ValueError(
                    f"{', '.join(_mask_consumers)} is on, but {Path(_mpath).name} was "
                    f"built with coverage={_m.get('coverage')!r} and records "
                    "masks_on_disk: false -- there is no mask file behind ANY of its "
                    f"{len(_m.get('samples', []))} samples. Every target would get an "
                    "all-zero plane, which under edit1 means 'clean, preserve here' "
                    "while the loss still demands a repair. Compute masks for this "
                    "tree, or run the arm with --geofix-no-mask.")

    if geofix_bridge_mask_noise:
        if not geofix_bridge:
            raise ValueError(
                "geofix.bridge_mask_noise needs geofix.bridge_x0=artifact: with a "
                "noise x0 there is no per-token transport distance to modulate, and "
                "scaling the noise would just be a spatially varying SNR.")
        if geofix_bridge_noise_tau <= 0:
            raise ValueError(
                f"geofix.bridge_mask_noise is on but bridge_noise_tau="
                f"{geofix_bridge_noise_tau}; sigma_i = tau * M_edit_i would be 0 "
                "everywhere and the arm would be a plain deterministic bridge.")
    # `source_level` is parsed further down (line ~638), so read it from the config
    # here rather than from the not-yet-bound local.
    if geofix_bridge and training_cfg.get("source_level", None) is not None:
        raise ValueError(
            "geofix.bridge_x0 and training.source_level both write the x0 end of "
            "the flow; the second would overwrite the first and the run would look "
            "like a bridge while being a cross-level flow. Pick one.")
    if geofix_bridge and geofix_bridge_noise_tau < 0:
        raise ValueError(
            f"geofix.bridge_noise_tau={geofix_bridge_noise_tau} must be >= 0.")

    # ----------------------------------------------------------------------
    # WHAT THE FLOW IS SUPERVISED TOWARD -- and the bug this flag exists to fix.
    #
    # `image = batch['gt_inp']`, and from the GeoFix loader `gt_inp` is the
    # ARTIFACT RENDER. The clean photograph is a separate key, `gt_clean`. Until
    # 2026-08-22 the swap to `gt_clean` fired ONLY when slot 1 was on, so
    # `--geofix-no-cond-artifact` did not merely remove the conditioning: it
    # silently moved the flow TARGET from the clean photograph to the degraded
    # render. Both `maskonly_*` runs (7292471/7292472) trained that way. They are
    # therefore "GLD finetuned to REPRODUCE 3DGS artifacts", which is not the
    # ft-only control anyone reading their name would assume, and the mistake is
    # invisible in every metric -- a model that reproduces the render scores
    # exactly as well as the render.
    #
    # `clean_target` is now explicit. It defaults to the historical behaviour
    # (on iff slot 1 or the bridge is on) so no completed run is reinterpreted,
    # and `--geofix-clean-target` turns it on independently. THE NVS CONTROL --
    # plain GLD finetuned on our 20 DL3DV scenes, no render conditioning, no mask,
    # supervised toward clean novel views -- is exactly
    #     --geofix-no-cond-artifact --geofix-no-mask --geofix-clean-target
    # and it is the baseline that separates "finetuning on this data" from "the
    # conditioning signal" in every session-8/9 number.
    # ----------------------------------------------------------------------
    geofix_clean_target = geofix_cfg.get("clean_target", None)
    if getattr(args, "geofix_clean_target", False):
        geofix_clean_target = True
    if geofix_clean_target is None:
        # `blend_train` belongs in this default for the same reason the other two
        # do: it composites x_t toward the ARTIFACT latents, which `prepare_data`
        # only produces when `artifact_images` is passed, which only happens under
        # `geofix_clean_target`. Omitting it here (the state before 2026-08-27)
        # did not merely mis-default -- a `blend_train: true` arm with
        # cond_artifact and bridge_x0 both off died several hundred lines later
        # inside `prepare_data` with "return_artifact=True but artifact_images was
        # not given", which names the symptom and not the flag that caused it.
        geofix_clean_target = geofix_cond_artifact or geofix_bridge or geofix_blend_train
    geofix_clean_target = bool(geofix_clean_target)
    if (geofix_cond_artifact or geofix_bridge or geofix_blend_train) and not geofix_clean_target:
        raise ValueError(
            "geofix.clean_target=false with cond_artifact/bridge_x0/blend_train on "
            "would condition the model on the render AND supervise it toward the "
            "render, i.e. train an identity map that scores like the input.")

    if (geofix_cond_artifact or geofix_mask_in_camera or geofix_bridge
            or geofix_clean_target) and dataset_name.lower() != "geofix":
        raise ValueError(
            f"geofix.* conditioning is on but dataset.name={dataset_name!r}. "
            "Only the GeoFix loader supplies 'gt_clean' and 'mask'.")
    # All datasets must provide OpenCV c2w at load time
    # Feature-to-Feature Flow Matching: Optional source level conditioning
    # source_level: If set, use features from this level (+ noise) as x0 instead of pure noise
    source_level = training_cfg.get("source_level", None)
    if source_level is not None:
        source_level = int(source_level)
    # source_level_stat_path: Normalization stat path for SOURCE level features
    # IMPORTANT: Source and target levels have DIFFERENT distributions, must use different norm stats!
    source_level_stat_path = training_cfg.get("source_level_stat_path", None)
    if source_level is not None and source_level_stat_path is None:
        # Use print since logger is not initialized yet
        print(
            f"[WARNING] source_level={source_level} is set but source_level_stat_path is not specified! "
            "This may cause incorrect normalization for Feature-to-Feature Flow."
        )
    # Cache source level stats at startup (avoid per-batch disk I/O)
    _cached_source_stats = None
    if source_level is not None and source_level_stat_path is not None:
        _cached_source_stats = torch.load(source_level_stat_path, map_location='cpu')
        print(f"[INFO] Cached source level stats from: {source_level_stat_path}")
    # NEW: Feature Propagation Mode (Image 1 Architecture)
    # source_condition_level: If set, use features from this level as ENCODER CONDITION (not x0 init)
    # This is different from source_level which uses features for x0 initialization
    source_condition_level = training_cfg.get("source_condition_level", None)
    if source_condition_level is not None:
        source_condition_level = int(source_condition_level)
    source_condition_stat_path = training_cfg.get("source_condition_stat_path", None)
    if source_condition_level is not None and source_condition_stat_path is None:
        # Use print since logger is not initialized yet
        print(
            f"[WARNING] source_condition_level={source_condition_level} is set but source_condition_stat_path is not specified! "
            "This may cause incorrect normalization for Feature Propagation Mode."
        )
    # Cache source condition stats at startup (avoid per-batch disk I/O)
    _cached_source_cond_stats = None
    if source_condition_level is not None and source_condition_stat_path is not None:
        _cached_source_cond_stats = torch.load(source_condition_stat_path, map_location='cpu')
        print(f"[INFO] Cached source condition stats from: {source_condition_stat_path}")
    # Noise augmentation: sigma ~ |N(0, tau^2)| (following Dinh et al., Ho et al., Zhai et al.)
    # This smooths the latent distribution for better decoder generalization
    noise_tau_gt_feat = float(training_cfg.get("noise_tau_gt_feat", 0.0))
    global_seed = args.global_seed if args.global_seed is not None else default_seed

    if grad_accum_steps < 1:
        raise ValueError("Gradient accumulation steps must be >= 1.")
    # DA3 uses patch_size=14, so image dimensions must be divisible by 14
    # Skip this check for multi-resolution mode (indicated by -1)
    if args.image_height != -1 and args.image_width != -1:
        if args.image_height % 14 != 0 or args.image_width % 14 != 0:
            raise ValueError(
                f"Image dimensions ({args.image_height}x{args.image_width}) must be divisible by 14 for DA3."
            )
    #
    # if global_batch_size % (world_size * grad_accum_steps) != 0:
    #     raise ValueError(f"Global batch size {global_batch_size} must be divisible by world_size * grad_accum_steps ({world_size} * {grad_accum_steps} = {world_size * grad_accum_steps})")
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()

    rank = dist.get_rank()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    # global_batch_size is the total ACROSS accumulation, so this floor-divides to
    # 0 whenever it is smaller than world_size * grad_accum_steps -- and the
    # divisibility check that would have caught that is commented out just above.
    # The failure then surfaces far away as "batch_size should be a positive
    # integer value, but got batch_size=0" from the DataLoader.
    if micro_batch_size < 1:
        raise ValueError(
            f"micro_batch_size = global_batch_size // (world_size * grad_accum_steps) "
            f"= {global_batch_size} // ({world_size} * {grad_accum_steps}) = "
            f"{micro_batch_size}. Raise training.global_batch_size to at least "
            f"{world_size * grad_accum_steps}, or lower grad_accum_steps.")
    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)
    latent_dtype = autocast_kwargs["dtype"] if use_bf16 else torch.float32

    transport_params = dict(transport_cfg.get("params", {}))
    path_type = transport_params.get("path_type", "Linear")
    prediction = transport_params.get("prediction", "velocity")
    loss_weight = transport_params.get("loss_weight")
    transport_params.pop("time_dist_shift", None)
    is_concat_mode = model_config.get('params', {}).get('is_concat_mode', False)

    sampler_mode = sampler_cfg.get("mode", "ODE").upper()
    sampler_params = dict(sampler_cfg.get("params", {}))

    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    if cfg_scale_override is not None:
        guidance_scale = float(cfg_scale_override)
    guidance_method = guidance_cfg.get("method", "cfg")

    def guidance_value(key: str, default: float) -> float:
        if key in guidance_cfg:
            return guidance_cfg[key]
        dashed_key = key.replace("_", "-")
        return guidance_cfg.get(dashed_key, default)

    t_min = float(guidance_value("t_min", 0.0))
    t_max = float(guidance_value("t_max", 1.0))

    # PAG config (from validation or top level)
    pag_scale = float(pag_cfg.get("scale", 0.0))
    pag_layer_idx = pag_cfg.get("layer_idx", 22)

    experiment_name = None
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"{args.run_name}-{timestamp}"

    # Broadcast experiment_name to all ranks to ensure consistent experiment_dir
    # This prevents the "experiment_dir is None" crash on non-zero ranks
    name_obj = [experiment_name]

    dist.broadcast_object_list(name_obj, src=0)
    experiment_name = name_obj[0]
 
    experiment_dir = os.path.join(args.results_dir, experiment_name)

    if rank == 0:
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

    logger = create_logger(experiment_dir)
    if rank == 0:
        logger.info(f"Experiment directory created at {experiment_dir}")

    # ---------------------------------------------------------------------------------
    # 2. Stage-1 RAE (DA3) Setup
    # ---------------------------------------------------------------------------------
    #
    # Pass 'level' from stage2 config to stage1 config if present
    if 'params' in model_config and 'level' in model_config.params:
        if 'params' not in rae_config:
            rae_config.params = OmegaConf.create({})
        # Ensure we don't overwrite the entire params dict
        rae_config.params.level = model_config.params.level
        if rank == 0:
            logger.info(f"Passed level={rae_config.params.level} from Stage 2 to Stage 1 RAE.")

    # Inject cam_in_channels based on camera_mode (camray: 4ch, plucker: 7ch)
    cam_in_channels = 4 if camera_mode == "camray" else 7
    if 'params' not in model_config:
        model_config.params = OmegaConf.create({})
    model_config.params.cam_in_channels = cam_in_channels

    # ------------------------------------------------------------------
    # THE RESOLVED GeoFix SETTINGS, IN ONE PLACE.
    #
    # Every flag above is the result of a three-way resolution -- config default,
    # `geofix:` block, `--geofix-*` CLI override -- and until 2026-08-27 the ONLY
    # record of the outcome was the banner printed a few lines below. A log line
    # is a weak record: it lives beside the checkpoint rather than inside it, it
    # is lost when a run directory is moved or pruned, and it cannot be read at
    # all by anything holding only a `.pt` file. `arm_train_test_gate` had to
    # regex it, and regexed exactly one flag out of eight.
    #
    # So this dict is built ONCE and used TWICE -- it is printed as the banner and
    # stored in every checkpoint under the "geofix" key. Building it once is the
    # point: a banner and a checkpoint field maintained separately would drift,
    # and a parity gate reading a drifted record is worse than no gate. Anything
    # added here must be added to `arm_train_test_gate.CHECKS` as well, which is a
    # single dict entry by construction.
    #
    # `mask_types` / `pooling` / `gamma` come from the DATASET block, not the
    # `geofix:` one, because they are applied at LOAD time by `GeoFixPairs`. An
    # arm trained under one pooling and sampled under another is a different
    # conditioning signal, and nothing downstream would notice.
    # ------------------------------------------------------------------
    _ds = dataset_config or {}
    geofix_settings = {
        "cond_artifact": bool(geofix_cond_artifact),
        "mask_in_camera": bool(geofix_mask_in_camera),
        "clean_target": bool(geofix_clean_target),
        # The string, not the derived bool: 'artifact' / None. `geofix_bridge` is
        # `== "artifact"`, so the string is the strictly more informative of the
        # two and the gate can still derive the bool.
        "bridge_x0": geofix_bridge_x0 or None,
        "bridge_noise_tau": float(geofix_bridge_noise_tau),
        "bridge_mask_noise": bool(geofix_bridge_mask_noise),
        "blend_train": bool(geofix_blend_train),
        "loss_keep_weight": float(geofix_loss_keep_weight),
        "mask_types": list(_ds.get("mask_types", []) or []),
        "pooling": _ds.get("pooling", "max"),
        "gamma": float(_ds.get("gamma", 1.0)),
    }

    if rank == 0:
        logger.info(f"Camera mode: {camera_mode}, cam_in_channels: {cam_in_channels}")
        logger.info(
            f"[GeoFix] flow target = "
            f"{'gt_clean (clean photograph)' if geofix_settings['clean_target'] else 'gt_inp (ARTIFACT RENDER)'}"
            f"  bridge_x0={geofix_settings['bridge_x0'] or 'noise'}"
            f"  bridge_noise_tau={geofix_settings['bridge_noise_tau']}"
            f"  bridge_mask_noise={geofix_settings['bridge_mask_noise']}"
            f"  blend_train={geofix_settings['blend_train']}"
            f"  loss_keep_weight={geofix_settings['loss_keep_weight']}")
        logger.info(f"[GeoFix] cond_artifact={geofix_settings['cond_artifact']} "
                    f"mask_in_camera={geofix_settings['mask_in_camera']}")
        # The MASK's own three knobs. They are applied at LOAD time, so an arm
        # trained with one and scored with another is a different conditioning
        # signal that nothing downstream would notice -- the same parity gap the
        # bridge_x0 gate exists to close. The gate now covers them, from the
        # checkpoint; this line remains the only record for runs that predate the
        # checkpoint field, which is why it stays.
        logger.info(f"[GeoFix] mask_types={geofix_settings['mask_types']} "
                    f"pooling={geofix_settings['pooling']} "
                    f"gamma={geofix_settings['gamma']}")

    rae = instantiate_from_config(rae_config).to(device)

    rae.eval()
    lpips = LPIPS().to(device)
    lpips.eval()

    # Ensure model config is resolved to primitives (fixes ListConfig issues in ProPE)
    if OmegaConf.is_config(model_config):
        model_config = OmegaConf.to_container(model_config, resolve=True)

    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    opt_state = None
    sched_state = None
    train_steps = 0
    ckpt_meta = None

    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        # The RELEASED GLD checkpoints (SeonghuJeon/GLD, e.g. da3_level1.pt) ship
        # EMA weights only -- their single top-level key is "ema", with no "model".
        # The original `checkpoint["model"] if "model" in checkpoint else checkpoint`
        # then hands the whole {"ema": OrderedDict} down as if it were a state dict
        # and dies on `.to(bfloat16)`. Handled explicitly rather than by widening
        # the fallback, so an unrecognised layout still fails loudly instead of
        # silently loading nothing under strict=False.
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "ema" in checkpoint:
            state_dict = checkpoint["ema"]
            print("[pretrained] loading EMA weights (released GLD layout).")
        else:
            state_dict = checkpoint
        if not all(torch.is_tensor(v) for v in state_dict.values()):
            raise ValueError(
                f"{args.pretrained}: expected a flat tensor state dict, got keys "
                f"{list(state_dict)[:5]}. Unrecognised checkpoint layout.")
        state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("y_embedder.")
        }
        for k in state_dict:
            state_dict[k] = state_dict[k].to(torch.bfloat16)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # A finetune that silently loaded nothing is a from-scratch run wearing a
        # finetune's name, which hard rule 2 exists to prevent. `strict=False` is
        # kept (y_embedder is deliberately dropped) but the damage is bounded.
        if len(missing) > 20 or len(unexpected) > 20:
            raise ValueError(
                f"{args.pretrained}: {len(missing)} missing / {len(unexpected)} "
                f"unexpected keys -- this is not the architecture the checkpoint "
                f"was trained for. missing[:5]={missing[:5]} "
                f"unexpected[:5]={unexpected[:5]}")
        print(f"[pretrained] loaded {len(state_dict)} tensors; "
              f"{len(missing)} missing, {len(unexpected)} unexpected.")
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        ckpt_meta = checkpoint
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        opt_state = checkpoint.get("opt")
        sched_state = checkpoint.get("scheduler")
        train_steps = int(checkpoint.get("train_steps", 0))

    model_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {model_param_count/1e6:.2f}M")

    model = DDP(model, device_ids=[device_idx], gradient_as_bucket_view=False)

    opt, opt_msg = build_optimizer(model.parameters(), training_cfg)
    if opt_state is not None:
        opt.load_state_dict(opt_state)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    # dataset = ImageFolder(args.data_path, transform=transform)
    # sampler = DistributedSampler(
    #     dataset,
    #     num_replicas=world_size,
    #     rank=rank,
    #     shuffle=True,
    #     seed=global_seed,
    #     )
    # loader = DataLoader(
    #     dataset,
    #     batch_size=micro_batch_size,
    #     shuffle=False,
    #     sampler=sampler,
    #     num_workers=num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    if args.overfit: 
        #
        # Pass tuple (H, W) to ensure DummyMultiviewDataset picks the correct resolution
        # args.image_size might be int (legacy), but we want explicit (h, w)
        
        loader, sampler = create_multiview_dataloader(
            dataset_name=dataset_name,
            video_path=val_video_path,
            pose_path=val_pose_path,
            image_size=dl_image_size,
            num_views=total_view,
            cond_num=cond_num,
            ref_view_sampling=ref_view_sampling,
            batch_size=micro_batch_size,
            workers=num_workers,
            rank=rank,
            world_size=world_size,
            shuffle=False,
            overfit=True,
            desired_steps=1000,
        )
        val_loader, _ = create_multiview_dataloader(
            dataset_name=dataset_name,
            video_path=val_video_path,
            pose_path=val_pose_path,
            image_size=dl_image_size,
            num_views=total_view,
            cond_num=cond_num,
            ref_view_sampling=ref_view_sampling,
            batch_size=micro_batch_size,
            workers=num_workers,
            rank=rank,
            world_size=world_size,
            shuffle=False,
            overfit=True,
            desired_steps=1,
            **dataset_kwargs,
        )
    else:
        #
        loader, sampler = create_multiview_dataloader(
            dataset_name=dataset_name,
            video_path=train_video_path,
            pose_path=train_pose_path,
            image_size=dl_image_size,
            num_views=total_view,
            cond_num=cond_num,
            ref_view_sampling=ref_view_sampling,
            batch_size=micro_batch_size,
            workers=num_workers,
            rank=rank,
            world_size=world_size,
            **dataset_kwargs,
        )
        #
        val_num_views = int(validation_cfg.get("num_views", total_view))
        val_cond_num = validation_cfg.get("cond_num", cond_num)
        if val_cond_num is None: # 가변 범위인 경우 중간값 사용
            val_cond_num = (cond_num_min + cond_num_max) // 2
        else:
            val_cond_num = int(val_cond_num)
        val_test_baseline = int(validation_cfg.get("test_baseline", 40))
        val_min_interval = validation_cfg.get("min_interval", 1)
        val_max_interval = validation_cfg.get("max_interval", val_test_baseline) # Fallback to test_baseline if max_interval not set
        val_seed = validation_cfg.get("seed", None)
        

        val_batch_size = int(validation_cfg.get("batch_size", 1))
        
        # 전체 55개를 보장하기 위해 각 rank가 수행할 batch 개수 계산
        total_val_batches = int(validation_cfg.get("val_num_batches", 55))
        val_num_batches_per_rank = total_val_batches // world_size
        if rank < (total_val_batches % world_size):
            val_num_batches_per_rank += 1

        # 검증 시에는 멀티 해상도 리스트 중 첫 번째 해상도로 고정
        val_image_size = dl_image_size[0] if isinstance(dl_image_size, (list, tuple)) else dl_image_size
        
        # Prepare validation-specific dataset kwargs
        val_dataset_kwargs = deepcopy(dataset_kwargs)
        val_dataset_kwargs.update({
            "min_interval": val_min_interval,
            "max_interval": val_max_interval,
        })

        val_loader, _ = create_multiview_dataloader(
            dataset_name=dataset_name,
            video_path=val_video_path,
            pose_path=val_pose_path,
            image_size=val_image_size,
            num_views=val_num_views,
            cond_num=val_cond_num,
            ref_view_sampling=ref_view_sampling,
            batch_size=val_batch_size,
            workers=num_workers,
            rank=rank,
            world_size=world_size,
            shuffle=False,
            mode='test',
            test_baseline=val_test_baseline,
            seed=val_seed,
            **val_dataset_kwargs,
        )
        
        
    # logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(
        f"Gradient accumulation: steps={grad_accum_steps}, micro batch={micro_batch_size}, "
        f"per-GPU batch={micro_batch_size * grad_accum_steps}, global batch={global_batch_size}"
    )
    logger.info(f"Precision mode: {args.precision}")
    loader_batches = len(loader)
    steps_per_epoch = loader_batches // grad_accum_steps
    # The ragged tail is DROPPED, and the epoch stops after a whole number of
    # optimizer steps. Before this, an indivisible loader length was a warning here
    # and a hard crash an hour later -- `RuntimeError: Gradient accumulation counter
    # not zero at epoch end` at line ~1661, which is the same fact discovered far
    # from its cause. It cost job 7264790 after 269 of 269 iterations.
    #
    # Dropping is the right resolution rather than demanding divisibility: GeoFix's
    # masked manifest is 269 samples and 269 is PRIME, so no grad_accum_steps above
    # 1 divides it and no config value could satisfy the check. The alternative --
    # stepping the optimizer on a partial accumulation -- would make one step per
    # epoch carry a different effective batch size than every other step, silently.
    #
    # `drop_last=True` on the loader already discards a ragged tail for the same
    # reason one level down; this is the same policy at the accumulation level.
    micro_batches_per_epoch = steps_per_epoch * grad_accum_steps
    if loader_batches != micro_batches_per_epoch and rank == 0:
        logger.warning(
            f"Dropping the ragged tail: {loader_batches} loader batches is not a "
            f"multiple of grad_accum_steps={grad_accum_steps}, so each epoch "
            f"consumes {micro_batches_per_epoch} and skips the last "
            f"{loader_batches - micro_batches_per_epoch}. Shuffling reshuffles "
            f"which samples land in the tail, so nothing is permanently unseen.")
    if steps_per_epoch <= 0:
        raise ValueError("Gradient accumulation configuration results in zero optimizer steps per epoch.")
    schedl, sched_msg = build_scheduler(opt, steps_per_epoch, training_cfg, sched_state)
    if rank == 0:
        logger.info(f"Training configured for {num_epochs} epochs, {steps_per_epoch} steps per epoch.")
        logger.info(opt_msg + "\n" + sched_msg)

    # ---------------------------------------------------------------------------------
    # Resume logic (epoch-aware)
    #
    # - We checkpoint `train_steps` (optimizer steps), but the dataset/sampler are epoch-based.
    # - To resume deterministically, map:
    #     start_epoch = train_steps // steps_per_epoch
    #     step_in_epoch = train_steps % steps_per_epoch
    #     micro_batches_to_skip = step_in_epoch * grad_accum_steps
    # ---------------------------------------------------------------------------------
    start_epoch = 0
    micro_batches_to_skip = 0
    if args.ckpt is not None and train_steps > 0:
        start_epoch = train_steps // steps_per_epoch
        step_in_epoch = train_steps % steps_per_epoch
        micro_batches_to_skip = int(step_in_epoch * grad_accum_steps)
        if start_epoch >= num_epochs:
            raise ValueError(
                f"Resume requested but train_steps={train_steps} maps to start_epoch={start_epoch} which is >= num_epochs={num_epochs}. "
                f"steps_per_epoch={steps_per_epoch}, grad_accum_steps={grad_accum_steps}."
            )
        if micro_batches_to_skip >= loader_batches:
            raise ValueError(
                f"Resume skip exceeds epoch length: micro_batches_to_skip={micro_batches_to_skip} >= loader_batches={loader_batches}. "
                f"train_steps={train_steps}, step_in_epoch={step_in_epoch}, steps_per_epoch={steps_per_epoch}, grad_accum_steps={grad_accum_steps}."
            )
        if rank == 0:
            logger.info(
                f"Resuming from ckpt: train_steps={train_steps} -> start_epoch={start_epoch}, "
                f"skip_micro_batches={micro_batches_to_skip} (step_in_epoch={step_in_epoch}, grad_accum_steps={grad_accum_steps})."
            )

    transport = create_transport(
        **transport_params,
        time_dist_shift=time_dist_shift,
    )
    transport_sampler = Sampler(transport)

    # ------------------------------------------------------------------
    # THE VALIDATION BLEND HOOK. `blend_train` composites x_t toward the artifact
    # latents inside `training_multiview_losses`; at SAMPLING time the same
    # operation can only reach the trajectory through `sample_ode`'s `blend_fn`,
    # which is baked in at construction. So the object is built here, once, and
    # `validate_da3_multiview` re-arms it with each batch's own features and mask.
    #
    # Without it a blend-trained checkpoint would be validated on an UNBLENDED
    # trajectory -- 776 complete views, no error, a plausible number, which is the
    # exact failure mode `geofix.tools.arm_train_test_gate` was written for after
    # it happened four times. Validation now refuses to run rather than produce
    # that number (`geofix_blend_hook is None` raises there).
    #
    # `None` for every other arm, and `sample_ode(blend_fn=None)` is the stock
    # path byte-for-byte (hard rule 9).
    # ------------------------------------------------------------------
    geofix_blend_hook = None
    if geofix_blend_train:
        if args.overfit:
            # Pre-existing gap, not one this introduces: the `--overfit` branch of
            # the dataloader setup above never binds `val_num_views` /
            # `val_cond_num`, so the validation call at the bottom of this file
            # would already NameError on them. Named here rather than allowed to
            # surface as a NameError two thousand lines away.
            raise NotImplementedError(
                "--overfit with geofix.blend_train: the overfit branch does not "
                "bind val_num_views/val_cond_num, so the blend hook cannot be told "
                "which views are references -- and it MUST know, because M_edit is "
                "forced to 1.0 there.")
        from stage2.transport.blending import LatentBlend
        geofix_blend_hook = LatentBlend(total_view=val_num_views, cond_num=val_cond_num)

    if sampler_mode == "ODE":
        # Original RAE code already flips time in ode.__init__ with `1 - linspace()`
        # So no need for reverse=True here
        eval_sampler = transport_sampler.sample_ode(
            **sampler_params, blend_fn=geofix_blend_hook)
    elif sampler_mode == "SDE":
        if geofix_blend_hook is not None:
            raise NotImplementedError(
                "geofix.blend_train with sampler_mode=SDE: `sample_sde` has no "
                "`blend_fn` hook, so validation would sample an unblended "
                "trajectory for a blend-trained model. Validate this arm under ODE "
                "or score it offline with geofix_infer.py --blend-mask.")
        eval_sampler = transport_sampler.sample_sde(**sampler_params)
    # elif sampler_mode == "ODE_MULTI":
    #     eval_sampler = transport_sampler.sample_ode_multiview(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")

    guid_model_forward = None
    if guidance_scale > 1.0 and guidance_method == "autoguidance":
        guidance_model_cfg = guidance_cfg.get("guidance_model")
        if guidance_model_cfg is None:
            raise ValueError("Please provide a guidance model config when using autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guidance_model_cfg).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    # Overfit mode: compute mean/var from first batch for normalization - REMOVED per user request
    # if args.overfit:
    #     logger.info("Overfit mode: Computing normalization statistics from first batch...")
    #     ... (logic removed to use global stats instead)

    log_steps = 0
    running_loss = 0.0
    running_ref_loss = 0.0
    running_tgt_loss = 0.0
    geofix_run = _geofix_diag_zero()   # GeoFix mask-split diagnostic
    wandb_initialized = False  # Track wandb initialization state for resume support
    start_time = time()

    # ys = torch.randint(num_classes, size=(micro_batch_size,), device=device)
    # using_cfg = guidance_scale > 1.0
    # n = ys.size(0)
    # zs = torch.randn(n, *latent_size, device=device, dtype=latent_dtype)

    # if using_cfg:
    #     zs = torch.cat([zs, zs], dim=0)
    #     y_null = torch.full((n,), null_label, device=device)
    #     ys = torch.cat([ys, y_null], dim=0)
    #     sample_model_kwargs = dict(
    #         y=ys,
    #         cfg_scale=guidance_scale,
    #         cfg_interval=(t_min, t_max),
    #     )
    #     if guidance_method == "autoguidance":
    #         if guid_model_forward is None:
    #             raise RuntimeError("Guidance model forward is not initialized.")
    #         sample_model_kwargs["additional_model_forward"] = guid_model_forward
    #         model_fn = ema.forward_with_autoguidance
    #     else:
    #         model_fn = ema.forward_with_cfg
    # else:
    #     sample_model_kwargs = dict(y=ys)
    #     model_fn = ema.forward

    logger.info(f"Training for {num_epochs} epochs...")
    stop_training = False
    for epoch in range(start_epoch, num_epochs):
        if stop_training:
            break
        model.train()
        
        # Fast resume: compute batches to skip for this epoch
        batches_to_skip_this_epoch = micro_batches_to_skip if epoch == start_epoch else 0
        
        # Check if sampler supports fast resume (CUT3R BatchedRandomSampler)
        use_fast_resume = False
        if hasattr(sampler, 'set_epoch'):
            import inspect
            sig = inspect.signature(sampler.set_epoch)
            if 'start_batch_idx' in sig.parameters:
                use_fast_resume = True
                sampler.set_epoch(epoch, start_batch_idx=batches_to_skip_this_epoch)
                if batches_to_skip_this_epoch > 0 and rank == 0:
                    logger.info(f"[Fast Resume] Skipping {batches_to_skip_this_epoch} batches via sampler.set_epoch()")
            else:
                sampler.set_epoch(epoch)
        
        # Also set epoch for dataset if available (required for CUT3R multi-res)
        if hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(epoch)
        elif hasattr(loader.dataset, "dataset") and hasattr(loader.dataset.dataset, "set_epoch"):
             loader.dataset.dataset.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        opt.zero_grad()
        accum_counter = 0
        step_loss_accum = 0.0
        step_ref_loss_accum = 0.0
        step_tgt_loss_accum = 0.0
        geofix_step = _geofix_diag_zero()
        if rank == 0:
        # Wrap the loader with tqdm on Rank 0 only
        # Set initial to show correct position after fast resume
            pbar = tqdm(loader, total=steps_per_epoch, initial=batches_to_skip_this_epoch, 
                       desc=f"Epoch {epoch}/{num_epochs}")
        else:
            # Use the raw loader on other ranks
            pbar = loader
        pbar_iter = iter(pbar)
        
        # Slow fallback: skip via next() if sampler doesn't support fast resume
        if not use_fast_resume and batches_to_skip_this_epoch > 0:
            if rank == 0:
                logger.info(f"[Slow Resume] Skipping {batches_to_skip_this_epoch} micro-batches via next()...")
            for _ in range(batches_to_skip_this_epoch):
                next(pbar_iter)
        
        # Clear skip count after first epoch
        micro_batches_to_skip = 0

        # Stop at a whole number of optimizer steps, dropping the ragged tail (see
        # `micro_batches_per_epoch` above). Counted from what this epoch will
        # actually consume, so a resumed epoch that already skipped
        # `batches_to_skip_this_epoch` still ends on the same boundary.
        pbar_iter = itertools.islice(
            pbar_iter, max(micro_batches_per_epoch - batches_to_skip_this_epoch, 0))

        for batch in pbar_iter:
            # Dynamic cond_num: sample per batch if range was specified
            if cond_num is None:
                import random
                batch_cond_num = random.randint(cond_num_min, cond_num_max)
            else:
                batch_cond_num = cond_num
            
            # CUT3R batch conversion (List[Dict] -> Dict)
            if isinstance(batch, list) and isinstance(batch[0], dict):
                from video.cut3r_adapter import convert_cut3r_batch
                batch = convert_cut3r_batch(batch, batch_cond_num, ref_view_sampling)
            
            # image_dict, dict_keys(['enc_inp', 'gt_inp', 'fxfycxcy_252', 'c2w', 'video_id', 'frame_indices'])
            image = batch['gt_inp']

            intrinsic = batch['fxfycxcy']
            extrinsic = batch['c2w']

            # ------------------------------------------------------------------
            # GeoFix conditioning. Both slots are off by default, so a stock GLD
            # run is byte-for-byte unaffected by any of this.
            #
            # `gt_inp` from the GeoFix loader is the ARTIFACT render and
            # `gt_clean` is the clean photograph. The flow target must be CLEAN
            # (that is what we supervise toward), so when slot 1 is on the two
            # swap roles: `image` becomes the clean frame and the render is
            # handed to prepare_data separately as conditioning.
            # ------------------------------------------------------------------
            artifact_images = None
            geofix_mask = None
            # The swap is the same for slot 1 and for the bridge -- both need the
            # render encoded and the CLEAN frame as the flow target -- so it fires
            # once for either, and never twice.
            if geofix_clean_target:
                if 'gt_clean' not in batch:
                    raise ValueError(
                        "geofix.clean_target is on but the batch has no 'gt_clean'. "
                        "Only the GeoFix loader supplies it; check dataset.name=geofix.")
                artifact_images = image      # the render, -> latents_cond target slots
                image = batch['gt_clean']    # the clean frame, -> flow target
            if not (geofix_cond_artifact or geofix_bridge or geofix_blend_train):
                # Nothing consumes the render: no slot-1 fill, no bridge x0. Keep it
                # out of prepare_data so the encode is not paid for and the arm is
                # provably render-free.
                artifact_images = None
            # `geofix_mask` is what reaches prepare_data (slot 2). The bridge's
            # start-noise modulation needs the SAME tensor but must not turn slot 2
            # on, so it is bound separately -- the 2x2 of {input, start-noise} only
            # exists if those two are independently controllable.
            geofix_mask_tokens = None
            if (geofix_mask_in_camera or geofix_bridge_mask_noise or geofix_blend_train
                    or geofix_loss_keep_weight != 0.0):
                if 'mask' not in batch:
                    raise ValueError(
                        "geofix.mask_in_camera / bridge_mask_noise / blend_train is "
                        "on but the batch has no 'mask'. Only the GeoFix loader "
                        "supplies it; check dataset.name=geofix.")
                geofix_mask_tokens = batch['mask']
                # ------------------------------------------------------------
                # THE REFERENCE SLOTS MUST BE ZERO. Every consumer downstream
                # slices `[:, cond_num:]` and is safe by construction -- slot 2
                # writes `random_masks[:, cond_views:]`, the loss diagnostic
                # takes `m[:, cond_num:]`, the training blend composites
                # `xt_5d[:, cond_num:]` -- with ONE exception:
                # `bridge_mask_noise` computes `sigma = tau * m` over all B*V
                # views, so a nonzero reference plane would inject start noise
                # into views that are supposed to be clean conditioning, and the
                # arm would quietly stop being the control it is named for.
                #
                # It holds today only because `GeoFixPairs` emits literal zeros
                # there, which is a property of the loader and not of this file.
                # Inference already asserts it (`geofix_infer.assert_ref_slots_zero`);
                # training asserted nothing until 2026-08-27, so the invariant
                # was enforced on exactly the half of the pipeline that could
                # afford to lose it. Message mirrors the inference one on
                # purpose: two lookalike checks with different wording drift.
                # ------------------------------------------------------------
                if geofix_mask_tokens[:, :batch_cond_num].abs().max() > 0:
                    raise ValueError(
                        "reference slots [0, cond_num) carry a nonzero mask; "
                        "GeoFixPairs emits zeros there and bridge_mask_noise's "
                        "sigma = tau * M_edit runs over ALL views, so a nonzero "
                        "reference plane would noise the clean conditioning "
                        f"views. cond_num={int(batch_cond_num)}, max="
                        f"{float(geofix_mask_tokens[:, :batch_cond_num].abs().max())}.")
            if geofix_mask_in_camera:
                geofix_mask = geofix_mask_tokens

            # ------------------------------------------------------------------
            # Fail loudly on view-count mismatches (critical for multiview runs)
            # ------------------------------------------------------------------
            if image.ndim != 5:
                raise ValueError(f"Expected gt_inp shape (B, V, C, H, W), got {tuple(image.shape)}")
            B_batch, V_batch = int(image.shape[0]), int(image.shape[1])
            if V_batch != int(total_view):
                raise ValueError(
                    f"Config/Dataloader mismatch: total_view={int(total_view)} but dataloader returned V={V_batch}. "
                    "This is unsafe for DDP and indicates a dataset/sampler bug. Fix the dataloader to always return fixed V."
                )
            if intrinsic.ndim != 3:
                raise ValueError(f"Expected fxfycxcy shape (B, V, 4), got {tuple(intrinsic.shape)}")
            if extrinsic.ndim != 4:
                raise ValueError(f"Expected c2w shape (B, V, 4, 4), got {tuple(extrinsic.shape)}")
            if int(intrinsic.shape[0]) != B_batch or int(intrinsic.shape[1]) != V_batch:
                raise ValueError(
                    f"Intrinsic shape mismatch: expected (B={B_batch}, V={V_batch}, 4) but got {tuple(intrinsic.shape)}"
                )
            if int(extrinsic.shape[0]) != B_batch or int(extrinsic.shape[1]) != V_batch:
                raise ValueError(
                    f"Extrinsic shape mismatch: expected (B={B_batch}, V={V_batch}, 4, 4) but got {tuple(extrinsic.shape)}"
                )
            if not (1 <= int(batch_cond_num) < int(total_view)):
                raise ValueError(
                    f"Invalid cond_num/total_view: expected 1 <= cond_num < total_view but got cond_num={batch_cond_num}, total_view={total_view}"
                )
            
            predict_cls = training_cfg.get('predict_cls', False) or model_config.get('params', {}).get('predict_cls', False)
            
            if predict_cls:
                # For now, following user's strict return signature.
                _prep = prepare_data(
                    rae, image, intrinsic, extrinsic, device,
                    random_cond_num=batch_cond_num, return_cls=True,
                    camera_mode=camera_mode, return_scale=use_prope,
                    artifact_images=artifact_images, geofix_mask=geofix_mask,
                    cond_artifact=geofix_cond_artifact,
                    return_artifact=(geofix_bridge or geofix_blend_train)
                )
                if geofix_bridge or geofix_blend_train:
                    x1_cond, x1_all, camera_embedding, geofix_latents_art = _prep
                else:
                    x1_cond, x1_all, camera_embedding = _prep
                    geofix_latents_art = None
            else:
                _prep = prepare_data(
                    rae, image, intrinsic, extrinsic, device,
                    random_cond_num=batch_cond_num, return_cls=False,
                    camera_mode=camera_mode, return_scale=use_prope,
                    artifact_images=artifact_images, geofix_mask=geofix_mask,
                    cond_artifact=geofix_cond_artifact,
                    return_artifact=(geofix_bridge or geofix_blend_train)
                )
                if geofix_bridge or geofix_blend_train:
                    x1_cond, x1_all, camera_embedding, geofix_latents_art = _prep
                else:
                    x1_cond, x1_all, camera_embedding = _prep
                    geofix_latents_art = None
            
            # Ensure cond_num is set to actual sampled value for model_kwargs
            cond_num_for_model = int(batch_cond_num)
            
            # Expected channels: 1 (mask) + 3 (camray) or 6 (plucker)
            expected_cam_ch = 4 if camera_mode == "camray" else 7
            assert camera_embedding.shape[1] == expected_cam_ch, f"Camera embedding should have {expected_cam_ch} channels for mode={camera_mode}, got {camera_embedding.shape[1]}"
            
            # Classifier-Free Guidance training: drop camera embedding (channels 1-6)
            camera_drop = training_cfg.get('camera_drop', 0.0)
            drop_mask = None
            if camera_drop > 0:
                V_num = V_batch
                B_num = B_batch
                if int(x1_cond.shape[0]) != B_num * V_num:
                    raise ValueError(
                        f"Latent batch size mismatch: x1_cond.shape[0]={int(x1_cond.shape[0])} but expected B*V={B_num}*{V_num}={B_num*V_num}. "
                    )
                # Generate mask for each sample in batch
                drop_mask_b = (torch.rand(B_num, 1, 1, 1, device=device) > camera_drop).float()
                # Expand to all views of the same sample
                drop_mask = drop_mask_b.repeat_interleave(V_num, dim=0)  # (B*V, 1, 1, 1)
                # Keep channel 0 (mask), drop 1-6
                camera_embedding[:, 1:] = camera_embedding[:, 1:] * drop_mask
                
            # Setup Model Args
            model_kwargs = dict(
                camera_embedding=camera_embedding, 
                total_view=total_view, 
                cond_num=cond_num_for_model,
                is_concat_mode=True,
                ref_cond=x1_cond,        # Conditioning part [clean_ref | zeros]
                x1_global=x1_all,        # Target part [all views clean]
                freeze_cond=False        
            )

            # GeoFix DIAGNOSTIC (loss_edit / loss_keep). The token-grid mask is
            # handed to the transport so it can split the target loss by mask;
            # it changes no gradients and is dropped before the model call. Only
            # added when a mask actually exists, so stock GLD and the
            # `--geofix-no-mask` ablation arm see the identical model_kwargs
            # they saw before and log nothing rather than a misleading 0.0.
            #
            # GATED ON `geofix_mask_tokens`, NOT ON `geofix_mask` (fixed
            # 2026-08-27). `geofix_mask` is bound only under `mask_in_camera`,
            # while `geofix_mask_tokens` is bound under ANY of the three routes
            # that consume a mask. Gating on the former silently switched the
            # diagnostic off for the `bridge_mask_noise`-only and
            # `blend_train`-only arms -- precisely the arms where the mask enters
            # the TRANSPORT rather than the camera embedding, and therefore
            # precisely the arms whose loss_keep/loss_edit split is the only
            # evidence the mask did anything at all.
            if geofix_mask_tokens is not None:
                model_kwargs['geofix_mask_tokens'] = geofix_mask_tokens
            # Separate key from the diagnostic one above: `geofix_mask_tokens` is
            # observability and changes no gradient, while this one rewrites x_t.
            # Sharing a key would make an arm that only wanted the LOG silently
            # start blending.
            if geofix_loss_keep_weight != 0.0:
                # The gate reads the mask LOSS-SIDE, so it needs the tokens even
                # in an arm with slot 2 off -- which is exactly the arm that
                # isolates "gating the supervision" from "handing the network a
                # mask".
                model_kwargs['geofix_mask_tokens'] = geofix_mask_tokens
                model_kwargs['geofix_loss_keep_weight'] = geofix_loss_keep_weight
            if geofix_blend_train:
                model_kwargs['geofix_blend_tokens'] = geofix_mask_tokens
                # Passed explicitly rather than read off x0 inside the transport,
                # so the blend works under a noise start as well as under a bridge.
                model_kwargs['geofix_blend_toward'] = geofix_latents_art

            
            # Support ProPE (Fail-loud + consistent normalization)
            if use_prope:
                 # Use actual batch resolution for ProPE (supports multi-res)
                 model_kwargs['prope_image_size'] = (image.shape[-2], image.shape[-1])
                 # Compute viewmats (w2c) and Ks
                 with torch.no_grad():
                    # ------------------------------------------------------------------
                    # ProPE safety: translation normalization + outlier detection
                    #
                    # - camera_embedding(plucker) already normalizes translation scale (normalize_t=True),
                    #   but ProPE consumes raw viewmats (w2c). If we feed raw, huge translations can
                    #   explode activations and loss.
                    # - We therefore (1) require an explicit max_translation_norm config, (2) normalize
                    #   c2w translations by a per-sample scale, and (3) normalize extrinsics relative
                    #   to a reference view (match get_camera_embedding: normalize_extrinsic_tgt=-1).
                    # ------------------------------------------------------------------
                    max_t_allowed = multiview_cfg.get("max_translation_norm", None)
                    if max_t_allowed is None:
                        max_t_allowed = dataset_cfg.get("max_translation_norm", None)
                    if max_t_allowed is None:
                        raise ValueError(
                            "Config must explicitly set multiview.max_translation_norm (or dataset.max_translation_norm) "
                            "when use_prope=true, to fail loudly on pose scale outliers."
                        )
                    max_t_allowed = float(max_t_allowed)

                    # Intrinsic: (B, V, 4) -> (B, V, 3, 3) 
                    fx, fy, cx, cy = intrinsic.to(device).unbind(dim=-1)
                    zeros = torch.zeros_like(fx)
                    ones = torch.ones_like(fx)
                    Ks = torch.stack(
                        (
                            torch.stack((fx, zeros, cx), dim=-1),
                            torch.stack((zeros, fy, cy), dim=-1),
                            torch.stack((zeros, zeros, ones), dim=-1),
                        ),
                        dim=-2,
                    )

                    # Extrinsic: (B, V, 4, 4) or 3x4
                    c2w = extrinsic.to(device)
                    if c2w.shape[-2:] == (3, 4):
                        last_row = torch.zeros(c2w.shape[:-2] + (1, 4), device=device)
                        last_row[..., 3] = 1.0
                        c2w = torch.cat([c2w, last_row], dim=-2)
                    
                    # 1) Normalize extrinsics relative to reference view (last view)
                    ref_inv = torch.linalg.inv(c2w[:, -1])  # (B, 4, 4)
                    c2w = ref_inv.unsqueeze(1) @ c2w        # (B, V, 4, 4)
                    
                    # 2) Fail loudly on translation outliers (before any normalization)
                    t = c2w[:, :, :3, 3]
                    t_norm = torch.linalg.vector_norm(t, dim=-1)  # (B, V)
                    max_t = t_norm.max(dim=1).values              # (B,)
                    bad = max_t > max_t_allowed
                    if bad.any():
                        bad_b = torch.where(bad)[0][:10].tolist()
                        raise ValueError(
                            f"ProPE pose outlier detected: max|t| exceeds max_translation_norm={max_t_allowed}. "
                            f"bad batch indices (first up to 10): {bad_b}. "
                            f"max|t| values: {max_t[bad][:10].detach().cpu().tolist()}. "
                            f"cut3r_idx (if available): {batch.get('cut3r_idx', 'N/A')}"
                        )
                    
                    # 3) Translation normalization (match get_camera_embedding normalize_t)
                    # scale = 1 / (max_{v,xyz} |t| + eps)
                    farthest = t.abs().amax(dim=1).amax(dim=1, keepdim=True)  # (B,1)
                    scale = 1.0 / (farthest + 1e-8)
                    c2w = c2w.clone()
                    c2w[:, :, :3, 3] = c2w[:, :, :3, 3] * scale.unsqueeze(1)
                    
                    # Compute w2c = inv(c2w)
                    w2c = torch.linalg.inv(c2w)
                    
                    # CFG camera drop: Replace viewmats with identity for dropped samples
                    if drop_mask is not None:
                        # drop_mask: (B*V, 1, 1, 1), 0 = dropped, 1 = keep
                        # w2c: (B, V, 4, 4)
                        if w2c.shape[0] != B_num or w2c.shape[1] != V_num:
                            raise ValueError(
                                f"w2c shape mismatch: got {tuple(w2c.shape)} but expected (B={B_num}, V={V_num}, 4, 4). "
                                "This usually means total_view does not match the dataloader output."
                            )
                        drop_mask_bv = drop_mask.view(B_num, V_num, 1, 1)  # (B, V, 1, 1)
                        
                        # Create identity viewmats
                        identity_viewmat = torch.eye(4, device=device, dtype=w2c.dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
                        
                        # Replace viewmats with identity where dropped (drop_mask == 0)
                        w2c = drop_mask_bv * w2c + (1 - drop_mask_bv) * identity_viewmat
                        
                        # Replace Ks with identity where dropped (drop_mask == 0)
                        # Ks: (B, V, 3, 3)
                        # drop_mask_bv: (B, V, 1, 1) broadcastable
                        identity_K = torch.eye(3, device=device, dtype=Ks.dtype).unsqueeze(0).unsqueeze(0)
                        Ks = drop_mask_bv * Ks + (1 - drop_mask_bv) * identity_K
                    
                    model_kwargs['viewmats'] = w2c
                    model_kwargs['Ks'] = Ks
            
            with autocast(**autocast_kwargs):
                # Add debugging for timestep sampling
                try:
                    # Feature-to-Feature Flow: Construct x0 from source features + noise
                    x0_init = None
                    # GeoFix LATENT BRIDGE MATCHING. Same level, different image:
                    # x0 = level-1 features of the RENDER, x1 = level-1 features of
                    # the clean photograph. `latents_art` came out of prepare_data
                    # so it is normalised identically to x1 -- re-encoding here
                    # would be the one way to get that wrong.
                    #
                    # THE WHOLE TENSOR IS PASSED, AND THE REFERENCE HALF IS NOT
                    # INERT. This comment used to claim that
                    # `training_multiview_losses` "clamps views [0, cond_num)
                    # back to clean x1 regardless of x0
                    # (`xt = th.where(target_mask, xt, x1_5d)`)", so the
                    # reference slots' x0 was discarded. IT DOES NOT, on this
                    # path. That clamp lives in the `else` branch of
                    # `transport.training_multiview_losses` -- target-only mode --
                    # and `is_concat_mode=True` is hardcoded in `model_kwargs`
                    # below, so the branch that actually runs concatenates
                    # `[x1_cond | xt]` and never touches `xt` on the reference
                    # views. Their x_t therefore interpolates from the ARTIFACT
                    # features like every other view, and `terms['loss']` --
                    # `mean_flat` over all V views, the tensor that is backwarded
                    # -- includes them.
                    #
                    # Corrected 2026-08-27, and the consequence was MEASURED
                    # before the comment was rewritten rather than reasoned about:
                    # reference views contribute 9.7% of the backpropagated loss
                    # in bridge arms against 11.5% in noise arms. A 1.8-point
                    # shift in a term that is under an eighth of the total is not
                    # meaningful gradient dilution, so NO existing result is
                    # invalidated by this and no arm needs re-running.
                    #
                    # DO NOT "FIX" THIS BY ADDING THE CLAMP. Adding it would
                    # change what every arm trained so far was supervised on, for
                    # a 1.8-point effect, and would make new numbers
                    # incomparable with the ones already in RESULTS.md. The
                    # defect was the comment; the code stays.
                    #
                    # `ref_loss` is logged separately (and split out in the
                    # transport), so if this ever does start to matter it is
                    # visible in the existing training curve rather than needing
                    # a new instrument.
                    if geofix_bridge:
                        if geofix_latents_art is None:
                            raise RuntimeError(
                                "geofix.bridge_x0 is on but prepare_data returned no "
                                "artifact latents; the run would fall back to a noise "
                                "x0 and quietly not be a bridge.")
                        if geofix_latents_art.shape != x1_cond.shape:
                            raise ValueError(
                                f"bridge x0 {tuple(geofix_latents_art.shape)} != x1 "
                                f"{tuple(x1_cond.shape)}; the two ends of the flow "
                                "must share a shape and a normalisation.")
                        if geofix_bridge_mask_noise:
                            # sigma_i = tau * M_edit_i -- a PER-TOKEN start noise.
                            # tau is used directly here rather than resampled as
                            # |N(0, tau^2)|: the mask is already the source of
                            # variation across tokens, and multiplying it by a
                            # per-step scalar would blur the very contrast the arm
                            # is testing.
                            if geofix_mask_tokens is None:
                                raise RuntimeError(
                                    "geofix.bridge_mask_noise is on but no mask "
                                    "reached the training step.")
                            m = geofix_mask_tokens.to(
                                device=device, dtype=geofix_latents_art.dtype)
                            if m.ndim != 5 or m.shape[2] != 1:
                                raise ValueError(
                                    f"mask must be (B, V, 1, g, g), got {tuple(m.shape)}.")
                            if geofix_latents_art.ndim != 4 or geofix_latents_art.shape[-1] == 1:
                                raise ValueError(
                                    "bridge_mask_noise needs SPATIAL latents "
                                    f"(BV, C, h, w); got {tuple(geofix_latents_art.shape)}. "
                                    "In the packed CLS format there is no token grid "
                                    "to align the mask to, and broadcasting would "
                                    "silently scale the wrong axis.")
                            m = m.reshape(-1, 1, m.shape[3], m.shape[4])
                            if m.shape[-2:] != geofix_latents_art.shape[-2:]:
                                raise ValueError(
                                    f"mask grid {tuple(m.shape[-2:])} != latent grid "
                                    f"{tuple(geofix_latents_art.shape[-2:])}; the mask is "
                                    "MAX-pooled to the token grid at load time and must "
                                    "match it exactly (hard rule 6).")
                            sigma = geofix_bridge_noise_tau * m   # (BV, 1, h, w)
                        elif geofix_bridge_noise_tau > 0:
                            # sigma ~ |N(0, tau^2)|, then renormalise -- the exact
                            # form GLD's cross-level path uses a few lines below, so
                            # the two starts are perturbed comparably.
                            sigma = torch.abs(
                                torch.randn(1, device=device) * geofix_bridge_noise_tau
                            ).item()
                        else:
                            sigma = 0.0
                        # Written to broadcast over a scalar sigma AND a per-token
                        # one; `sigma ** 2` and the division are elementwise either
                        # way, so the two arms differ only in sigma's shape.
                        x0_init = (
                            geofix_latents_art
                            + torch.randn_like(geofix_latents_art) * sigma
                        ) / (1 + sigma ** 2) ** 0.5
                    if source_level is not None:
                        # Extract source level features WITH SOURCE-LEVEL NORMALIZATION
                        # CRITICAL: Source and target levels have DIFFERENT distributions!
                        # We must normalize source features with source-level stats.
                        with torch.no_grad():
                            # Normalize images for encoder (ImageNet normalization)
                            images_norm = (image.to(device) - rae.encoder_mean[None]) / rae.encoder_std[None]
                            
                            # Save original state for safe restoration
                            original_do_norm = rae.do_normalization
                            original_mean = rae.latent_mean
                            original_var = rae.latent_var
                            
                            try:
                                if _cached_source_stats is not None:
                                    # Use cached stats (loaded once at startup)
                                    rae.latent_mean = _cached_source_stats.get('mean', None)
                                    rae.latent_var = _cached_source_stats.get('var', None)
                                    if rae.latent_mean is not None:
                                        rae.latent_mean = rae.latent_mean.to(device)
                                    if rae.latent_var is not None:
                                        rae.latent_var = rae.latent_var.to(device)
                                    rae.do_normalization = True
                                
                                # Encode with source-level normalization
                                latents_source = rae.encode(images_norm, mode='single', level=source_level)
                                # latents_source: (B*V, C, H, W) - normalized with source-level stats
                            finally:
                                # ALWAYS restore target-level normalization
                                rae.do_normalization = original_do_norm
                                rae.latent_mean = original_mean
                                rae.latent_var = original_var
                        
                        # Sample noise scale: sigma ~ |N(0, tau^2)| (following RAE paper)
                        # This stochastic sigma regularizes training and improves robustness
                        if noise_tau_gt_feat > 0:
                            noise_std = torch.abs(torch.randn(1, device=device) * noise_tau_gt_feat).item()
                        else:
                            noise_std = 0.0
                        noise = torch.randn_like(latents_source) * noise_std
                        x0_init = (latents_source + noise) / math.sqrt(1 + noise_std**2)

                    
                    # NEW: Feature Propagation Mode (Image 1 Architecture)
                    # Extract L1 features for encoder conditioning (NOT for x0 init)
                    if source_condition_level is not None:
                        with torch.no_grad():
                            # Normalize images for encoder (ImageNet normalization)
                            images_norm = (image.to(device) - rae.encoder_mean[None]) / rae.encoder_std[None]
                            
                            # Save original state for safe restoration
                            original_do_norm = rae.do_normalization
                            original_mean = rae.latent_mean
                            original_var = rae.latent_var
                            
                            try:
                                if _cached_source_cond_stats is not None:
                                    # Use cached stats (loaded once at startup)
                                    rae.latent_mean = _cached_source_cond_stats.get('mean', None)
                                    rae.latent_var = _cached_source_cond_stats.get('var', None)
                                    if rae.latent_mean is not None:
                                        rae.latent_mean = rae.latent_mean.to(device)
                                    if rae.latent_var is not None:
                                        rae.latent_var = rae.latent_var.to(device)
                                    rae.do_normalization = True
                                
                                # Encode L1 features with source-condition-level normalization
                                latents_source_cond = rae.encode(images_norm, mode='single', level=source_condition_level)
                                # latents_source_cond: (B*V, C, H, W)
                            finally:
                                # ALWAYS restore target-level normalization
                                rae.do_normalization = original_do_norm
                                rae.latent_mean = original_mean
                                rae.latent_var = original_var
                        
                        # Add noise to L1 condition features for training robustness
                        if noise_tau_gt_feat > 0:
                            cond_noise_std = torch.abs(torch.randn(1, device=device) * noise_tau_gt_feat).item()
                        else:
                            cond_noise_std = 0.0
                        cond_noise = torch.randn_like(latents_source_cond) * cond_noise_std
                        latents_source_cond_noisy = (latents_source_cond + cond_noise) / math.sqrt(1 + cond_noise_std**2)
                        
                        # Add noisy L1 features to model_kwargs for encoder conditioning
                        model_kwargs['source_condition'] = latents_source_cond_noisy
                    
                    # In concat mode: 
                    # - 2nd arg (x1) is the conditioning latent (x1_cond)
                    # - model_kwargs['x1_global'] is the target latent (x1_all)
                    loss_dict = transport.training_multiview_losses(model, x1_cond, total_view, batch_cond_num, model_kwargs, x0=x0_init)
                    loss_tensor = loss_dict["loss"].mean()
                    ref_loss_tensor = loss_dict.get("ref_loss", loss_tensor).mean()
                    tgt_loss_tensor = loss_dict.get("tgt_loss", loss_tensor).mean()
                    # GeoFix diagnostic; absent (None) whenever there is no mask.
                    # Never `or 0.0` -- a zero would read as "no error in the
                    # keep region", which is false.
                    edit_loss_tensor = loss_dict.get("loss_edit", None)
                    keep_loss_tensor = loss_dict.get("loss_keep", None)
                    mask_frac_tensor = loss_dict.get("mask_edit_frac", None)
                except Exception as e:
                    logger.error(f"Exception during forward pass at step {train_steps}: {e}")
                    logger.error(f"Latent stats before forward - Mean: {x1_cond.mean().item():.6f}, Std: {x1_cond.std().item():.6f}")
                    logger.error(f"Latent range before forward - Min: {x1_cond.min().item():.6f}, Max: {x1_cond.max().item():.6f}")
                    raise

            # ============================================================
            # Loss sanity check: skip abnormal loss without backward/logging (DDP-safe)
            #
            # NOTE:
            # - If only one rank skips backward while others backprop, DDP can deadlock on all-reduce.
            # - We therefore all-reduce a bad-flag and, if ANY rank reports abnormal loss, ALL ranks
            #   skip this batch (no loss accumulation, no logging, no backward, no optimizer step).
            # ============================================================
            loss_isfinite = torch.isfinite(loss_tensor).all()
            loss_value = loss_tensor.detach().float().item() if bool(loss_isfinite) else float("nan")
            local_bad = (not bool(loss_isfinite)) or (loss_value > 1000.0)

            if dist.is_available() and dist.is_initialized():
                bad_flag = torch.tensor(int(local_bad), device=device, dtype=torch.int32)
                dist.all_reduce(bad_flag, op=dist.ReduceOp.MAX)
                bad_any = bool(bad_flag.item())
            else:
                bad_any = bool(local_bad)

            if bad_any:
                if rank == 0:
                    video_id = batch.get("video_id", "N/A")
                    frame_indices = batch.get("frame_indices", "N/A")
                    if torch.is_tensor(frame_indices):
                        frame_info = f"tensor(shape={tuple(frame_indices.shape)}, dtype={frame_indices.dtype})"
                    else:
                        frame_info = str(frame_indices)
                    logger.warning(
                        f"[step={train_steps}] Abnormal loss detected on at least one rank; "
                        "skipping this batch (no backward/optimizer/logging). "
                        f"batch video_id={video_id} frame_indices={frame_info}"
                    )
                opt.zero_grad()
                accum_counter = 0
                step_loss_accum = 0.0
                step_ref_loss_accum = 0.0
                step_tgt_loss_accum = 0.0
                geofix_step = _geofix_diag_zero()
                continue

            # ============================================================
            # BACKWARD PASS with Gradient Accumulation (Efficient DDP)
            # ============================================================
            step_loss_accum += loss_tensor.item()
            step_ref_loss_accum += ref_loss_tensor.item()
            step_tgt_loss_accum += tgt_loss_tensor.item()
            # GeoFix mask-split diagnostic. Present only when a mask was passed;
            # `edit` and `keep` always appear together or not at all.
            if edit_loss_tensor is not None and keep_loss_tensor is not None:
                geofix_step["edit"] += edit_loss_tensor.item()
                geofix_step["keep"] += keep_loss_tensor.item()
                geofix_step["frac"] += mask_frac_tensor.item()
                geofix_step["n"] += 1
            
            # Use no_sync() for all but the last accumulation step to avoid unnecessary communication
            if (accum_counter + 1) % grad_accum_steps != 0:
                with model.no_sync():
                    (loss_tensor / grad_accum_steps).backward()
            else:
                (loss_tensor / grad_accum_steps).backward()
            
            # ============================================================
            # PROTECTION: Check for NaN gradients after backward pass
            # ============================================================
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    logger.error(f"NaN/Inf gradient detected in {name} at step {train_steps}")
                    break
            
            if has_nan_grad:
                logger.error(f"NaN gradient detected at step {train_steps}, zeroing gradients and skipping step")
                opt.zero_grad()
                accum_counter = 0
                step_loss_accum = 0.0
                step_ref_loss_accum = 0.0
                step_tgt_loss_accum = 0.0
                geofix_step = _geofix_diag_zero()
                continue
            
            accum_counter += 1

            if accum_counter < grad_accum_steps:
                continue

            if clip_grad > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                # Log if gradient was clipped significantly
                if grad_norm > clip_grad * 10:
                    logger.warning(f"[Step {train_steps}] Large gradient norm before clipping: {grad_norm:.2f}")
                
            opt.step()
            schedl.step()
            update_ema(ema, model.module, decay=ema_decay)
            opt.zero_grad()

            running_loss += step_loss_accum / grad_accum_steps
            running_ref_loss += step_ref_loss_accum / grad_accum_steps
            running_tgt_loss += step_tgt_loss_accum / grad_accum_steps
            # Average over the micro-batches that carried a mask, not over
            # grad_accum_steps, so a mask-free micro-batch cannot pull the
            # diagnostic toward zero.
            if geofix_step["n"] > 0:
                for _k in ("edit", "keep", "frac"):
                    geofix_run[_k] += geofix_step[_k] / geofix_step["n"]
                geofix_run["n"] += 1
            log_steps += 1
            train_steps += 1
            accum_counter = 0
            step_loss_accum = 0.0
            step_ref_loss_accum = 0.0
            step_tgt_loss_accum = 0.0
            geofix_step = _geofix_diag_zero()

            # Smoke-test bound (GeoFix). `epochs` cannot express "10 steps", and a
            # smoke run that has to be killed by hand never proves checkpointing
            # works -- which is half of what the smoke test is for.
            if args.max_steps is not None and train_steps >= int(args.max_steps):
                if rank == 0:
                    logger.info(f"[max_steps] reached {train_steps}; saving and stopping.")
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": schedl.state_dict(),
                        "args": vars(args),
                        "train_steps": train_steps,
                        # F3: the RESOLVED GeoFix conditioning, so a scorer can
                        # check train/test parity from the checkpoint instead of
                        # regexing log.txt. Saved at BOTH sites -- a smoke
                        # checkpoint that lacked it would train the habit of
                        # falling back to the log.
                        "geofix": geofix_settings,
                    }
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(checkpoint, f"{checkpoint_dir}/{train_steps:07d}.pt")
                    logger.info(f"Saved checkpoint to {checkpoint_dir}/{train_steps:07d}.pt")
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                stop_training = True
                break

            # Delay WandB initialization to avoid clutter from failed starts
            # Use >= 5 instead of == 5 to handle resume from checkpoint
            if not wandb_initialized and train_steps >= 0:
                if rank == 0 and args.wandb:
                    entity = os.environ.get("ENTITY", "gld")
                    project = os.environ.get("PROJECT", "gld-mvdiffusion")
                    wandb_utils.initialize(args, entity, experiment_name, project)
                wandb_initialized = True

            if train_steps % log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_ref_loss = torch.tensor(running_ref_loss / log_steps, device=device)
                avg_tgt_loss = torch.tensor(running_tgt_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_ref_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_tgt_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                avg_ref_loss = avg_ref_loss.item() / world_size
                avg_tgt_loss = avg_tgt_loss.item() / world_size

                # ------------------------------------------------------------
                # GeoFix mask-split diagnostic (loss_edit / loss_keep). The
                # all_reduce is UNCONDITIONAL -- ranks can legitimately disagree
                # about whether they saw a mask, and a rank-conditional
                # collective deadlocks DDP. Ranks with no mask contribute zeros
                # AND a zero count, so they do not bias the mean; we divide by
                # the reduced count, not by world_size.
                #
                # Reading it: if the model is learning "preserve where M = 0",
                # `Keep` should fall RELATIVE to `Edit` over training. A flat
                # ratio means the mask is an input the model is ignoring.
                # ------------------------------------------------------------
                geofix_vec = torch.tensor(
                    [geofix_run["edit"], geofix_run["keep"], geofix_run["frac"],
                     float(geofix_run["n"])],
                    device=device, dtype=torch.float64)
                dist.all_reduce(geofix_vec, op=dist.ReduceOp.SUM)
                geofix_n = geofix_vec[3].item()
                geofix_msg = ""
                geofix_wandb = {}
                if geofix_n > 0:
                    avg_edit_loss = geofix_vec[0].item() / geofix_n
                    avg_keep_loss = geofix_vec[1].item() / geofix_n
                    avg_mask_frac = geofix_vec[2].item() / geofix_n
                    geofix_msg = (f", Edit: {avg_edit_loss:.4f}, Keep: {avg_keep_loss:.4f}"
                                  f", MaskFrac: {avg_mask_frac:.3f}")
                    geofix_wandb = {
                        "edit loss": avg_edit_loss,
                        "keep loss": avg_keep_loss,
                        "mask edit frac": avg_mask_frac,
                    }

                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Ref: {avg_ref_loss:.4f}, Tgt: {avg_tgt_loss:.4f}{geofix_msg}, Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        {
                            "train loss": avg_loss, 
                            "ref loss": avg_ref_loss,
                            "tgt loss": avg_tgt_loss,
                            **geofix_wandb,
                            "train steps/sec": steps_per_sec
                        },
                        step=train_steps,
                    )
                running_loss = 0.0
                running_ref_loss = 0.0
                running_tgt_loss = 0.0
                geofix_run = _geofix_diag_zero()
                log_steps = 0
                start_time = time()

            if train_steps % ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": schedl.state_dict(),
                        "train_steps": train_steps,
                        "epoch": epoch,
                        "steps_per_epoch": steps_per_epoch,
                        "config_path": args.config,
                        "training_cfg": training_cfg,
                        # F3: every knob that changes what the model was
                        # conditioned on, resolved AFTER the CLI overrides. The
                        # `training_cfg` above is the `training:` block only, so
                        # before this the mask flags, the bridge and the
                        # dataset's mask_types/pooling/gamma existed nowhere but
                        # the run banner -- and the train/test gate could only
                        # check `bridge_x0`, which is how this defect class
                        # recurred four times.
                        "geofix": geofix_settings,
                        "cli_overrides": {
                            "data_path": args.data_path,
                            "results_dir": args.results_dir,
                            "image_size": args.image_size,
                            "precision": args.precision,
                            "global_seed": global_seed,
                        },
                        # ADDITIVE, 2026-08-27. `config_path` + `training_cfg`
                        # record the run's shape but NOT how the mask reaches the
                        # model: the `geofix:` block is not part of `training:`,
                        # and the `--geofix-*` overrides are not in
                        # `cli_overrides` either. So a checkpoint could be loaded,
                        # sampled with the wrong x0, and produce 776 complete
                        # views and a plausible number -- the failure
                        # `arm_train_test_gate` documents four occurrences of.
                        # Reading the intent off `log.txt` was the workaround; it
                        # breaks the moment a run directory is moved. This is the
                        # record that travels WITH the weights.
                        #
                        # Already-resolved (config + CLI collapsed), so a reader
                        # never has to re-run the resolution to know what trained.
                        # Nothing existing is changed or removed, so every loader
                        # of an older checkpoint is unaffected and every loader of
                        # a newer one that does not know the key ignores it.
                        "geofix": dict(geofix_settings),
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

        # if train_steps % sample_every == 0 or train_steps == 1:
        #     logger.info("Generating EMA samples...")
        #     with torch.no_grad():
        #         with autocast(**autocast_kwargs):
        #             samples = eval_sampler(zs, model_fn, **sample_model_kwargs)[-1]
        #         dist.barrier()

        #         if using_cfg:
        #             samples, _ = samples.chunk(2, dim=0)
        #         samples = rae.decode(samples.to(torch.float32))
        #         out_samples = torch.zeros(
        #             (global_batch_size // grad_accum_steps, 3, args.image_size, args.image_size),
        #             device=device,
        #         )
        #         dist.all_gather_into_tensor(out_samples, samples)
        #         if args.wandb:
        #             wandb_utils.log_image(out_samples, train_steps)
        #     logger.info("Generating EMA samples done.")
            if train_steps % sample_every == 0 or train_steps == 1:
                # All ranks participate in validation for distributed aggregation
                logger.info(f"Running validation at Step {train_steps}...") 
                # Use EMA model for validation instead of training model
                
                # Import shared validation
                from utils.da3_validation import validate_da3_multiview
                
                val_stats = validate_da3_multiview(
                    rae=rae, 
                    model=ema, 
                    transport=transport,
                    sampler=eval_sampler, 
                    loader=val_loader, 
                    device=device, 
                    total_view=val_num_views, 
                    cond_num=val_cond_num, 
                    compute_loss=True,
                    val_num_batches=val_num_batches_per_rank, 
                    use_prope=use_prope,
                    validation_mode=val_mode,
                    output_dir=os.path.join(experiment_dir, f"validation_step_{train_steps}"),
                    pag_scale=pag_scale if pag_scale > 0 else None,
                    pag_layer_idx=pag_layer_idx,
                    run_config=to_dict(cfg),
                    # prope_image_size=args.image_size, # Removed: uses batch resolution internally
                    predict_cls=predict_cls,
                    ref_view_sampling=ref_view_sampling,
                    camera_mode=camera_mode,
                    rank=rank,
                    world_size=world_size,
                    joint_ode=False,  # Disabled when using concat mode
                    is_concat_mode=is_concat_mode,
                    source_level=source_level,  # Feature-to-Feature Flow
                    source_level_stat_path=source_level_stat_path,  # Source-level normalization stats
                    noise_tau_gt_feat=noise_tau_gt_feat / 4.0,  # Fixed noise at tau/4 for validation
                    # NEW: Feature Propagation Mode (Image 1 Architecture)
                    source_condition_level=source_condition_level,
                    source_condition_stat_path=source_condition_stat_path,
                    # The same conditioning the training step uses. Passing these
                    # is not optional once training uses them: validation builds
                    # its own conditioning from scratch, so without them it scores
                    # a GeoFix-trained model under stock GLD inputs, against the
                    # RENDER rather than the clean frame. Finite, plotted, wrong.
                    geofix_cond_artifact=geofix_cond_artifact,
                    geofix_mask_in_camera=geofix_mask_in_camera,
                    # ADDED 2026-08-27. The two above were the only ones passed,
                    # and the trainer supports five more routes. The two measured
                    # consequences: a bridge arm was validated FROM A NOISE START
                    # (val/loss_ref 0.0786 -> 0.5016 against a noise arm's 0.0786
                    # -> 0.0674, a curve that reads as divergence and means "wrong
                    # x0"), and the NVS control -- `clean_target` alone, with
                    # cond_artifact off -- had its PSNR taken against the ARTIFACT
                    # RENDER, because validation gated the gt_clean swap on
                    # `cond_artifact` where the trainer gates it on `clean_target`.
                    #
                    # `tests/test_val_geofix_reflection.py` now asserts by
                    # reflection that this list cannot fall behind again.
                    geofix_clean_target=geofix_clean_target,
                    geofix_bridge_x0=geofix_bridge_x0,
                    geofix_bridge_noise_tau=geofix_bridge_noise_tau,
                    geofix_bridge_mask_noise=geofix_bridge_mask_noise,
                    geofix_blend_train=geofix_blend_train,
                    geofix_blend_hook=geofix_blend_hook,
                )
                # remove images from printed log
                log_copy = {k: v for k, v in val_stats.items() if k != "val/images"}
                if rank == 0:
                    logger.info(f"[Validation @ Step {train_steps}] " + ", ".join(f"{k}: {v:.4f}" for k, v in log_copy.items()))
                    if args.wandb:
                        wandb_utils.log(log_copy, step=train_steps)
                        if val_stats["val/images"] is not None:
                            # Log list of images
                            images_to_log = [wandb.Image(img) for img in val_stats["val/images"]]
                            wandb_utils.log({"val/reconstructions": images_to_log}, step=train_steps)
            
            # Ensure all ranks are synchronized after validation before moving to next training step
            if dist.is_initialized():
                dist.barrier()

        if accum_counter != 0:
            # Kept as a genuine invariant assert. The islice above makes the epoch
            # end on an optimizer-step boundary, so reaching this now means
            # something ELSE ended the epoch early -- an exception swallowed inside
            # the loop, or a `break` added without draining the accumulation. The
            # ragged-tail case that used to land here is handled at the source.
            raise RuntimeError(
                f"Gradient accumulation counter is {accum_counter}, expected 0 at "
                f"epoch end (grad_accum_steps={grad_accum_steps}, "
                f"micro_batches_per_epoch={micro_batches_per_epoch}). The epoch "
                "ended mid-accumulation, which the islice bound should prevent; "
                "look for an early exit inside the training loop, not for an "
                "indivisible loader length.")

    model.eval()
    logger.info("Done!")
    cleanup()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DA3 MVDiffusion Training")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the training dataset root.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to store training outputs.")
    # NOTE: image-size is now read from config (stage_1.params.encoder_input_size)
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32", help="Compute precision for training.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    #
    parser.add_argument("--level", type=int, default=-1, help="DA3 feature level to train (-1, -2, -3, -4)")


    parser.add_argument("--pretrained", type=str, default=None, help="Resume from PRETRAINED DDT, only loading for model")
    parser.add_argument("--overfit",  action="store_true", help="overfit to a single sample")
    # Check for duplicates or cleanup if needed.
    # We already have run_name logic in config possibly? But keep it here.
    parser.add_argument("--run_name", type=str, default="da3_mvdiff", help="name of the run")
    parser.add_argument("--global-seed", type=int, default=0, help="Global seed")
    parser.add_argument("--vae-type", type=str, default="RAE", help="VAE type: RAE or VAE")
    parser.add_argument("--ckpt", type=str, default=None, help="Resume from checkpoint (model+ema+opt)")
    parser.add_argument("--git-hash", type=str, default=None, help="Git hash for current run")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop after N optimizer steps, saving a checkpoint first. "
                             "For smoke tests (GeoFix: 'smoke test before every real run').")
    parser.add_argument("--global-batch-size", type=int, default=None,
                        help="Override training.global_batch_size. For running one "
                             "config on GPUs with different memory; hold it FIXED "
                             "across ablation arms.")
    parser.add_argument("--grad-accum-steps", type=int, default=None,
                        help="Override training.grad_accum_steps. Changes only how "
                             "the effective batch is split, not its size.")
    parser.add_argument("--geofix-no-mask", action="store_true",
                        help="Force geofix.mask_in_camera off, leaving cond_artifact "
                             "as configured. This is the GeoFix ablation arm -- "
                             "refinement WITHOUT the mask, which is the honest "
                             "baseline for 'does the mask add anything'. Uses the "
                             "same config file as the mask-on arm so nothing else "
                             "can differ between them.")

    parser.add_argument("--geofix-no-cond-artifact", action="store_true",
                        help="Force geofix.cond_artifact off, leaving mask_in_camera "
                             "as configured. This is the REDUNDANCY control: it "
                             "measures what the mask is worth when the artifact "
                             "render is NOT supplied. Pairs with --geofix-no-mask "
                             "to give all four cells of the two-slot ablation off "
                             "ONE config file.")

    parser.add_argument("--geofix-manifest", default=None,
                        help="Override dataset.manifest for the TRAINING loader "
                             "only, leaving val_manifest and every other config "
                             "value untouched. Exists for the scene-count scaling "
                             "curve, where the manifest is the only thing that may "
                             "differ between arms.")
    parser.add_argument("--geofix-clean-target", action="store_true",
                        help="Supervise the flow toward the CLEAN photograph even "
                             "with both conditioning slots off. Without it, an arm "
                             "run with --geofix-no-cond-artifact is supervised "
                             "toward the ARTIFACT RENDER (`gt_inp`), which is how "
                             "the two `maskonly_*` runs of 2026-08-20 came to be "
                             "trained to reproduce 3DGS artifacts. Combine with "
                             "--geofix-no-cond-artifact --geofix-no-mask for the NVS "
                             "control: plain GLD finetuned on our DL3DV scenes, no "
                             "render conditioning, no mask.")

    parser.add_argument("--geofix-bridge-x0", default=None, choices=("artifact",),
                        help="LATENT BRIDGE MATCHING. Start the flow at the level-1 "
                             "features of the 3DGS render instead of at noise, so "
                             "the model learns F_artifact -> F_clean. Overrides "
                             "geofix.bridge_x0.")
    parser.add_argument("--geofix-blend-train", action="store_true",
                        help="LATENT BLENDING AT TRAINING TIME: composite x_t "
                             "toward the artifact features by (1 - M_edit), which "
                             "under a bridge is exactly the per-token time "
                             "t_i = 1 - M_i (1 - t). Requires --geofix-bridge-x0 "
                             "artifact. The loss target is NOT modulated: clean is "
                             "the truth everywhere, including where the mask says "
                             "preserve.")
    parser.add_argument("--geofix-no-blend-train", action="store_true",
                        help="Force geofix.blend_train off so the control runs off "
                             "the SAME config file as the blended arm.")
    parser.add_argument("--geofix-bridge-mask-noise", action="store_true",
                        help="THE MASK-MODULATED BRIDGE: scale the bridge start "
                             "noise per token by M_edit, so the flow starts "
                             "deterministically where the mask says 'preserve' and "
                             "noisily where it says 'repair'. Requires "
                             "--geofix-bridge-x0 artifact. Overrides "
                             "geofix.bridge_mask_noise.")
    parser.add_argument("--geofix-no-bridge-mask-noise", action="store_true",
                        help="Force geofix.bridge_mask_noise off so the unmodulated "
                             "bridge control runs off the SAME config file as the "
                             "mask-modulated arm.")
    parser.add_argument("--geofix-loss-keep-weight", type=float, default=None,
                        help="PHASE 3a, the mask-gated loss: weight the per-token "
                             "flow loss by 1 + lambda*(1-M_edit) on target views, "
                             "renormalised to mean 1 so lambda is not a learning "
                             "rate in disguise. Upweights the PRESERVE region. "
                             "0 = inert. Overrides geofix.loss_keep_weight.")
    parser.add_argument("--geofix-no-bridge-x0", action="store_true",
                        help="Force geofix.bridge_x0 off, so the noise-x0 control "
                             "runs off the SAME config file as the bridge arm.")

    args = parser.parse_args()
    # Log level info
    print(f"Training DA3 MVDiffusion at level {args.level}")
    main(args)
