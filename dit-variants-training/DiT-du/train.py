# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT_models, DOWNSAMPLERS, UPSAMPLERS
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

from lsc_adapter import add_lsc

from albumentations.pytorch import ToTensorV2
from albumentations import CenterCrop, HorizontalFlip, Normalize, Compose


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        if args.model.startswith('DiT-UNet'):
            model_string_name = f'{args.model.replace("/", "-")}-{args.downsample_type}-{args.upsample_type}-{args.lsc_type.replace(" ", "-").replace("/", "_")}'
        else:
            model_string_name = f'{args.model.replace("/", "-")}-{args.lsc_type.replace(" ", "-").replace("/", "_")}'
        # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        downsample_type=args.downsample_type,
        upsample_type=args.upsample_type,
    )
    # MODIFICATION
    # add lsc
    if args.lsc_type != 'w/o LSC':
        if args.lsc_type == 'w/ LSC dense':
            skip_table = {
              'down': {
                0: 27,
                1: 26,
                2: 25,
                3: 24,
                4: 23,
                5: 22,
                6: 21,
                7: 20,
                8: 19,
                9: 18,
                10: 17,
                11: 16,
                12: 15,
              },
              'up': {
                27: 0,
                26: 1,
                25: 2,
                24: 3,
                23: 4,
                22: 5,
                21: 6,
                20: 7,
                19: 8,
                18: 9,
                17: 10,
                16: 11,
                15: 12,
              }
            }
            if args.lsc_addon == '':
                lsc_type = 'basic'
            elif args.lsc_addon == 'denoising':
                lsc_type = 'denoising'
            elif args.lsc_addon == 'shuffle':
                lsc_type = 'shuffle'
            elif args.lsc_addon == 'highpass':
                lsc_type = 'highpass'
            elif args.lsc_addon == 'external-semantic-noise':
                lsc_type = 'external-semantic-noise'
                model.add_external_embedder()
            elif args.lsc_addon == 'external_highpass':
                lsc_type = 'external_highpass'
                model.add_external_embedder()
            elif args.lsc_addon == 'external-random-noise':
                lsc_type = 'external-random-noise'
                model.add_external_embedder()
            elif args.lsc_addon == 'external-x':
                lsc_type = 'external-x'
                model.add_external_embedder()
            elif args.lsc_addon == 'external-0.25noise-0.75noise2':
                lsc_type = 'external-0.25noise-0.75noise2'
                model.add_external_embedder()
            elif args.lsc_addon == 'external-0.75noise-0.25noise2':
                lsc_type = 'external-0.75noise-0.25noise2'
                model.add_external_embedder()
            elif args.lsc_addon == 'subtraction-noise':
                lsc_type = 'subtraction-noise'
            elif args.lsc_addon == 'subtraction_highpass-x':
                lsc_type = 'subtraction_highpass-x'
            elif args.lsc_addon == 'external-x-and-noise':
                lsc_type = 'external-x-and-noise'
                model.add_external_embedder()
            elif args.lsc_addon == 'external-x-and-noise2':
                lsc_type = 'external-x-and-noise2'
                model.add_external_embedder()
            elif args.lsc_addon == 'external-x2-and-noise':
                lsc_type = 'external-x2-and-noise'
                model.add_external_embedder()
            elif args.lsc_addon == 'external-x2-and-noise2':
                lsc_type = 'external-x2-and-noise2'
                model.add_external_embedder()
            elif args.lsc_addon == 'dual':
                lsc_type = 'dual'
            elif args.lsc_addon == 'dual_false':
                lsc_type = 'dual_false'
            add_lsc(model, 1152, layer_count=len(model.blocks), lsc_type=lsc_type, skip_table=skip_table)
        elif args.lsc_type == 'w/ LSC sparse':
            add_lsc(model, 1152, layer_count=len(model.blocks), lsc_type='basic-advanced')
        elif args.lsc_type == 'upper LSC':
            skip_table = {
                'down': {0: 27, 2: 24},
                'up': {27: 0, 24: 2},
            }
            add_lsc(model, 1152, layer_count=len(model.blocks), skip_table=skip_table)
        elif args.lsc_type == 'lower LSC':
            skip_table = {
                'down': {6: 18, 8: 15},
                'up': {18: 6, 15: 8},
            }
            add_lsc(model, 1152, layer_count=len(model.blocks), skip_table=skip_table)
        elif args.lsc_type == 'skewed LSC':
            skip_table = {
                'down': {
                    0: 27,
                    4: 26,
                    8: 25,
                    12: 24,
                    16: 23,
                },
                'up': {
                    27: 0,
                    26: 4,
                    25: 8,
                    24: 12,
                    23: 16,
                },
            }
            add_lsc(model, 1152, layer_count=len(model.blocks), skip_table=skip_table)
        elif args.lsc_type == 'upper LSC dense':
            skip_table = {
              'down': {
                0: 27,
                1: 26,
                2: 25,
                3: 24,
                4: 23,
                5: 22,
              },
              'up': {
                27: 0,
                26: 1,
                25: 2,
                24: 3,
                23: 4,
                22: 5,
              }
            }
            add_lsc(model, 1152, layer_count=len(model.blocks), skip_table=skip_table)
        elif args.lsc_type == 'lower LSC dense':
            skip_table = {
              'down': {
                7: 20,
                8: 19,
                9: 18,
                10: 17,
                11: 16,
                12: 15,
              },
              'up': {
                20: 7,
                19: 8,
                18: 9,
                17: 10,
                16: 11,
                15: 12,
              }
            }
            add_lsc(model, 1152, layer_count=len(model.blocks), skip_table=skip_table)
        elif args.lsc_type == 'skewed LSC dense':
            skip_table = {
                'down': {
                    6: 27,
                    7: 26,
                    8: 25,
                    9: 24,
                    10: 23,
                    11: 22,
                    12: 21,
                    13: 20,
                    14: 19,
                    15: 18,
                },
                'up': {
                    27: 6,
                    26: 7,
                    25: 8,
                    24: 9,
                    23: 10,
                    22: 11,
                    21: 12,
                    20: 13,
                    19: 14,
                    18: 15,
                }
            }
            add_lsc(model, 1152, layer_count=len(model.blocks), skip_table=skip_table)
        elif args.lsc_type == 'alignment':
            skip_table = {
                'down': {
                    0: 1,
                    1: 2,
                    2: 3,
                    3: 4,
                    4: 5,
                    5: 6,
                    6: 7,
                    7: 8,
                    8: 9,
                    9: 10,
                    10: 11,
                    11: 12,
                    12: 13,
                },
                'up': {
                    1: 0,
                    2: 1,
                    3: 2,
                    4: 3,
                    5: 4,
                    6: 5,
                    7: 6,
                    8: 7,
                    9: 8,
                    10: 9,
                    11: 10,
                    12: 11,
                    13: 12,
                }
            }
            add_lsc(model, 1152, layer_count=len(model.blocks), skip_table=skip_table)
        else:
            raise NotImplementedError
        print(model.lsc.skip_table)
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    # transform = Compose([
    #     CenterCrop(height=args.image_size, width=args.image_size),
    #     HorizontalFlip(p=0.5),
    #     Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #     ToTensorV2()
    # ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=8,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0

    scaler = torch.amp.GradScaler("cuda")
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        # timer = time()
        # timer_big = time()
        for x, y in loader:
            # print('big:', train_steps, time() - timer_big)
            # timer_big = time()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # print('fetch data:', time() - timer)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                # timer = time()
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                # print('vae:', time() - timer)
                # timer = time()
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                model_kwargs = dict(y=y)
                # print('sample t:', time() - timer)
                # timer = time()
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                # print('forward:', time() - timer)
                # timer = time()
                loss = loss_dict["loss"].mean()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            # print('backward:', time() - timer)
            # timer = time()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--lsc_type", type=str, default='w/o LSC')
    parser.add_argument('--lsc_addon', type=str, default='')
    parser.add_argument("--downsample-type", type=str, choices=list(DOWNSAMPLERS.keys()), default='conv',
                        help="Downsampling module for DiT-UNet models.")
    parser.add_argument("--upsample-type", type=str, choices=list(UPSAMPLERS.keys()), default='conv',
                        help="Upsampling module for DiT-UNet models.")
    args = parser.parse_args()
    main(args)
    print(args.lsc_type)
