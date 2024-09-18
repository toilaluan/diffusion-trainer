from torch.utils.data import Dataset
from typing import List, Tuple
import torch
import json
import glob
import random
import os
from PIL import Image
import math
import diffusers


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class CoreDataset(Dataset):
    def __init__(self, config):
        with open(config.metadata_file, "r") as f:
            self.metadata = json.load(f)
        self.root_folder = config.root_folder
        self.bucket_config = self._init_bucket()

    def _init_bucket(
        self,
        base_size: int = 1024,
        min_size: int = 512,
        max_size: int = 1256,
        divisible: int = 32,
    ):
        widths = [
            base_size - divisible * i
            for i in range(1, (base_size - min_size) // divisible)
        ] + [
            base_size + divisible * i
            for i in range(1, (max_size - base_size) // divisible)
        ]
        heights = [
            base_size - divisible * i
            for i in range(1, (base_size - min_size) // divisible)
        ] + [
            base_size + divisible * i
            for i in range(1, (max_size - base_size) // divisible)
        ]
        sizes = {}
        base_res = (base_size * base_size) ** 0.5
        for width in widths:
            for height in heights:
                res = (width * height) ** 0.5
                if not (base_res * 0.8 < res < base_res * 1.25):
                    print(base_res, res)
                    continue
                ratio = width / height
                sizes[ratio] = (width, height)
        print(f"Initialized {len(sizes)} bucket sizes")
        for k, v in sizes.items():
            print(f"Bucket ratio: {k} size: {v}")
        return sizes

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        metadata = self.metadata[index]
        caption = metadata["caption"]
        image_path = metadata["image_path"]
        image_path = os.path.join(self.root_folder, image_path)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        ratio = width / height
        # select nearest bucket size
        bucket_ratio = min(self.bucket_config.keys(), key=lambda x: abs(x - ratio))
        bucket_size = self.bucket_config[bucket_ratio]
        image = image.resize(bucket_size)
        return image, caption

    @staticmethod
    def get_args(parser):
        parser.add_argument(
            "--core_dataset.dataset_root", type=str, default="dataset/tshirt/images"
        )
        parser.add_argument(
            "--core_dataset.metadata_file",
            type=str,
            default="dataset/tshirt/metadata.json",
        )


class CoreCachedDataset(Dataset):
    def __init__(self, config):
        self.cached_files = glob.glob(f"{config.cached_folder}/*.pt")
        self.max_len = config.max_len
        self.max_step = 1000
        self.pipeline = diffusers.FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=None,
            text_encoder=None,
            text_encoder_2=None,
        )

    def __len__(self):
        return len(self.cached_files)

    def add_noise(self, latent: torch.Tensor):
        timestep = random.randint(1, 250)
        sigma = timestep / self.max_step
        noise = torch.randn_like(latent).to(latent.dtype)
        noised_latent = (1 - sigma) * latent + sigma * noise
        return noised_latent, sigma, noise

    def __getitem__(self, index):
        cached_file = self.cached_files[index]
        feeds = torch.load(cached_file)
        latent = feeds["vae_latents"]
        dtype = latent.dtype
        noised_latent, sigma, noise = self.add_noise(latent, dtype)
        packed_latent = self.pipeline._pack_latents(
            noised_latent,
            batch_size=noised_latent.shape[0],
            num_channels_latents=noised_latent.shape[1],
            height=noised_latent.shape[2],
            width=noised_latent.shape[3],
        )
        feeds["timestep"] = torch.Tensor([sigma])
        feeds["latents"] = packed_latent
        step = int(sigma * self.max_step)
        target = noise - latent
        metadata = {
            "step": step,
            "sigma": sigma,
        }

        return feeds, target, metadata

    def get_noised_latent(self, idx, sigma):
        cached_file = self.cached_files[idx]
        feeds = torch.load(cached_file)
        latent = feeds["latents"]
        mu = calculate_shift(latent.shape[1])
        shift = math.exp(mu)
        print("mu", mu)
        print("sigma", sigma)
        sigma = (sigma * shift) / (1 + (shift - 1) * sigma)
        print("shifted sigma", sigma)
        dtype = latent.dtype
        noise = torch.randn_like(latent).to(dtype)
        noised_latent = (1 - sigma) * latent + sigma * noise
        return noised_latent

    @staticmethod
    def get_args(parser):
        parser.add_argument(
            "--core_cached_dataset.cached_folder",
            type=str,
            default="data/cache",
            help="Cached folder",
        )
        parser.add_argument(
            "--core_cached_dataset.max_len",
            type=int,
            default=512,
            help="Max sequence length",
        )


def collate_fn(batch):
    feeds, targets, metadata = zip(*batch)
    feeds = {k: torch.cat([f[k] for f in feeds], dim=0) for k in feeds[0]}
    targets = torch.cat(targets, dim=0)
    return feeds, targets, metadata
