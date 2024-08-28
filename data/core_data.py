from torch.utils.data import Dataset
from typing import List, Tuple
import torch
import json
import glob
import random
import os


class CoreDataset(Dataset):
    def __init__(self, metadata_file: str, root_folder: str):
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)
        self.root_folder = root_folder

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        metadata = self.metadata[index]
        caption = metadata["caption"]
        image_path = metadata["image_path"]
        image_path = os.path.join(self.root_folder, image_path)
        image = torch.load(image_path)
        return image, caption


class CoreCachedDataset(Dataset):
    def __init__(self, cached_folder: str, max_len: int = 512):
        self.cached_files = glob.glob(f"{cached_folder}/*.pt")
        self.max_len = max_len
        self.max_step = 1000

    def __len__(self):
        return len(self.cached_files)

    def add_noise(self, latent: torch.Tensor):
        sigma = torch.randn_like(latent)
        noised_latent = (1 - sigma) * latent + sigma * torch.randn_like(latent)
        return noised_latent, sigma

    def __getitem__(self, index):
        cached_file = self.cached_files[index]
        feeds = torch.load(cached_file)
        latent = feeds["latents"]
        noised_latent, sigma = self.add_noise(latent)
        feeds["latents"] = noised_latent
        step = int(sigma * self.max_step)
        target = latent

        metadata = {
            "step": step,
            "sigma": sigma,
        }

        return feeds, target, metadata
