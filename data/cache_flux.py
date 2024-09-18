import torch
import torch.amp
import transformers
import diffusers
from data.core_data import CoreDataset
from PIL import Image
import os
from diffusers.image_processor import VaeImageProcessor
from utilities.dotable_config import Config
import argparse
import math


class CacheFlux:
    def __init__(
        self,
        config,
    ):
        """
        Args:
            config (Namespace): Configuration object containing the cache directory, pretrained model path, and torch dtype.
        """
        self.save_dir = config.cache_dir
        self.guidance_scale = 3.5
        self.pretrained_path = config.pretrained_path
        self.pipeline = diffusers.FluxPipeline.from_pretrained(
            self.pretrained_path,
            transformer=None,
            torch_dtype=eval(str(config.torch_dtype)),
        )
        self.transformer_config = transformers.PretrainedConfig.from_pretrained(
            self.pretrained_path,
            subfolder="transformer",
        )
        self.vae_scale_factor = 2 ** (len(self.pipeline.vae.config.block_out_channels))
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.device = "cuda"
        self.pipeline.to(self.device)
        self.torch_dtype = eval(str(config.torch_dtype))
        os.makedirs(self.save_dir, exist_ok=True)

    @torch.no_grad()
    def __call__(self, image: Image.Image, prompt: str, filename: str):
        """
        Args:
            image (PIL.Image.Image): Input image to be processed.
            prompt (str): The text prompt for generating image embeddings.
            filename (str): The filename to save the resulting cache data.
        """
        width, height = image.size
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=self.device,
            num_images_per_prompt=1,
        )

        num_channels_latents = self.transformer_config.in_channels // 4
        noise_latents, latent_image_ids = self.pipeline.prepare_latents(
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=self.device,
            generator=None,
            latents=None,
        )
        latents = self.image_processor.preprocess(
            image,
        )
        latents = latents.to(self.device, self.torch_dtype)
        latents = self.pipeline.vae.encode(latents).latent_dist.sample()
        latents = (
            latents - self.pipeline.vae.config.shift_factor
        ) * self.pipeline.vae.config.scaling_factor

        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)
        packed_latents = self.pipeline._pack_latents(
            latents,
            batch_size=latents.shape[0],
            num_channels_latents=latents.shape[1],
            height=height,
            width=width,
        )
        assert packed_latents.shape == noise_latents.shape
        guidance = (
            torch.tensor([self.guidance_scale]).to(self.torch_dtype).to(self.device)
        )

        feeds = {
            "latents": packed_latents.to(self.torch_dtype).cpu(),
            "pooled_prompt_embeds": pooled_prompt_embeds.to(self.torch_dtype).cpu(),
            "prompt_embeds": prompt_embeds.to(self.torch_dtype).cpu(),
            "text_ids": text_ids.to(self.torch_dtype).cpu(),
            "latent_image_ids": latent_image_ids.to(self.torch_dtype).cpu(),
            "guidance": guidance.to(self.torch_dtype).cpu(),
            "vae_latents": latents.to(self.torch_dtype).cpu(),
        }

        torch.save(feeds, os.path.join(self.save_dir, f"{filename}.pt"))

    @torch.no_grad()
    def decode_from_latent(self, latents: torch.Tensor, width, height):
        """
        Args:
            latents (torch.Tensor): Latent representations of the image.
            width (int): Width of the original image.
            height (int): Height of the original image.

        Returns:
            PIL.Image.Image: Decoded image from latent space.
        """
        latents = self.pipeline._unpack_latents(
            latents, height, width, self.vae_scale_factor
        )
        latents = latents.to(self.device)
        latents = (
            latents / self.pipeline.vae.config.scaling_factor
        ) + self.pipeline.vae.config.shift_factor

        image = self.pipeline.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="pil")
        return image[0]

    @staticmethod
    def calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.16,
    ):
        """
        Args:
            image_seq_len (int): Length of the image sequence.
            base_seq_len (int): Base sequence length for shift calculation. Default: 256.
            max_seq_len (int): Maximum sequence length for shift calculation. Default: 4096.
            base_shift (float): Base shift value. Default: 0.5.
            max_shift (float): Maximum shift value. Default: 1.16.

        Returns:
            float: Calculated shift value.
        """
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    @staticmethod
    def time_shift(mu: float, sigma: float, t: torch.Tensor):
        """
        Args:
            mu (float): Shift parameter for time calculation.
            sigma (float): Scaling factor for time shift.
            t (torch.Tensor): Input tensor for time calculation.

        Returns:
            float: Result of the time shift operation.
        """
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    @staticmethod
    def get_args(parser):
        """
        Defines the arguments for configuring the CacheFlux pipeline.

        Args:
            parser (ArgumentParser): Argument parser object used to define command-line arguments.

        Arguments:
            --cache_flux.cache_dir (str): Cache directory to store the processed files. Default: "cache/tshirt".
            --cache_flux.pretrained_path (str): Path to the pretrained model. Default: "black-forest-labs/FLUX.1-dev".
            --cache_flux.torch_dtype (str): Data type for torch tensors (e.g., "torch.float32", "torch.float16"). Default: "torch.float32".
            --cache_flux.guidance_scale (float): Guidance scale value for image generation. Default: 3.5.
        """
        parser.add_argument(
            "--cache_flux.cache_dir",
            default="cache/tshirt",
            type=str,
            help="Cache directory",
        )
        parser.add_argument(
            "--cache_flux.pretrained_path",
            default="black-forest-labs/FLUX.1-dev",
            type=str,
            help="Pretrained path",
        )
        parser.add_argument(
            "--cache_flux.torch_dtype",
            default="torch.float32",
            type=str,
            help="Torch dtype",
        )
        parser.add_argument(
            "--cache_flux.guidance_scale",
            default=3.5,
            type=float,
            help="Guidance scale",
        )
