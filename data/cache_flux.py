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
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    @staticmethod
    def time_shift(mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    @staticmethod
    def get_args(parser):
        parser.add_argument(
            "--cache_flux.cache_dir",
            default="data/cache",
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


if __name__ == "__main__":
    from data.core_data import CoreCachedDataset
    import diffusers
    import argparse
    import math

    parser = argparse.ArgumentParser(
        description="Script to run training with various options."
    )

    parser.add_argument("--cache_dir", default="debug/cache_tshirt", type=str)
    parser.add_argument("--dataset_root", default="dataset/tshirt/images", type=str)
    parser.add_argument(
        "--metadata_file", default="dataset/tshirt/metadata.json", type=str
    )
    parser.add_argument("--save_debug_image", default="debug/image.jpg", type=str)
    parser.add_argument(
        "--save_debug_image_reconstructed",
        default="debug/image_reconstructed.jpg",
        type=str,
    )
    parser.add_argument(
        "--save_debug_image_noised", default="debug/image_noised.jpg", type=str
    )
    args = parser.parse_args()
