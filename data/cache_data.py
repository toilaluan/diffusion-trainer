import torch
import transformers
import diffusers
from data.core_data import CoreDataset
from PIL import Image
import os


class CacheFlux:
    def __init__(
        self,
        pretrained_path: str = "black-forest-labs/FLUX.1-dev",
        save_dir: str = "data/cache",
    ):
        self.save_dir = save_dir
        self.pretrained_path = pretrained_path
        self.pipeline = diffusers.FluxPipeline.from_pretrained(
            pretrained_path, transformer=None
        )
        self.transformer_config = transformers.PretrainedConfig.from_pretrained(
            pretrained_path, subfolder="transformer"
        )
        self.device = "cuda"
        self.pipeline.to(self.device)
        os.makedirs(save_dir, exist_ok=True)

    def __call__(self, image: Image.Image, prompt: str, filename: str):
        height, width = image.size
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=256,
        )

        num_channels_latents = self.transformer_config.in_channels // 4
        _, latent_image_ids = self.prepare_latents(
            1,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            latents,
        )
        latents = self.pipeline.image_processor.encode_image(
            image,
            height,
            width,
            device=self.device,
            latents=latents,
        )
        latents = (
            latents - self.pipeline.vae.config.shift_factor
        ) * self.pipeline.vae.config.scaling_factor

        feeds = {
            "latents": latents,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
            "latent_image_ids": latent_image_ids,
        }

        torch.save(feeds, os.path.join(self.save_dir, f"{filename}.pt"))


if __name__ == "__main__":
    cache_flux = CacheFlux()
    image = Image.open("data/image.webp")
    prompt = "A beautiful landscape painting"
    cache_flux(image, prompt, "image")
