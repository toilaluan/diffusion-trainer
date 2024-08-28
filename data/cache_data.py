import torch
import transformers
import diffusers
from data.core_data import CoreDataset
from PIL import Image
import os
from diffusers.image_processor import VaeImageProcessor


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
        vae_scale_factor = 2 ** (len(self.pipeline.vae.config.block_out_channels))
        self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
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
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=256,
        )

        num_channels_latents = self.transformer_config.in_channels // 4
        _, latent_image_ids = self.pipeline.prepare_latents(
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
        latents = latents.to(self.device)
        latents = self.pipeline.vae.encode(latents).latent_dist.sample()
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

    def decode_from_latent(self, latents: torch.Tensor):
        latents = latents.to(self.device)
        latents = (
            latents / self.pipeline.vae.config.scaling_factor
        ) + self.pipeline.vae.config.shift_factor

        image = self.pipeline.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="pil")
        return image[0]


if __name__ == "__main__":
    with torch.no_grad():
        cache_flux = CacheFlux()
        image = Image.open("data/image.webp")
        prompt = "A beautiful landscape painting"
        cache_flux(image, prompt, "image")
        feeds = torch.load("data/cache/image.pt")
        image = cache_flux.decode_from_latent(feeds["latents"])
        image.save("data/image_reconstructed.png")

        dataset = CoreDataset(
            root_folder="dataset/itay_test/images",
            metadata_file="dataset/itay_test/metadata.json",
        )
        image, caption = dataset[0]
        cache_flux(image, caption, "image")
        feeds = torch.load("data/cache/image.pt")
        image = cache_flux.decode_from_latent(feeds["latents"])
        image.save("data/image_reconstructed.png")
