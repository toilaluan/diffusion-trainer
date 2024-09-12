import torch
import torch.amp
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
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.save_dir = save_dir
        self.guidance_scale = 3.5
        self.pretrained_path = pretrained_path
        self.pipeline = diffusers.FluxPipeline.from_pretrained(
            pretrained_path, transformer=None, torch_dtype=torch_dtype
        )
        self.transformer_config = transformers.PretrainedConfig.from_pretrained(
            pretrained_path,
            subfolder="transformer",
        )
        self.vae_scale_factor = 2 ** (len(self.pipeline.vae.config.block_out_channels))
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.device = "cuda"
        self.pipeline.to(self.device)
        self.torch_dtype = torch_dtype
        os.makedirs(save_dir, exist_ok=True)

    @torch.no_grad()
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
        print(noise_latents.shape)
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
        print(height, width)
        packed_latents = self.pipeline._pack_latents(
            latents,
            batch_size=latents.shape[0],
            num_channels_latents=latents.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
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
    def decode_from_latent(self, latents: torch.Tensor, height, width):
        height = int(height * self.vae_scale_factor / 2)
        width = int(width * self.vae_scale_factor / 2)
        print(latents.shape)
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


if __name__ == "__main__":
    from data.core_data import CoreCachedDataset
    import diffusers
    import argparse

    parser = argparse.ArgumentParser(
        description="Script to run training with various options."
    )

    parser.add_argument("--cache_dir", default="debug/cache_tshirt", type=str)
    parser.add_argument("--dataset_root", default="dataset/tshirt/images", type=str)
    parser.add_argument(
        "--metadata_file", default="dataset/tshirt/metadata.json", type=str
    )
    parser.add_argument(
        "--sample_cached_file", default="debug/cache_tshirt/image_0.pt", type=str
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

    with torch.no_grad():
        print("Debugging cache flux")
        cache_flux = CacheFlux(save_dir=args.cache_dir)
        dataset = CoreDataset(
            root_folder=args.dataset_root,
            metadata_file=args.metadata_file,
        )
        image, caption = dataset[0]

        image.save(args.save_debug_image)
        cache_flux(
            image,
            caption,
            filename=args.sample_cached_file.split("/")[-1].split(".")[0],
        )
        feeds = torch.load(args.sample_cached_file)
        vae_output = feeds["vae_latents"]
        print(vae_output.shape)

        print("Debugging cache flux decode")
        image = cache_flux.decode_from_latent(
            feeds["latents"], vae_output.shape[2], vae_output.shape[3]
        )
        image.save(args.save_debug_image_reconstructed)

        cached_dataset = CoreCachedDataset(cached_folder=args.cache_dir)
        noised_latent = cached_dataset.get_noised_latent(0, 0.5)
        image = cache_flux.decode_from_latent(
            noised_latent, vae_output.shape[2], vae_output.shape[3]
        )
        image.save(args.save_debug_image_noised)

        print("Debugging transformer denoise")
        transformer = diffusers.FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        transformer.to("cuda")

        num_inferece_steps = 30
        dt = 1 / num_inferece_steps
        denoise_images = []
        for i in range(num_inferece_steps):
            print("Denoising step", i)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                noise_pred = transformer(
                    hidden_states=noised_latent,
                    timestep=torch.Tensor([dt]),
                    pooled_projections=feeds["pooled_prompt_embeds"],
                    encoder_hidden_states=feeds["prompt_embeds"],
                    txt_ids=feeds["text_ids"],
                    img_ids=feeds["latent_image_ids"],
                    joint_attention_kwargs=None,
                    guidance=feeds["guidance"],
                    return_dict=False,
                )[0]

            noised_latent = noised_latent - noise_pred * dt
            image = cache_flux.decode_from_latent(
                noised_latent, vae_output.shape[2], vae_output.shape[3]
            )
            denoise_images.append(image)

        # save denoise images as gif
        denoise_images[0].save(
            "debug/denoise.gif",
            save_all=True,
            append_images=denoise_images[1:],
            duration=100,
            loop=0,
        )

        for i, (image, caption) in enumerate(dataset):
            cache_flux(image, caption, filename=f"image_{i}")
        print("Done")
