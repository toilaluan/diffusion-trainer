from data import CoreDataset, CoreCachedDataset, CacheFlux, PixtralInference
from utilities import Config
import glob
from PIL import Image
from tqdm import tqdm
import json
import torch
import os
import diffusers
import argparse


class PreprocessPipeline:
    def __init__(self, config=None):
        """
        Initialize the PreprocessPipeline.

        Args:
            config (Namespace, optional): Configuration object for the preprocess pipeline. If not provided,
            the configuration will be loaded from arguments passed via argparse.

        Attributes:
            core_dataset (CoreDataset): Dataset object for handling the core dataset.
            cache_flux (CacheFlux): Caching utility for processing and storing image embeddings.
            pixtral_inference (PixtralInference, optional): Model used for generating image captions,
            initialized only if captioning is enabled.
            config (Namespace): Configuration object containing all necessary parameters.
        """
        if config is None:
            parser = argparse.ArgumentParser()
            config = Config.get_args(parser)
            print(config)
        self.core_dataset = CoreDataset(
            config=config.core_dataset,
        )
        self.cache_flux = CacheFlux(config=config.cache_flux)
        if config.preprocess_pipeline.do_captioning:
            self.pixtral_inference = PixtralInference(config.pixtral_inference)

        self.config = config

    @staticmethod
    def get_args(parser):
        """
        Define the command-line arguments for the PreprocessPipeline.

        Args:
            parser (ArgumentParser): Argument parser used to define command-line arguments.

        Arguments:
            --preprocess_pipeline.do_captioning (bool): Whether to generate captions for the images and create a metadata file. Default is False.
            --preprocess_pipeline.debug (bool): Whether to enable debugging mode for cache flux and inspect intermediate steps. Default is False.

        Returns:
            Config: Configuration object containing the parsed arguments.
        """
        CoreDataset.get_args(parser)
        CoreCachedDataset.get_args(parser)
        CacheFlux.get_args(parser)
        PixtralInference.get_args(parser)

        parser.add_argument(
            "--preprocess_pipeline.do_captioning",
            default=False,
            help="Whether to do captioning and create metadata file",
            action="store_true",
        )
        parser.add_argument(
            "--preprocess_pipeline.debug",
            default=False,
            help="Whether to do cache flux",
            action="store_true",
        )

        config = Config(parser=parser)

        return config

    def start(self):
        """
        Start the preprocessing pipeline. Depending on the configuration, this function can:
        - Perform captioning for images and save metadata in JSON format.
        - Apply cache flux to store latent embeddings for the images.
        - Debug intermediate outputs of the image caching and denoising process.
        """
        if self.config.preprocess_pipeline.do_captioning:
            # Image captioning process
            image_files = glob.glob(
                f"{self.config.core_dataset.root_folder}/images/*.jpg"
            )
            mistral_inference = PixtralInference(self.config.pixtral_inference)
            prompts = {
                "short": "Describe the image as a short caption",
                "long": "Describe the image",
            }

            prompt = prompts[self.config.pixtral_inference.caption_type]
            metadata = []
            for image_file in tqdm(image_files):
                try:
                    image = Image.open(image_file)
                    image = image.convert("RGB")
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                    continue
                caption = mistral_inference.infer(
                    prompt=prompt,
                    image=image,
                    max_tokens=self.config.pixtral_inference.max_tokens,
                    temperature=self.config.pixtral_inference.temperature,
                )
                caption = self.config.pixtral_inference.trigger + ". " + caption
                metadata.append(
                    {"image_path": image_file.split("/")[-1], "caption": caption}
                )

                with open(
                    f"{self.config.core_dataset.root_folder}/metadata.json",
                    "w",
                ) as f:
                    json.dump(metadata, f, indent=4)

        with torch.no_grad():
            if self.config.preprocess_pipeline.debug:
                # Debugging mode: cache flux and visualize intermediate results
                print("Debugging cache flux")
                os.makedirs("debug", exist_ok=True)
                image, caption = self.core_dataset[0]

                width, height = image.size

                image.save("debug/original_image.jpg")
                self.cache_flux(
                    image,
                    caption,
                    filename="cached_image",
                )
                feeds = torch.load(
                    os.path.join(self.config.cache_flux.cache_dir, "cached_image.pt"),
                    weights_only=True,
                )
                vae_output = feeds["vae_latents"]
                print("VAE shape", vae_output.shape)

                print("Debugging cache flux decode")
                image = self.cache_flux.decode_from_latent(
                    feeds["latents"], width, height
                )
                image.save("debug/encode_then_decode_image.jpg")
                core_cached_dataset = CoreCachedDataset(self.config.core_cached_dataset)
                noised_latent = core_cached_dataset.get_noised_latent(0, 0.5)
                image = self.cache_flux.decode_from_latent(
                    noised_latent, width=width, height=height
                )
                image.save("debug/noised_image.jpg")

                print("Debugging transformer denoise")
                transformer = diffusers.FluxTransformer2DModel.from_pretrained(
                    "black-forest-labs/FLUX.1-dev",
                    subfolder="transformer",
                    torch_dtype=torch.bfloat16,
                )
                transformer.to("cuda")

                num_inference_steps = 30
                denoise_images = []
                noised_latent = noised_latent.cuda()

                sigmas = torch.linspace(0, 1, num_inference_steps)
                mu = CacheFlux.calculate_shift(noised_latent.shape[1])
                print("mu", mu)
                sigmas = CacheFlux.time_shift(mu, 1.0, sigmas)
                sigmas = sigmas.flip(0)
                print("sigmas", sigmas)
                print("sigmas shape", sigmas.shape)

                # Reverse the sigmas
                pbar = tqdm(
                    total=num_inference_steps,
                    desc="Debugging denoising from cached image",
                )
                for i in range(num_inference_steps - 1):
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        noise_pred = transformer(
                            hidden_states=noised_latent,
                            timestep=sigmas[i].expand(1).cuda(),
                            pooled_projections=feeds["pooled_prompt_embeds"].cuda(),
                            encoder_hidden_states=feeds["prompt_embeds"].cuda(),
                            txt_ids=feeds["text_ids"].cuda(),
                            img_ids=feeds["latent_image_ids"].cuda(),
                            joint_attention_kwargs=None,
                            guidance=feeds["guidance"].cuda(),
                            return_dict=False,
                        )[0]

                    noised_latent = (
                        noised_latent + (sigmas[i + 1] - sigmas[i]) * noise_pred
                    )
                    image = self.cache_flux.decode_from_latent(
                        noised_latent, width=width, height=height
                    )
                    denoise_images.append(image)
                    pbar.update(1)
                pbar.close()

                # Save denoise images as a gif
                denoise_images[0].save(
                    "debug/denoising_process.gif",
                    save_all=True,
                    append_images=denoise_images[1:],
                    duration=100,
                    loop=0,
                )
                os.remove(
                    os.path.join(self.config.cache_flux.cache_dir, "cached_image.pt")
                )

            # Standard caching process
            pbar = tqdm(total=len(self.core_dataset), desc="Caching all dataset")
            for i, (image, caption) in enumerate(self.core_dataset):
                self.cache_flux(image, caption, filename=f"image_{i}")
                pbar.update(1)
            pbar.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    config = PreprocessPipeline.get_args(parser)
    print(config)
    pipeline = PreprocessPipeline(config)
    pipeline.start()
