import lightning as L
import diffusers
import torch
import schedulefree


class FluxLightning(L.LightningModule):
    def __init__(
        self,
        denoiser_pretrained_path: str,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.torch_dtype = torch_dtype
        self.denoiser = diffusers.FluxTransformer2DModel.from_pretrained(
            denoiser_pretrained_path,
            torch_dtype=self.torch_dtype,
            subfolder="transformer",
        )

    def forward(
        self,
        latents: torch.Tensor = None,
        timestep: int = None,
        pooled_prompt_embeds: torch.Tensor = None,
        prompt_embeds: torch.Tensor = None,
        text_ids: torch.Tensor = None,
        latent_image_ids: torch.Tensor = None,
        joint_attention_kwargs: dict = None,
        guidance: torch.Tensor = None,
        **kwargs
    ):
        print(latents.shape)
        print(pooled_prompt_embeds.shape)
        print(prompt_embeds.shape)
        print(text_ids.shape)
        print(latent_image_ids.shape)
        print(guidance.shape)
        noise_pred = self.denoiser(
            hidden_states=latents,
            timestep=timestep,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            guidance=guidance,
            return_dict=False,
        )[0]
        return noise_pred

    def loss_fn(self, noise_pred, targets):
        return torch.nn.functional.mse_loss(noise_pred, targets)

    def training_step(self, batch, batch_idx):
        feeds, targets, metadata = batch
        noise_pred = self(**feeds)
        loss = self.loss_fn(noise_pred, targets)
        return loss

    def configure_optimizers(self):
        optimizer = schedulefree.AdamWScheduleFree(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer
