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
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.denoiser = diffusers.FluxTransformer2DModel.from_pretrained(
            denoiser_pretrained_path,
            torch_dtype=torch.float16,
        )

    def forward(
        self,
        latents: torch.Tensor = None,
        timestep: int = None,
        guidance: float = None,
        pooled_prompt_embeds: torch.Tensor = None,
        prompt_embeds: torch.Tensor = None,
        text_ids: torch.Tensor = None,
        latent_image_ids: torch.Tensor = None,
        joint_attention_kwargs: dict = None,
        **kwargs
    ):
        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]
        return noise_pred

    def loss_fn(self, noise_pred, targets):
        return torch.nn.functional.mse_loss(noise_pred, targets)

    def training_step(self, batch, batch_idx):
        metadata, feeds, targets = batch
        noise_pred = self(**feeds)
        loss = self.loss_fn(noise_pred, targets)
        return loss

    def configure_optimizers(self):
        optimizer = schedulefree.AdamWScheduleFree(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer
