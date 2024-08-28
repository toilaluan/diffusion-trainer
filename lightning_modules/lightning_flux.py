import pytorch_lightning as L
import diffusers
import torch
import schedulefree
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
import math
from optimum.quanto import freeze, qfloat8, quantize, qint4
import bitsandbytes as bnb


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
            subfolder="transformer",
        )
        quantize(self.denoiser, weights=qint4)
        freeze(self.denoiser)
        self.apply_lora()
        self.denoiser.enable_gradient_checkpointing()
        self.print_trainable_parameters(self.denoiser)

    @staticmethod
    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def apply_lora(self):
        transformer_lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.denoiser.add_adapter(transformer_lora_config)

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
        **kwargs,
    ):
        print(latents.shape)
        print(timestep.shape)
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
        loss = torch.nn.functional.mse_loss(noise_pred, targets, reduction="none")
        loss = loss.mean(dim=(1, 2, 3))
        return loss

    def training_step(self, batch, batch_idx):
        feeds, targets, metadata = batch
        noise_pred = self(**feeds)
        loss = self.loss_fn(noise_pred, targets)
        mean_loss = loss.mean()
        steps = [item["step"] for item in metadata]
        log = {f"Step {step} loss": step_loss for step, step_loss in zip(steps, loss)}
        self.log(log, on_step=True, on_epoch=True)
        self.log("Mean loss", mean_loss, on_step=True, on_epoch=True)
        return mean_loss

    def configure_optimizers(self):
        params_to_optimize = list(
            filter(lambda p: p.requires_grad, self.denoiser.parameters())
        )
        optimizer = bnb.optim.Adam8bit(
            params_to_optimize,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            min_8bit_size=16384,
        )
        return optimizer
