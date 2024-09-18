import pytorch_lightning as L
import diffusers
import torch
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from optimum.quanto import freeze, qfloat8, quantize, qint4
import wandb
from prodigyopt import Prodigy


class FluxLightning(L.LightningModule):
    def __init__(
        self,
        model_config=None,
        optimizer_config=None,
    ):
        """
        Args:
            model_config (Namespace): Configuration object for model parameters, including LoRA settings and quantization.
            optimizer_config (Namespace): Configuration object for optimizer parameters such as learning rate and weight decay.
        """
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.learning_rate = optimizer_config.lr
        self.weight_decay = optimizer_config.weight_decay
        self.torch_dtype = torch.bfloat16
        self.denoiser = diffusers.FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
            torch_dtype=self.torch_dtype,
        )
        if not model_config.quanto:
            pass
        elif model_config.quanto == "qint4":
            print("Quantizing model to qint4")
            quantize(self.denoiser, weights=qint4)
            freeze(self.denoiser)
        elif model_config.quanto == "qfloat8":
            print("Quantizing model to qfloat8")
            quantize(self.denoiser, weights=qfloat8)
            freeze(self.denoiser)
        else:
            raise ValueError(f"Unknown quantization method {model_config.quanto}")
        self.apply_lora(model_config.lora_rank, model_config.lora_alpha)
        self.denoiser.enable_gradient_checkpointing()
        self.denoiser.train()
        self.print_trainable_parameters(self.denoiser)
        self.pipeline = diffusers.FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=self.torch_dtype,
            transformer=self.denoiser,
            text_encoder=None,
            text_encoder_2=None,
        )
        self.save_lora_every_n_epoch = 1
        self.save_hyperparameters()

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

    def apply_lora(self, r, lora_alpha):
        """
        Applies LoRA (Low-Rank Adaptation) to the model's transformer layers.

        Args:
            r (int): The rank of the LoRA adaptation.
            lora_alpha (int): Alpha parameter controlling the scaling of LoRA weights.
        """
        transformer_lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0."],
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
        """
        Forward pass for the denoiser model.

        Args:
            latents (torch.Tensor): Latent variables for the input.
            timestep (int): Timestep for the denoising process.
            pooled_prompt_embeds (torch.Tensor): Pooled embeddings from the prompt.
            prompt_embeds (torch.Tensor): Full prompt embeddings.
            text_ids (torch.Tensor): Tokenized text IDs.
            latent_image_ids (torch.Tensor): Image-related latent IDs.
            joint_attention_kwargs (dict): Dictionary of joint attention configurations.
            guidance (torch.Tensor): Guidance input for the denoising process.

        Returns:
            torch.Tensor: Predicted noise values for the input latents.
        """
        noise_pred = self.denoiser(
            hidden_states=latents,
            timestep=timestep,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            guidance=guidance,
            return_dict=False,
        )[0]
        return noise_pred

    def loss_fn(self, noise_pred, targets):
        """
        Loss function for training the denoiser model.

        Args:
            noise_pred (torch.Tensor): Predicted noise from the denoiser.
            targets (torch.Tensor): Ground truth noise for training.

        Returns:
            torch.Tensor: Computed loss.
        """
        noise_pred = self.pipeline._unpack_latents(
            noise_pred,
            height=targets.shape[2] * 8,
            width=targets.shape[3] * 8,
            vae_scale_factor=16,
        )
        loss = torch.mean(
            ((noise_pred.float() - targets.float()) ** 2).reshape(targets.shape[0], -1),
            1,
        ).mean()
        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step that computes loss and logs it during training.

        Args:
            batch (tuple): A tuple containing input features, targets, and metadata.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss for the current training step.
        """
        feeds, targets, metadata = batch
        noise_pred = self(**feeds)
        loss = self.loss_fn(noise_pred, targets)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_start(self) -> None:
        """
        Called at the start of training to save LoRA models at regular intervals.
        """
        super().on_train_start()
        if self.current_epoch % self.save_lora_every_n_epoch == 0:
            self.save_lora(f"checkpoints/lora_epoch-{self.current_epoch}")

    def validation_step(self, batch, batch_idx):
        """
        Validation step that generates and logs an image for visualization.

        Args:
            batch (tuple): A tuple containing input features, targets, and metadata.
            batch_idx (int): Index of the current batch.
        """
        feeds, targets, metadata = batch
        prompt_embeds = feeds["prompt_embeds"][:1]
        pooled_prompt_embeds = feeds["pooled_prompt_embeds"][:1]
        width = 1024
        height = 1024
        steps = 30
        image = self.pipeline(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(42),
            guidance_scale=3.5,
        ).images[0]
        self.denoiser.train()
        image = wandb.Image(image, caption="TODO: Add caption")
        wandb.log({f"Validation {batch_idx} image": image})

    def save_lora(self, path: str):
        """
        Saves the LoRA-adapted weights.

        Args:
            path (str): Path to save the LoRA weights.
        """
        transformer_lora_layers = get_peft_model_state_dict(self.denoiser)
        diffusers.FluxPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=transformer_lora_layers,
        )

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        params_to_optimize = list(
            filter(lambda p: p.requires_grad, self.denoiser.parameters())
        )
        if self.optimizer_config.type == "adamw":
            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=self.optimizer_config.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_config.type == "prodigy":
            optimizer = Prodigy(
                params_to_optimize,
                lr=1.0,
                weight_decay=self.optimizer_config.weight_decay,
                decouple=True,
                d_coef=0.8,
                use_bias_correction=True,
            )
        return optimizer

    @staticmethod
    def get_optimizer_args(parser):
        """
        Defines arguments for optimizer configuration.

        Args:
            parser (ArgumentParser): Argument parser object used to define optimizer-related command-line arguments.

        Arguments:
            --optimizer.weight_decay (float): Weight decay for the optimizer. Default: 0.1.
            --optimizer.lr (float): Learning rate for the optimizer. Default: 1.0.
            --optimizer.type (str): Type of optimizer to use (e.g., "adamw", "prodigy"). Default: "prodigy".
        """
        parser.add_argument("--optimizer.weight_decay", type=float, default=0.1)
        parser.add_argument("--optimizer.lr", type=float, default=1.0)
        parser.add_argument("--optimizer.type", type=str, default="prodigy")

    @staticmethod
    def get_model_args(parser):
        """
        Defines arguments for model configuration.

        Args:
            parser (ArgumentParser): Argument parser object used to define model-related command-line arguments.

        Arguments:
            --model.lora_rank (int): Rank for LoRA adaptation. Default: 16.
            --model.lora_alpha (int): Alpha for LoRA scaling. Default: 16.
            --model.quanto (str): Quantization method for model ("qint4", "qfloat8"). Default: "" (no quantization).
        """
        parser.add_argument("--model.lora_rank", type=int, default=16)
        parser.add_argument("--model.lora_alpha", type=int, default=16)
        parser.add_argument("--model.quanto", type=str, default="")

    @staticmethod
    def get_args(parser):
        """
        Combines model and optimizer argument definitions.

        Args:
            parser (ArgumentParser): Argument parser object used to define all command-line arguments.
        """
        FluxLightning.get_model_args(parser)
        FluxLightning.get_optimizer_args(parser)
