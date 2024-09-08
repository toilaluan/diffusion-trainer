from lightning_modules.lightning_flux import FluxLightning
from data.core_data import CoreCachedDataset, collate_fn
import torch
import os
import argparse
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
import wandb
import accelerate
import diffusers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to run training with various options."
    )

    parser.add_argument("--project", default="finetune-flux", help="Wandb project name")
    parser.add_argument("--max_epochs", type=int, default=20, help="Max epochs")
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=0.9,
        help="Validation check interval",
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--batch_size", default=1, help="Precision", type=int)

    return parser.parse_args()


args = parse_args()

wandb.init(project=args.project)

# model = FluxLightning(
#     denoiser_pretrained_path="black-forest-labs/FLUX.1-dev",
#     learning_rate=1e-5,
#     weight_decay=1e-8,
# )

transformer = diffusers.FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    subfolder="transformer",
)

FluxLightning.apply_lora(transformer)
FluxLightning.print_trainable_parameters(transformer)

cached_dataset = CoreCachedDataset(cached_folder="debug/test_cache")

train_dataloader = torch.utils.data.DataLoader(
    cached_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
)
val_dataloader = torch.utils.data.DataLoader(
    cached_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
)

optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-5, weight_decay=1e-8)

accelerator = accelerate.Accelerator()

transformer, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    transformer, optimizer, train_dataloader, val_dataloader
)

transformer.to(accelerator.device)
transformer.train()

total_steps = len(train_dataloader) * args.max_epochs
val_batch = next(iter(val_dataloader))

step = 0

lora_save_path = "lora_ckpt"


def loss_fn(pred, target):
    return torch.nn.functional.mse_loss(pred.float(), target.float(), reduction="mean")


while total_steps > 0:
    for i, batch in enumerate(train_dataloader):
        feeds, targets, metadata = batch
        for k, v in feeds.items():
            feeds[k] = v.to(accelerator.device)
        noise_pred = transformer(
            hidden_states=feeds["latents"],
            timestep=feeds["timestep"],
            pooled_projections=feeds["pooled_prompt_embeds"],
            encoder_hidden_states=feeds["prompt_embeds"],
            txt_ids=feeds["text_ids"],
            img_ids=feeds["latent_image_ids"],
            guidance=feeds["guidance"],
            return_dict=False,
        )[0]
        loss = loss_fn(noise_pred, targets)
        print(f"Step {step} Loss {loss}")

        # if step % 20 == 0:
        #     print("Validating")
        #     model = accelerator.unwrap_model(model)
        #     model.save_lora(lora_save_path)
        #     model.validation_step(val_batch, lora_save_path)
        # wandb.log({"loss": loss})
        step += 1
        total_steps -= 1
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
