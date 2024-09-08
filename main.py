from lightning_modules.lightning_flux import FluxLightning
from data.core_data import CoreCachedDataset, collate_fn
import torch
import os
import argparse
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
import wandb
import accelerate


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

model = FluxLightning(
    denoiser_pretrained_path="black-forest-labs/FLUX.1-dev",
    learning_rate=1e-5,
    weight_decay=1e-8,
)

cached_dataset = CoreCachedDataset(cached_folder="debug/test_cache")

train_dataloader = torch.utils.data.DataLoader(
    cached_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
)
val_dataloader = torch.utils.data.DataLoader(
    cached_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
)

optimizer = model.configure_optimizers()

accelerator = accelerate.Accelerator()

model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader
)

model.to(accelerator.device)
model.pipeline.to(accelerator.device)

total_steps = len(train_dataloader) * args.max_epochs

model.train()
val_batch = next(iter(val_dataloader))

step = 0

lora_save_path = "lora_ckpt"

while total_steps > 0:
    for i, batch in enumerate(train_dataloader):
        feeds, targets, metadata = batch
        for k, v in feeds.items():
            feeds[k] = v.to(model.denoiser.device)
        noise_pred = model(**feeds)
        loss = model.loss_fn(noise_pred, targets)
        print(f"Step {step} Loss {loss}")

        if step % 20 == 0:
            print("Validating")
            model = accelerator.unwrap_model(model)
            model.save_lora(lora_save_path)
            model.validation_step(val_batch, lora_save_path)
        wandb.log({"loss": loss})
        step += 1
        total_steps -= 1
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
