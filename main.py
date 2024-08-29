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
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=20, help="Max epochs")
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=0.9,
        help="Validation check interval",
    )
    parser.add_argument(
        "--log_every_n_steps", type=int, default=1, help="Log every n steps"
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--precision", default="bf16", help="Precision")
    parser.add_argument("--accelerator", default="gpu", help="Accelerator")
    parser.add_argument(
        "--accumulate_grad_batches", type=int, default=1, help="Accumulate grad batches"
    )
    parser.add_argument("--strategy", default="auto", help="Strategy")
    parser.add_argument("--devices", default=1)
    parser.add_argument("--check_val_every_n_epoch", default=5, type=int)

    return parser.parse_args()


args = parse_args()

wandb.init(project=args.project)

model = FluxLightning(
    denoiser_pretrained_path="black-forest-labs/FLUX.1-dev",
    learning_rate=1e-5,
    weight_decay=1e-5,
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

model.to(accelerator.device)

model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader
)

total_steps = len(train_dataloader) * args.max_epochs

model.train()

while total_steps > 0:

    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        loss = model.training_step(batch, 0)
        wandb.log({"loss": loss})
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Step {i} Loss {loss}")

        if i % 50 == 0:
            print("Validating")
            for j, val_batch in enumerate(val_dataloader):
                model.validation_step(val_batch, j)

        total_steps -= 1
