from lightning_modules.lightning_flux import FluxLightning
from data.core_data import CoreCachedDataset, collate_fn
import torch
import pytorch_lightning as pl
import os
import argparse
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor


class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")


callbacks = [
    MyPrintingCallback(),
    ModelCheckpoint(
        dirpath="checkpoints",
        every_n_train_steps=100,
    ),
    LearningRateMonitor("step"),
]


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
    parser.add_argument("--precision", default="32", help="Precision")
    parser.add_argument("--accelerator", default="gpu", help="Accelerator")
    parser.add_argument(
        "--accumulate_grad_batches", type=int, default=1, help="Accumulate grad batches"
    )
    parser.add_argument("--strategy", default="auto", help="Strategy")
    parser.add_argument("--devices", default=1)

    return parser.parse_args()


args = parse_args()

wandb_logger = WandbLogger(project="flux-lora")

model = FluxLightning(
    denoiser_pretrained_path="black-forest-labs/FLUX.1-dev",
    learning_rate=1e-5,
    weight_decay=1e-5,
)
model.to("cuda")

cached_dataset = CoreCachedDataset(cached_folder="debug/test_cache")

train_dataloader = torch.utils.data.DataLoader(
    cached_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
)

trainer = pl.Trainer(
    accelerator=args.accelerator,
    accumulate_grad_batches=args.accumulate_grad_batches,
    precision=args.precision,
    callbacks=callbacks,
    max_epochs=args.max_epochs,
    val_check_interval=args.val_check_interval,
    log_every_n_steps=args.log_every_n_steps,
    logger=wandb_logger,
    strategy=args.strategy,
    devices=args.devices,
    limit_val_batches=1,
)

trainer.fit(model, train_dataloader, train_dataloader)
