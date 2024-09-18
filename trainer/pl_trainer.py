import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from data.core_data import CoreCachedDataset, collate_fn
import torch


class Trainer(pl.Trainer):
    def __init__(self, config, **kwargs):
        super().__init__(
            logger=WandbLogger(project=config.project),
            **config,
            **kwargs,
        )
        self.cached_dataset = CoreCachedDataset(cached_folder=config.cache_dir)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.cached_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.cached_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def fit(self, model):
        model.save_lora_every_n_epoch = self.config.save_lora_every_n_epoch
        super().fit(model, self.train_dataloader, self.val_dataloader)

    @staticmethod
    def get_args(parser):
        parser.add_argument("--training.num_epochs", type=int, default=100)
        parser.add_argument(
            "--training.project",
            default="finetune-flux",
            help="Wandb project name",
        )
        parser.add_argument(
            "--training.batch_size",
            type=int,
            default=1,
            help="Batch size",
        )
        parser.add_argument(
            "--training.max_epochs",
            type=int,
            default=20,
            help="Max epochs",
        )
        parser.add_argument(
            "--training.log_every_n_steps",
            type=int,
            default=1,
            help="Log every n steps",
        )
        parser.add_argument(
            "--training.gpus",
            type=int,
            default=1,
            help="Number of GPUs",
        )
        parser.add_argument(
            "--training.precision",
            default="bf16",
            help="Precision",
        )
        parser.add_argument(
            "--training.accelerator",
            default="gpu",
            help="Accelerator",
        )
        parser.add_argument(
            "--training.accumulate_grad_batches",
            type=int,
            default=1,
            help="Accumulate grad batches",
        )
        parser.add_argument(
            "--training.strategy",
            default="auto",
            help="Strategy",
        )
        parser.add_argument("--training.devices", default=1)
        parser.add_argument(
            "--training.check_val_every_n_epoch",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--training.cache_dir",
            default="debug/cache_tshirt",
            type=str,
        )
        parser.add_argument(
            "--training.limit_val_batches",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--training.save_lora_every_n_epoch",
            default=1,
            type=int,
        )
