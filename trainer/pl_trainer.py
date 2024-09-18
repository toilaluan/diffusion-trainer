import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from data.core_data import CoreCachedDataset, collate_fn
import torch


class FluxTrainer(pl.Trainer):
    def __init__(self, train_config, dataset_config, **kwargs):
        """
        Args:
            train_config (Namespace): Configuration object for training parameters such as
                project name, batch size, number of epochs, logging frequency, precision, etc.
            dataset_config (Namespace): Configuration object for dataset-related parameters.
            **kwargs: Additional keyword arguments for customization.
        """
        super().__init__(
            logger=WandbLogger(project=train_config.project),
            max_epochs=train_config.max_epochs,
            log_every_n_steps=train_config.log_every_n_steps,
            precision=train_config.precision,
            accelerator=train_config.accelerator,
            accumulate_grad_batches=train_config.accumulate_grad_batches,
            strategy=train_config.strategy,
            devices=train_config.devices,
            check_val_every_n_epoch=train_config.check_val_every_n_epoch,
            limit_val_batches=train_config.limit_val_batches,
        )
        self.cached_dataset = CoreCachedDataset(dataset_config)
        self._train_dataloader = torch.utils.data.DataLoader(
            self.cached_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        self._val_dataloader = torch.utils.data.DataLoader(
            self.cached_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        self.train_config = train_config

    def fit(self, model):
        """
        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
        """
        model.save_lora_every_n_epoch = self.train_config.save_lora_every_n_epoch
        super().fit(model, self._train_dataloader, self._val_dataloader)

    @staticmethod
    def get_args(parser):
        """
        Defines the arguments for configuring the training and dataset setup.

        Args:
            parser (ArgumentParser): Argument parser object used to define command-line arguments.

        Arguments:
            --training.project (str): Wandb project name. Default: "finetune-flux".
            --training.batch_size (int): Batch size for training and validation. Default: 1.
            --training.max_epochs (int): Maximum number of epochs to train. Default: 100.
            --training.log_every_n_steps (int): Log metrics every 'n' steps. Default: 1.
            --training.gpus (int): Number of GPUs to use for training. Default: 1.
            --training.precision (str): Precision to use during training (e.g., "bf16", "fp16"). Default: "bf16".
            --training.accelerator (str): Accelerator for training (e.g., "gpu", "cpu"). Default: "gpu".
            --training.accumulate_grad_batches (int): Number of batches to accumulate gradients for. Default: 1.
            --training.strategy (str): Strategy for distributed training (e.g., "ddp", "dp"). Default: "auto".
            --training.devices (int): Number of devices to use for training (e.g., GPUs or TPUs). Default: 1.
            --training.check_val_every_n_epoch (int): How often to check validation (in epochs). Default: 10.
            --training.cache_dir (str): Directory to store cached data. Default: "debug/cache_tshirt".
            --training.limit_val_batches (int): Number of validation batches to run per epoch. Default: 1.
            --training.save_lora_every_n_epoch (int): Number of epochs between saving LoRA models. Default: 5.
        """
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
            default=100,
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
            help="Precision (e.g., bf16, fp16, 32-bit).",
        )
        parser.add_argument(
            "--training.accelerator",
            default="gpu",
            help="Accelerator (e.g., cpu, gpu, tpu).",
        )
        parser.add_argument(
            "--training.accumulate_grad_batches",
            type=int,
            default=1,
            help="Accumulate grad batches before performing a backward pass.",
        )
        parser.add_argument(
            "--training.strategy",
            default="auto",
            help="Training strategy (e.g., ddp, dp, auto).",
        )
        parser.add_argument(
            "--training.devices",
            default=1,
            help="Number of devices (e.g., GPUs) to use for training.",
        )
        parser.add_argument(
            "--training.check_val_every_n_epoch",
            default=10,
            type=int,
            help="Perform validation every n epochs.",
        )
        parser.add_argument(
            "--training.cache_dir",
            default="debug/cache_tshirt",
            type=str,
            help="Directory for cached datasets.",
        )
        parser.add_argument(
            "--training.limit_val_batches",
            default=1,
            type=int,
            help="Limit the number of validation batches.",
        )
        parser.add_argument(
            "--training.save_lora_every_n_epoch",
            default=5,
            type=int,
            help="Save LoRA weights every n epochs.",
        )
