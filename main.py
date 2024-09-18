from lightning_wrapped_modules.flux_lightning import FluxLightning
from trainer.pl_trainer import Trainer
from utilities.get_config import get_config

if __name__ == "__main__":

    config = get_config()

    print(config)

    model = FluxLightning(config.model, config.optimizer)

    trainer = Trainer(config.training)

    trainer.fit(model)
