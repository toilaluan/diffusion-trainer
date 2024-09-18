from utilities.get_config import get_config
from lightning_wrapped_modules import FluxLightning
from trainer import FluxTrainer

if __name__ == "__main__":

    config = get_config()

    print(config)

    model = FluxLightning(config.model, config.optimizer)
    model.denoiser.to("cuda")
    model.pipeline.to("cuda")

    trainer = FluxTrainer(config.training, config.core_cached_dataset)

    trainer.fit(model)
