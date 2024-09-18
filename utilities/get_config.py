import argparse
from utilities.dotable_config import Config
from trainer import FluxTrainer
from lightning_wrapped_modules import FluxLightning


def get_config():
    parser = argparse.ArgumentParser()
    FluxTrainer.get_args(parser)
    FluxLightning.get_args(parser)
    config = Config(parser=parser)
    return config


if __name__ == "__main__":
    config = get_config()
    print(config)

    def do_something(**kwargs):
        print(kwargs)

    do_something(**config.training)
