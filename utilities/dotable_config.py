import argparse
from munch import DefaultMunch
import copy
import yaml
import os
import sys
from typing import List, Optional, Any, Dict


class Config(DefaultMunch):
    """
    Class to combine argparse for command-line parsing with munch for nested dictionary support.
    """

    def __init__(
        self,
        parser: argparse.ArgumentParser = None,
        args: Optional[List[str]] = None,
        strict: bool = False,
        default: Optional[Any] = None,
    ) -> None:
        super().__init__(default)
        self["__is_set"] = {}

        if parser is None:
            return

        # Optionally add config-specific arguments
        parser.add_argument(
            "--config",
            type=str,
            help="Overrides defaults with passed configuration file.",
        )

        # Parse args from argv if not passed in
        if args is None:
            args = sys.argv[1:]

        # Load defaults from the config file, if provided
        config_file_path = vars(parser.parse_known_args(args)[0]).get("config")
        if config_file_path:
            config_file_path = os.path.expanduser(config_file_path)
            with open(config_file_path) as f:
                params_config = yaml.safe_load(f)
                parser.set_defaults(**params_config)

        # Parse arguments, non-strict
        params = self.__parse_args__(args=args, parser=parser, strict=False)

        # Split parameters into the config structure
        self.__split_params__(params)

    @staticmethod
    def __parse_args__(
        args: List[str], parser: argparse.ArgumentParser = None, strict: bool = False
    ) -> argparse.Namespace:
        """Parse the passed args using the parser."""
        if not strict:
            params, unrecognized = parser.parse_known_args(args=args)
            params_list = list(params.__dict__)
            for unrec in unrecognized:
                if unrec.startswith("--") and unrec[2:] in params_list:
                    setattr(params, unrec[2:], True)
        else:
            params = parser.parse_args(args=args)
        return params

    def __split_params__(self, params: argparse.Namespace):
        """Splits params on dot syntax and adds to config."""
        for arg_key, arg_val in params.__dict__.items():
            keys = arg_key.split(".")
            current = self
            for key in keys[:-1]:
                if key not in current:
                    current[key] = Config()
                current = current[key]
            current[keys[-1]] = arg_val

    def __deepcopy__(self, memo) -> "Config":
        _default = self.__default__
        config_state = self.__getstate__()
        config_copy = Config()
        memo[id(self)] = config_copy
        config_copy.__setstate__(config_state)
        config_copy.__default__ = _default
        config_copy["__is_set"] = copy.deepcopy(self["__is_set"], memo)
        return config_copy

    def __str__(self) -> str:
        visible = copy.deepcopy(self.toDict())
        visible.pop("__is_set", None)
        return "\n" + yaml.dump(visible, sort_keys=False)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model.learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for the model.",
    )
    parser.add_argument(
        "--model.optimizer", type=str, default="adam", help="Optimizer for the model."
    )
    parser.add_argument(
        "--data.batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--data.shuffle", action="store_true", help="Shuffle the dataset."
    )

    # Initialize the config
    cfg = Config(parser=parser)

    print(cfg)
