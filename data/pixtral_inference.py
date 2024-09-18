from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import (
    UserMessage,
    TextChunk,
    ImageChunk,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from huggingface_hub import snapshot_download
from pathlib import Path


class PixtralInference:
    def __init__(self, config):
        """
        Initialize the Pixtral inference pipeline.

        Args:
            config (Namespace): Configuration object containing model repository and other settings.
        """
        self.mistral_models_path = Path.home().joinpath("mistral_models", "Pixtral")
        self.mistral_models_path.mkdir(parents=True, exist_ok=True)

        self._download_model(config.model_repo)
        self.tokenizer = MistralTokenizer.from_file(
            f"{self.mistral_models_path}/tekken.json"
        )
        self.model = Transformer.from_folder(self.mistral_models_path)

    def _download_model(self, repo_id):
        """
        Download the model files from the repository.

        Args:
            repo_id (str): Repository ID to download the model from.
        """
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"],
            local_dir=self.mistral_models_path,
        )

    def infer(self, prompt, image, **kwargs):
        """
        Perform inference on a given image and prompt.

        Args:
            prompt (str): The text prompt for inference.
            image (PIL.Image.Image): The image for inference.
            kwargs: Additional keyword arguments for generation settings.

        Returns:
            str: The generated result after inference.
        """
        completion_request = ChatCompletionRequest(
            messages=[
                UserMessage(content=[ImageChunk(image=image), TextChunk(text=prompt)])
            ]
        )

        encoded = self.tokenizer.encode_chat_completion(completion_request)
        images = encoded.images
        tokens = encoded.tokens

        out_tokens, _ = generate(
            [tokens],
            self.model,
            images=[images],
            eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id,
            **kwargs,
        )

        result = self.tokenizer.decode(out_tokens[0])
        return result

    @staticmethod
    def get_args(parser):
        """
        Defines the arguments for configuring Pixtral inference.

        Args:
            parser (ArgumentParser): Argument parser object used to define command-line arguments.

        Arguments:
            --pixtral_inference.model_repo (str): Model repository to download the model from. Default: "mistral-community/pixtral-12b-240910".
            --pixtral_inference.max_tokens (int): Maximum number of tokens for the model's output. Default: 256.
            --pixtral_inference.temperature (float): Sampling temperature for inference. Default: 0.35.
            --pixtral_inference.trigger (str): Trigger keyword for generating responses. Default: "OHNX".
            --pixtral_inference.caption_type (str): Type of caption generation ("short" or "long"). Default: "short".
        """
        parser.add_argument(
            "--pixtral_inference.model_repo",
            type=str,
            default="mistral-community/pixtral-12b-240910",
            help="Model repository",
        )
        parser.add_argument(
            "--pixtral_inference.max_tokens",
            type=int,
            default=256,
            help="Max tokens",
        )
        parser.add_argument(
            "--pixtral_inference.temperature",
            type=float,
            default=0.35,
            help="Temperature",
        )
        parser.add_argument(
            "--pixtral_inference.trigger",
            type=str,
            default="OHNX",
            help="Trigger",
        )
        parser.add_argument(
            "--pixtral_inference.caption_type",
            choices=["short", "long"],
            default="short",
            help="Caption type",
        )