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


class MistralInference:
    def __init__(self, model_repo="mistral-community/pixtral-12b-240910"):
        self.mistral_models_path = Path.home().joinpath("mistral_models", "Pixtral")
        self.mistral_models_path.mkdir(parents=True, exist_ok=True)

        self._download_model(model_repo)
        self.tokenizer = MistralTokenizer.from_file(
            f"{self.mistral_models_path}/tekken.json"
        )
        self.model = Transformer.from_folder(self.mistral_models_path)

    def _download_model(self, repo_id):
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"],
            local_dir=self.mistral_models_path,
        )

    def infer(self, prompt, image, **kwargs):
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


if __name__ == "__main__":
    import argparse
    import glob
    import json
    from PIL import Image
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Mistral Inference")
    parser.add_argument("--root-folder", type=str, default="dataset/tshirt")
    parser.add_argument("--caption-type", choices=["short", "long"], default="short")
    parser.add_argument("--trigger", type=str, default="OHNX tshirt")
    args = parser.parse_args()

    image_files = glob.glob(f"{args.root_folder}/images/*.jpg")

    mistral_inference = MistralInference()
    prompts = {
        "short": "Describe the image as a short caption",
        "long": "Describe the image",
    }

    prompt = prompts[args.caption_type]
    metadata = []
    for image_file in tqdm(image_files):
        try:
            image = Image.open(image_file)
            image = image.convert("RGB")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
        caption = mistral_inference.infer(
            prompt=prompt, image=image, max_tokens=256, temperature=0.35
        )
        caption = args.trigger + ". " + caption
        metadata.append({"image": image_file.split("/")[-1], "caption": caption})

        with open(f"{args.root_folder}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
