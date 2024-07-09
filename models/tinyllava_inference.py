import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from PIL import Image

from models.base import Mllm


class TinyLlava(Mllm):

    def __init__(self, model_name_or_path, **kwargs):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
        )
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)
        self.config = self.model.config
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            model_max_length=self.config.tokenizer_model_max_length,
            padding_side=self.config.tokenizer_padding_side,
        )

    def evaluate(self, prompt, filepath):
        image = Image.open(filepath)
        output_text, genertaion_time = self.model.chat(
            prompt=prompt, image=image, tokenizer=self.tokenizer
        )
        return output_text
