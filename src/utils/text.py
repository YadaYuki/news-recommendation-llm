from typing import Callable
from transformers import PreTrainedTokenizer
import torch


def create_transform_fn_from_pretrained_tokenizer(
    tokenizer: PreTrainedTokenizer, max_length: int, padding: bool = True
) -> Callable[[list[str]], torch.Tensor]:
    def transform(texts: list[str]) -> torch.Tensor:
        return tokenizer(texts, return_tensors="pt", max_length=64, padding="max_length", truncation=True)["input_ids"]

    return transform
