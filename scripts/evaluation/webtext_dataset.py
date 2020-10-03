import json
import logging
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def blockify(tokens: np.array, block_size: int, inline_meta_tokens: List[int]):
    if inline_meta_tokens:
        block_size -= len(inline_meta_tokens)

    block_idxs = np.arange(block_size, len(tokens), block_size)
    examples = np.split(tokens, block_idxs)

    if len(examples[-1]) != block_size:
        examples = examples[:-1]

    if inline_meta_tokens:
        examples = [np.concatenate((inline_meta_tokens, ex)) for ex in examples]

    return examples


class WebTextPretokenizedDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, inline_meta: str = None,
                 local_rank=-1):
        assert os.path.isfile(file_path)
        logger.info(f"WebText: Loading WebText test set features from {file_path}, block size {block_size}")

        inline_metadata_tokens = []
        if inline_meta:
            print("Adding inline metadata:", inline_meta)
            inline_metadata_tokens = tokenizer.encode(inline_meta)
            print("Tokenized metadata:", inline_metadata_tokens)

        shard = np.load(file_path)
        self.examples = blockify(shard, block_size, inline_metadata_tokens)
        assert all(len(block) == block_size for block in self.examples)
        logger.info(f"WebText: Loaded {len(self.examples)} blocks of size {block_size}")
        logger.info(f"WebText: first example is {self.examples[0]}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class WebTextJsonlinesDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, local_rank=-1, add_eos=True):
        assert os.path.isfile(file_path)
        logger.info(f"WEBTEXT: Loading WebText test set features from {file_path}, block size {block_size}")

        with open(file_path, encoding="utf-8") as f:
            lines = []
            for line in map(json.loads, f):
                text = line['text']
                if add_eos and line['length'] < block_size:
                    text += tokenizer.eos_token
                lines.append(text)

        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        print(f"WEBTEXT: Loaded {len(self.examples)} webtext lines")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
