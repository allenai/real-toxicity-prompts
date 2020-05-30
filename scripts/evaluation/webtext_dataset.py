import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class WebTextPretokenizedDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, local_rank=-1):
        assert os.path.isfile(file_path)
        logger.info(f"WebText: Loading WebText test set features from {file_path}, block size {block_size}")

        shard = np.load(file_path)
        block_idxs = np.arange(block_size, len(shard) - block_size + 1, block_size)
        self.examples = np.split(shard, block_idxs)
        assert all(len(block) == block_size for block in self.examples)
        logger.info(f"WebText: Loaded {len(self.examples)} blocks of size {block_size}")

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
