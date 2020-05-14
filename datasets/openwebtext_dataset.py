import os
from typing import Optional, List, Tuple

import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

from utils.constants import TEXTS_DIR


class OpenWebText(Dataset):
    def __init__(self, filenames: Optional[List[str]] = None):
        super().__init__()
        print("Loading list of OpenWebText files...")
        if filenames is not None:
            self.files = [TEXTS_DIR / filename for filename in filenames]
            assert all(file.exists for file in self.files)
        else:
            self.files = list(TEXTS_DIR.iterdir())

    def __getitem__(self, idx):
        file = self.files[idx]
        return file.name, file.read_text(errors='ignore')

    def __len__(self):
        return len(self.files)


class OpenWebTextTokenized(OpenWebText):
    def __init__(self, filenames: Optional[List[str]] = None, tokenizer: Optional[GPT2Tokenizer] = None):
        super().__init__(filenames=filenames)
        if not tokenizer:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer = tokenizer

    def __getitem__(self, idx) -> Tuple[str, np.array]:
        filename, text = super().__getitem__(idx)
        return filename, np.array(self.tokenizer.encode(text))


def openwebtext_dataloader(filenames: Optional[List[str]] = None, tokenized=False):
    if tokenized:
        dataset = OpenWebTextTokenized(filenames)
    else:
        dataset = OpenWebText(filenames)
    dataloader = DataLoader(dataset, num_workers=os.cpu_count(), collate_fn=lambda x: x[0])
    return dataloader
