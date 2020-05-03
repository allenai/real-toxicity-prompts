import json
import os
from typing import Set

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets.openwebtext_dataset import openwebtext_dataloader
from utils.constants import DATA_DIR, TEXTS_DIR


def load_webtext() -> Set[str]:
    print("Loading sample of WebText...")
    webtext_dir = DATA_DIR / 'webtext'
    texts = set()

    for file in webtext_dir.iterdir():
        if file.suffix != '.jsonl':
            continue
        with file.open() as f:
            curr_texts = []
            for line in f:
                example = json.loads(line)
                curr_texts.append(example['text'])
                if example['ended']:
                    text = ''.join(curr_texts)
                    texts.add(text)
                    curr_texts = []
    return texts


def main():
    webtext_set = load_webtext()
    open_webtext_dataloader = openwebtext_dataloader()
    count = sum(text in webtext_set
                for filename, text
                in tqdm(open_webtext_dataloader, desc='Comparing texts'))
    print("Number of overlapping documents:", count)


if __name__ == '__main__':
    main()
