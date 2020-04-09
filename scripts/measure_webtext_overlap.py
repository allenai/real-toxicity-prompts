import json
import os
from typing import Set

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.constants import DATA_DIR, TEXTS_DIR


class OpenWebText(Dataset):
    def __init__(self):
        super().__init__()
        print("Loading list of OpenWebText files...")
        self.files = list(TEXTS_DIR.iterdir())

    def __getitem__(self, idx):
        file = self.files[idx]
        return file.name, file.read_text(errors='ignore')

    def __len__(self):
        return len(self.files)


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
    open_webtext_dataset = OpenWebText()
    open_webtext_dataloader = DataLoader(open_webtext_dataset, num_workers=os.cpu_count(), collate_fn=lambda x: x[0])
    count = sum(text in webtext_set
                for filename, text
                in tqdm(open_webtext_dataloader, desc='Comparing texts'))
    print("Number of overlapping documents:", count)


if __name__ == '__main__':
    main()
