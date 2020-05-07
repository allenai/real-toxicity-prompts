from pathlib import Path
from typing import List, Optional

import click
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

from datasets.openwebtext_dataset import openwebtext_dataloader


def write_corpus(filenames: List[str], out_file: Path, eos_token='<|endoftext|>'):
    dataloader = openwebtext_dataloader(filenames)
    with out_file.open('a') as f:
        for i, (filename, text) in enumerate(tqdm(dataloader)):
            print(text, file=f, end='')
            if i != len(dataloader) - 1:
                print(eos_token, file=f, end='')


@click.command()
@click.option('--csv_path', required=True, type=str)
@click.option('--limit', default=None, type=int)
@click.option('--test_size', default=0.01, type=float)
def create_finetune_data(csv_path: str, limit: Optional[int], test_size=0.01):
    csv_path = Path(csv_path)
    assert csv_path.exists() and csv_path.is_file()

    experiment_name = csv_path.stem
    out_dir = csv_path.parent / experiment_name
    out_dir.mkdir()

    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit).copy()

    train, test = train_test_split(df, test_size=test_size)

    train.to_csv(out_dir / f'{experiment_name}_train.csv', index=False)
    test.to_csv(out_dir / f'{experiment_name}_test.csv', index=False)

    write_corpus(train.filename.tolist(), out_dir / f'{experiment_name}_train.txt')
    write_corpus(test.filename.tolist(), out_dir / f'{experiment_name}_test.txt')


if __name__ == '__main__':
    create_finetune_data()
