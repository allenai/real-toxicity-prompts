import json
from pathlib import Path
from typing import TypeVar, Iterable, List, Sequence, Union, Any

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

T = TypeVar('T')


def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch


def first(iterable):
    return next(iter(iterable), None)


def save_gpt2_training_data(corpus: Sequence[str], out_file: Union[str, Path], eos_token='<|endoftext|>'):
    with open(out_file, 'a') as f:
        for i, text in enumerate(tqdm(corpus, desc='Saving training data')):
            print(text, file=f, end='')
            if i != len(corpus) - 1:
                print(eos_token, file=f, end='')


def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file) as f:
        for line in f:
            yield json.loads(line)


def big_flat_jsonl_to_csv(jsonl_file, csv_file, chunksize=100_000, header=True):
    chunks = pd.read_json(jsonl_file, lines=True, chunksize=chunksize)

    for chunk in chunks:
        chunk.to_csv(csv_file, header=header, mode='a', index=False)
        header = False  # disable header after first rows are printed


def reorder_csv(csv_file_in, csv_file_out, columns, chunksize=100_000):
    chunks = pd.read_csv(csv_file_in, chunksize=chunksize)

    header = True
    for chunk in chunks:
        chunk.to_csv(csv_file_out, header=header, mode='a', index=False, columns=columns)
        header = False  # disable header after first rows are printed


def make_corpus_iter(corpus_dir: Path):
    files = sorted([file for file in corpus_dir.iterdir() if file.suffix == '.joblib'])

    i = 0
    for file in files:
        docs = joblib.load(file)

        # Load filenames or ids
        filenames_file = file.with_name(f'{file.stem}_filenames.txt')
        doc_ids = (
            filenames_file.read_text().split()
            if filenames_file.exists()
            else map(lambda idx: f'{file.stem}-{idx}', range(len(docs)))
        )

        print("Loading file:", file)
        for doc_id, doc in zip(doc_ids, docs):
            # Yield name and doc
            yield doc_id, doc
            i += 1
