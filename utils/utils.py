from pathlib import Path
from typing import TypeVar, Iterable, List, Sequence, Union, Any

import json
import numpy as np
import torch
from tqdm.auto import tqdm

from utils.constants import TEXTS_DIR

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


def load_text(filename: str) -> str:
    return (TEXTS_DIR / filename).read_text()


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file) as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(content: Any, file: Union[str, Path], mode='a'):
    with open(file, mode) as f:
        print(json.dumps(content), f)
