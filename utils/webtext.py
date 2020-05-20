from pathlib import Path

import dask
import dask.array as da
import numpy as np
from tqdm.auto import tqdm
from transformers import GPT2Config

# Get GPT-2 constants
config = GPT2Config.from_pretrained('gpt2')
EOS = config.eos_token_id
vocab_size = config.vocab_size


def load_meta(bpe_dir: Path):
    files = sorted([file for file in bpe_dir.iterdir() if file.suffix == '.npy'])
    meta = [(np.count_nonzero(array == EOS) - 1, array.dtype)
            for array
            in tqdm(map(np.load, files), total=len(files), desc='Loading meta')]
    shapes, dtypes = zip(*meta)
    return files, shapes, dtypes[0]


def split_docs(tokens: np.array) -> np.array:
    idx = np.nonzero(tokens == EOS)[0]
    docs = np.split(tokens, idx)
    docs = [doc[1:] for doc in docs if len(doc) > 1]
    return np.array(docs)


def delayed_corpus(meta):
    files, shapes, dtype = meta

    # Create delayed arrays
    delayed_load = dask.delayed(lambda f: split_docs(np.load(f)))
    delayed_arrays = list(map(delayed_load, files))

    # Concatenate arrays
    corpus = da.concatenate([da.from_delayed(array, shape=(shape,), dtype=np.ndarray)
                             for array, shape in zip(delayed_arrays, shapes)])

    return corpus
