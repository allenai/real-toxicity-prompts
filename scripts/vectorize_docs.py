from pathlib import Path
from typing import List

import numpy as np
from joblib import Memory, Parallel, delayed
from scipy.sparse import vstack, save_npz
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm

from utils.constants import DATA_DIR, OUTPUT_DIR

# Create joblib memory
mem = Memory(OUTPUT_DIR / 'cache' / 'webtext_overlap')

EOS = 50256
vocab_size = EOS + 1


def bpe_files(bpe_dir: Path) -> List[Path]:
    return [file for file in bpe_dir.iterdir() if file.suffix == '.npy']


def load_meta(bpe_dir: Path):
    files = bpe_files(bpe_dir)
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


def _load_shard(file: Path):
    print("Loading shard:", file.stem)
    shard = np.load(file)
    print("Splitting documents:", file.stem)
    docs = split_docs(shard)
    return docs


def load_corpus_vectors(files: List[Path], n_jobs: int):
    with Parallel(n_jobs=n_jobs) as parallel:
        print("Loading shards...")
        shards = parallel(
            delayed(_load_shard)(file) for file in files
        )

        print("CountVectorizing...")
        identity = lambda x: x
        vectorizer = CountVectorizer(vocabulary=range(vocab_size), analyzer=identity)

        vectorized_docs = parallel(
            delayed(vectorizer.transform)(shard) for shard in shards
        )

    return vectorized_docs


def main():
    # Cache calls to load_meta
    load_meta_cached = mem.cache(load_meta)

    # Load metadata
    wt_dir = DATA_DIR / 'webtext'
    wt_meta = load_meta_cached(wt_dir)
    wt_files = wt_meta[0]

    vectorized_docs = load_corpus_vectors(wt_files, n_jobs=20)
    vectorized_docs = vstack(vectorized_docs)
    save_npz(OUTPUT_DIR / 'wt_vecs', vectorized_docs)


if __name__ == '__main__':
    main()
