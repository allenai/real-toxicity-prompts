from pathlib import Path
from typing import List

import numpy as np
from joblib import Memory, Parallel, delayed, dump
from transformers import GPT2Tokenizer

from utils.constants import DATA_DIR, OUTPUT_DIR
from utils.webtext import load_meta, split_docs

# Create joblib memory
mem = Memory(OUTPUT_DIR / 'cache' / 'webtext_overlap')
cached_meta = mem.cache(load_meta)


def _helper(file: Path, num_samples: int):
    print("Loading file:", file)
    shard = np.load(file)
    docs = split_docs(shard)
    random_idxs = np.random.choice(len(docs), size=num_samples, replace=False)
    return random_idxs, docs[random_idxs]


def sample_corpus(files: List[Path], num_samples: int):
    num_samples_per_doc = num_samples // len(files)  # FIXME: ASSUMES ALL SHARDS ARE SAME LEN
    shard_samples = Parallel(n_jobs=8)(
        delayed(_helper)(file, num_samples_per_doc) for file in files
    )

    idxs, docs = map(np.concatenate, zip(*shard_samples))
    return idxs, docs


def detokenize(docs: np.array, tokenizer: GPT2Tokenizer):
    detokenized_docs = Parallel(n_jobs=8, backend='threading')(
        delayed(tokenizer.decode)(doc, clean_up_tokenization_spaces=False)
        for doc in docs
    )

    return detokenized_docs


def main(out_dir: Path, num_samples: int):
    wt_files = cached_meta(DATA_DIR / 'webtext')[0]
    print("Sampling from corpus...")
    idxs, docs = sample_corpus(wt_files, num_samples=num_samples)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print("Detokenizing docs...")
    detokenized_docs = detokenize(docs, tokenizer=tokenizer)

    np.save(out_dir / 'idxs.npy', idxs)
    dump(detokenized_docs, out_dir / 'docs.joblib')


if __name__ == '__main__':
    out_dir = OUTPUT_DIR / 'webtext_100k'
    out_dir.mkdir()
    main(out_dir, num_samples=100_000)
