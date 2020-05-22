from pathlib import Path

import numpy as np
from joblib import Memory, Parallel, delayed, dump
from tqdm import tqdm
from transformers import GPT2Tokenizer

from utils.constants import DATA_DIR, OUTPUT_DIR
from utils.webtext import load_meta, split_docs

# Create joblib memory
mem = Memory(OUTPUT_DIR / 'cache' / 'webtext_overlap')
cached_meta = mem.cache(load_meta)


def corpus(file: Path):
    print("Loading file:", file)
    shard = np.load(file)
    docs = split_docs(shard)
    return docs


def main(out_dir: Path):
    wt_files = cached_meta(DATA_DIR / 'webtext')[0]
    print('Files:', wt_files)

    print("Detokenizing docs...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    file: Path
    for file in tqdm(wt_files):
        detokenized_docs = Parallel(n_jobs=8, backend='threading')(
            delayed(tokenizer.decode)(doc, clean_up_tokenization_spaces=False)
            for doc in tqdm(corpus(file))
        )
        dump(detokenized_docs, out_dir / f'{file.stem}.joblib')


if __name__ == '__main__':
    out_dir = OUTPUT_DIR / 'webtext_detokenized'
    out_dir.mkdir()
    main(out_dir)
