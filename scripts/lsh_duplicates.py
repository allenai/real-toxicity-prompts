import json
import multiprocessing as mp
import os
import pickle
import time
from functools import partial
from pathlib import Path

import click
from datasketch import MinHash, LeanMinHash, MinHashLSH
from nltk import ngrams
from tqdm.auto import tqdm as _tqdm

from utils.constants import DATA_DIR
from utils.utils import make_corpus_iter

# Default settings for tqdm
tqdm = partial(_tqdm, dynamic_ncols=True)


def make_minhash_mapping(item, shingles: int, num_perm: int):
    doc_id, doc = item

    # Create MinHash
    shingles_set = set(ngrams(doc, shingles))
    m = MinHash(num_perm=num_perm)
    for s in shingles_set:
        s = ''.join(s).encode('utf8')
        m.update(s)

    # Convert to LeanMinHash
    m = LeanMinHash(m)

    return doc_id, m


def parallel_create_minhashes(corpus_iter, shingles: int, num_perm: int, n_jobs: int, chunksize=1000):
    make_minhash_mapping_ = partial(make_minhash_mapping, shingles=shingles, num_perm=num_perm)

    with mp.Pool(n_jobs) as pool:
        yield from pool.imap(make_minhash_mapping_, corpus_iter, chunksize=chunksize)


@click.command()
@click.option('--corpus', type=click.Choice(['webtext', 'openwebtext']), required=True)
@click.option('--mode', type=click.Choice(['lsh', 'query', 'minhash-only']), default='minhash-only')
@click.option('--lsh_file', default=None, type=str)
@click.option('--num_perm', default=128)
@click.option('--shingles', default=5)
@click.option('--jaccard', default=0.9)
@click.option('--n_jobs', default=os.cpu_count())
@click.argument('output_dir', type=str)
def main(corpus: str, mode: str, lsh_file: str, num_perm: int, shingles: int, jaccard: float, n_jobs: int,
         output_dir: str):
    assert mode == 'query' or not lsh_file

    print("Making output dir:", output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir()

    if corpus == 'webtext':
        corpus_len = 8_282_020
        corpus_iter = make_corpus_iter(DATA_DIR / 'webtext_detokenized')
    elif corpus == 'openwebtext':
        corpus_len = 8_013_769
        corpus_iter = make_corpus_iter(DATA_DIR / 'openwebtext_shards')
    else:
        raise RuntimeError

    print("Starting", n_jobs, "threads for minhashing...")
    minhash_iter = parallel_create_minhashes(corpus_iter, shingles=shingles, num_perm=num_perm, n_jobs=n_jobs)
    minhashes = {}

    print("Starting...")
    if mode == 'lsh':
        lsh = MinHashLSH(threshold=jaccard, num_perm=num_perm)
        with lsh.insertion_session() as session:
            for key, minhash in tqdm(minhash_iter, total=corpus_len, desc='Making MinHashLSH'):
                minhashes[key] = minhash
                session.insert(key, minhash, check_duplication=False)  # All keys are unique doc ids

        # Save LSH
        print("Saving LSH...")
        start = time.time()
        with open(output_dir / 'lsh.pkl', 'wb') as f:
            pickle.dump(lsh, f)
        print("Done saving LSH, time elapsed (sec):", time.time() - start)
    elif mode == 'query':
        print('Loading LSH:', lsh_file)
        start = time.time()
        with open(lsh_file, 'rb') as f:
            lsh = pickle.load(f)
        assert isinstance(lsh, MinHashLSH) and lsh.h == num_perm
        print("Done loading LSH, time elapsed (sec):", time.time() - start)

        duplicates_file = output_dir / 'duplicates.jsonl'
        print("Writing duplicates to", duplicates_file)
        with open(duplicates_file, 'a') as f:
            for key, minhash in tqdm(minhash_iter, total=corpus_len, desc='Querying'):
                minhashes[key] = minhash
                duplicates = lsh.query(minhash)
                if duplicates:
                    json.dump({key: duplicates}, f)
                    f.write('\n')
    elif mode == 'minhash-only':
        for key, minhash in tqdm(minhash_iter, total=corpus_len, desc='MinHashing'):
            minhashes[key] = minhash
    else:
        raise RuntimeError

    # Save MinHashes
    print("Saving MinHashes...")
    start = time.time()
    with open(output_dir / 'minhashes.pkl', 'wb') as f:
        pickle.dump(minhashes, f)
    print("Done saving MinHashes, time elapsed (sec):", time.time() - start)


if __name__ == '__main__':
    main()
