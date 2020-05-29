import os
import multiprocessing as mp
import pickle
from functools import partial
from pathlib import Path

import click
from datasketch import MinHash, LeanMinHash, MinHashLSH
from nltk import ngrams
from tqdm.auto import tqdm

from utils.constants import DATA_DIR
from utils.utils import make_corpus_iter


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
@click.option('--corpus', type=click.Choice(['webtext', 'openwebtext']))
@click.option('--lsh/--no-lsh', default=False)
@click.option('--num_perm', default=128)
@click.option('--shingles', default=5)
@click.option('--jaccard', default=0.9)
@click.option('--n_jobs', default=os.cpu_count())
@click.argument('output_dir', type=str)
def main(corpus: str, lsh: bool, num_perm: int, shingles: int, jaccard: float, n_jobs: int, output_dir: str):
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

    if lsh:
        print("Starting LSH session...")
        lsh = MinHashLSH(threshold=jaccard, num_perm=num_perm)
        with lsh.insertion_session() as session:
            for key, minhash in tqdm(minhash_iter, total=corpus_len):
                minhashes[key] = minhash
                session.insert(key, minhash, check_duplication=False)  # All keys are unique doc ids

        # Save LSH
        print("Saving LSH...")
        with open(output_dir / 'lsh.pkl', 'wb') as f:
            pickle.dump(lsh, f)
        print("Done saving LSH")
    else:
        print("Starting minhashing...")
        for key, minhash in tqdm(minhash_iter, total=corpus_len):
            minhashes[key] = minhash

    # Save MinHashes
    print("Saving MinHashes...")
    with open(output_dir / 'minhashes.pkl', 'wb') as f:
        pickle.dump(minhashes, f)
    print("Done saving MinHashes")


if __name__ == '__main__':
    main()
