from pathlib import Path
from typing import List

from joblib import Memory, Parallel, delayed, dump
from lsh import cache, minhash
import numpy as np
from itertools import chain, islice

from utils.constants import DATA_DIR, OUTPUT_DIR
from utils.webtext import load_meta, split_docs

from tqdm.auto import tqdm

# Create joblib memory
mem = Memory(OUTPUT_DIR / 'cache' / 'webtext_overlap')

cached_meta = mem.cache(load_meta)

wt_meta = cached_meta(DATA_DIR / 'webtext')
wt_files = wt_meta[0]
owtc_meta = cached_meta(DATA_DIR / 'openwebtext_bpe')
owtc_files = owtc_meta[0]


def train(document_feed, char_ngram=3, seeds=100, bands=5, hashbytes=4, n_jobs=1):
    hasher = minhash.MinHasher(seeds=seeds, char_ngram=char_ngram, hashbytes=hashbytes)
    if seeds % bands != 0:
        raise ValueError('Seeds has to be a multiple of bands. {} % {} != 0'.format(seeds, bands))

    out = Parallel(n_jobs=n_jobs, verbose=1, backend='threading')(
        delayed(lambda doc_id, doc: (doc_id, hasher.fingerprint(doc)))(doc_id, doc)
        for doc_id, doc in document_feed
    )

    lshcache = cache.Cache(num_bands=bands, hasher=hasher)
    for doc_id, fingerprint in tqdm(out, 'Adding fingerprints to cache'):
        lshcache.add_fingerprint(fingerprint, doc_id=doc_id)

    return hasher, lshcache


def corpus_iter(files: List[Path], name: str):
    i = 0
    for file in files:
        print("Loading file:", file)
        shard = np.load(file)
        docs = split_docs(shard)
        for doc in docs:
            # Yield name and doc as 4-byte
            yield (i, name), doc.astype(np.int32).tobytes()
            i += 1


def run_lsh(char_ngram: int, seeds: int, bands: int, n_jobs: int, out_dir: Path):
    out_dir.mkdir()

    corpus = chain(
        islice(corpus_iter(wt_files, name='wt'), 100),
        islice(corpus_iter(owtc_files, name='owtc'), 100)
    )
    hasher, cache = train(document_feed=corpus,
                          char_ngram=char_ngram,
                          seeds=seeds,
                          bands=bands,
                          n_jobs=n_jobs)

    # Save fingerprints and doc ids
    doc_ids, fingerprints = zip(*cache.fingerprints.items())
    dump(doc_ids, out_dir / 'doc_ids.joblib')
    fingerprints_arr = np.stack(fingerprints)
    np.save(out_dir / 'fingerprints.npy', fingerprints_arr)

    # Save bins
    dump(cache.bins, out_dir / 'bins.joblib')

    # Save duplicates
    all_duplicates = cache.get_all_duplicates()
    dump(all_duplicates, out_dir / 'all_duplicates.joblib')


if __name__ == '__main__':
    NUM_JOBS = 96
    experiments = [
        {'char_ngram': 3, 'seeds': 100, 'bands': 5},
        {'char_ngram': 5, 'seeds': 100, 'bands': 5},
        {'char_ngram': 3, 'seeds': 100, 'bands': 5},
    ]

    for kwargs in experiments:
        out_dirname = '_'.join([f"{k}_{v}" for k, v in kwargs.items()])
        run_lsh(**kwargs, n_jobs=NUM_JOBS, out_dir=OUTPUT_DIR / out_dirname)
