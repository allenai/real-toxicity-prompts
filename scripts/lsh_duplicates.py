import multiprocessing as mp
from itertools import chain
from pathlib import Path

import numpy as np
from joblib import dump, load
from lsh import cache, minhash
from tqdm.auto import tqdm

from utils.constants import DATA_DIR, OUTPUT_DIR

TOTAL = 8_282_020 + 8_013_769


class Fingerprinter:
    def __init__(self, hasher):
        self.hasher = hasher

    def fingerprint(self, doc_id, doc):
        return doc_id, self.hasher.fingerprint(doc)


def train(document_feed, char_ngram: int, seeds: int, bands: int, hashbytes: int = 4, n_jobs: int = 1):
    hasher = minhash.MinHasher(seeds=seeds, char_ngram=char_ngram, hashbytes=hashbytes)
    if seeds % bands != 0:
        raise ValueError('Seeds has to be a multiple of bands. {} % {} != 0'.format(seeds, bands))

    lshcache = cache.Cache(num_bands=bands, hasher=hasher)
    fingerprinter = Fingerprinter(hasher)
    with mp.Pool(processes=n_jobs) as pool:
        for doc_id, fingerprint in tqdm(pool.starmap(fingerprinter.fingerprint, document_feed, chunksize=10_000),
                                        desc='Hashing',
                                        dynamic_ncols=True,
                                        total=TOTAL):
            lshcache.add_fingerprint(fingerprint, doc_id=doc_id)

    return hasher, lshcache


def corpus_iter(corpus_dir: Path, name: str):
    files = sorted([file for file in corpus_dir.iterdir() if file.suffix == '.joblib'])

    i = 0
    for file in files:
        docs = load(file)

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
            yield (doc_id, name), doc
            i += 1


def run_lsh(char_ngram: int, seeds: int, bands: int, n_jobs: int, out_dir: Path, save_bins: bool = False):
    corpus = chain(
        corpus_iter(DATA_DIR / 'detokenized_webtext', name='wt'),
        corpus_iter(DATA_DIR / 'openwebtext_shards', name='owtc'),
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

    if save_bins:
        dump(cache.bins, out_dir / 'bins.joblib')

    # Save duplicates
    all_duplicates = cache.get_all_duplicates()
    dump(all_duplicates, out_dir / 'all_duplicates.joblib')


def main():
    NUM_JOBS = 96
    experiments = [
        {'char_ngram': 5, 'seeds': 100, 'bands': 10},
    ]

    for kwargs in experiments:
        out_dirname = '_'.join([f"{k}_{v}" for k, v in kwargs.items()]) + '_str'
        out_dir = OUTPUT_DIR / 'lsh_duplicates' / out_dirname
        if out_dir.exists():
            continue
        out_dir.mkdir(parents=True)

        print('*' * 100)
        print('Starting', out_dirname)
        try:
            run_lsh(**kwargs, n_jobs=NUM_JOBS, out_dir=out_dir)
        except Exception as e:
            print("Exception during experiment ", out_dirname)
            print(e)


if __name__ == '__main__':
    main()
