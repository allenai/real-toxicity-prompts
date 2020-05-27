import pickle
from functools import partial
import multiprocessing as mp
from pathlib import Path

from datasketch import MinHash, LeanMinHash, MinHashLSH
from joblib import load
from nltk import ngrams
from tqdm.auto import tqdm

from utils.constants import DATA_DIR, OUTPUT_DIR

NUM_PERM = 128
SHINGLES = 5
JACCARD = 0.9


def make_corpus_iter(corpus_dir: Path):
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
            yield doc_id, doc
            i += 1


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


def main():
    wt_len = 8_282_020
    wt_iter = make_corpus_iter(DATA_DIR / 'detokenized_webtext')
    mh_iter = parallel_create_minhashes(wt_iter, shingles=SHINGLES, num_perm=NUM_PERM, n_jobs=96)

    wt_minhashes = {}
    wt_lsh = MinHashLSH(threshold=JACCARD, num_perm=NUM_PERM)
    with wt_lsh.insertion_session() as session:
        for key, minhash in tqdm(mh_iter, total=wt_len):
            wt_minhashes[key] = minhash
            session.insert(key, minhash, check_duplication=False)  # All keys are unique doc ids

    with open(OUTPUT_DIR / 'webtext_minhashes.pkl', 'wb') as f:
        pickle.dump(wt_minhashes, f)

    with open(OUTPUT_DIR / 'webtext_lsh.pkl', 'wb') as f:
        pickle.dump(wt_lsh, f)


if __name__ == '__main__':
    main()
