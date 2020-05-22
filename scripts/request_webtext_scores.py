from pathlib import Path
from typing import List, Iterable

from tqdm import tqdm

from scripts.perspective_api_request import perspective_api_request
from joblib import load

from utils.constants import DATA_DIR, OUTPUT_DIR
from utils.utils import load_jsonl

WEBTEXT_SIZE = 8_282_020
SHARD_SIZE = 414_101  # HACK: all webtext shards are this size


def corpus_iter(corpus_dir: Path, offset_i: int) -> Iterable[str]:
    """
    Yield a request id (simply the document index as a string) and a document string
    """
    files = sorted([file for file in corpus_dir.iterdir() if file.suffix == '.joblib'])

    total_i = 0
    for file in files:
        docs = None
        for shard_i in range(SHARD_SIZE):
            # Only start returning documents after reaching the offset
            if total_i >= offset_i:
                if not docs:
                    tqdm.write('Loading', file)
                docs = docs or load(file)  # Lazy-load docs
                yield str(total_i), docs[shard_i]

            total_i += 1


def main():
    webtext_dir = DATA_DIR / 'webtext_detokenized'
    out_file = OUTPUT_DIR / 'webtext_scores.jsonl'

    # Calculate offset
    offset = 0
    if out_file.exists():
        # Check file for consistency and find offset
        for line in load_jsonl(out_file):
            assert int(line['request_id']) == offset
            offset += 1
    # POST-CONDITION: offset is now one more than the number of already-computed responses

    # Request scores
    perspective_api_request(corpus=corpus_iter(webtext_dir, offset),
                            responses_file=out_file,
                            pbar=tqdm(total=WEBTEXT_SIZE - offset, dynamic_ncols=True))


if __name__ == '__main__':
    main()
