from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from utils.perspective_api import perspective_api_request
from joblib import load

from utils.constants import DATA_DIR, OUTPUT_DIR
from utils.utils import load_jsonl

SHARD_SIZE = 414_101  # HACK: all webtext shards are this size
WEBTEXT_SIZE = 8_282_020 - SHARD_SIZE  # Remove test shard


def corpus_iter(corpus_dir: Path, offset_i: int) -> Iterable[str]:
    """
    Yield a request id (simply the document index as a string) and a document string
    """
    files = sorted([file for file in corpus_dir.iterdir()
                    if file.suffix == '.joblib' and file.name != 'webtext_19.joblib'])  # Remove test shard
    print(files)

    total_i = 0
    for file in files:
        docs = None
        for shard_i in range(SHARD_SIZE):
            # Only start returning documents after reaching the offset
            if total_i >= offset_i:
                if not docs:  # Lazy load documents
                    tqdm.write(f'Loading {file}')
                    docs = load(file)
                    assert len(docs) == SHARD_SIZE
                yield str(total_i), docs[shard_i]

            total_i += 1


def main():
    webtext_dir = DATA_DIR / 'webtext_detokenized'
    out_file = OUTPUT_DIR / 'webtext_scores.jsonl'

    pbar = tqdm(total=WEBTEXT_SIZE, dynamic_ncols=True, desc='Perspective API')

    # Calculate offset
    offset = 0
    if out_file.exists():
        pbar.set_description('Loading cached files')
        # Check file for consistency and find offset
        for line in load_jsonl(out_file):
            assert int(line['request_id']) == offset
            offset += 1
            pbar.update(1)
    # POST-CONDITION: offset is now one more than the number of already-computed responses

    # Request scores
    perspective_api_request(corpus=corpus_iter(webtext_dir, offset),
                            responses_file=out_file,
                            pbar=pbar)


if __name__ == '__main__':
    success = False
    while not success:
        try:
            main()
            success = True
        except Exception as e:
            print("Caught unhandled exception:", e)
            print("Restarting...")
