import csv
import json
import time
from pathlib import Path
from typing import List, Union, Optional

import click
from googleapiclient import discovery
from tqdm import tqdm

from utils.constants import PERSPECTIVE_API_ATTRIBUTES, PERSPECTIVE_API_LEN_LIMIT


def perspective_request(text: str, service):
    analyze_request = {
        'comment': {'text': text},
        'requestedAttributes': {attr: {} for attr in PERSPECTIVE_API_ATTRIBUTES},
        'spanAnnotations': True,
    }
    return service.comments().analyze(body=analyze_request)


def response_path(response_dir: Path, id_: str) -> Path:
    return response_dir / (id_ + ".json")


def log_failure(failures_file: Optional[Path], id_: str, message: str = None):
    if not failures_file:
        return

    row = [id_, message or '']
    with failures_file.open('a') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def load_batches(docs: Union[List[Path], List[str]], batch_size: int):
    assert batch_size > 0

    batch = []
    for i, doc in enumerate(docs):
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        # Add to current batch
        if isinstance(doc, Path):
            request_id = doc.name
            text = doc.read_text()
        else:
            request_id = str(i)
            text = doc

        batch.append((request_id, text))

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch


def request(corpus: Union[List[Path], List[str]], responses_dir: Path, failures_file: Optional[Path], service,
            batch_size: int):
    num_failures = 0

    def response_callback(text_filename, response, exception):
        nonlocal num_failures

        if exception:
            log_failure(failures_file, text_filename, exception)
            num_failures += 1
        else:
            try:
                with response_path(responses_dir, text_filename).open('w') as f:
                    json.dump(response, f)
            except OSError as e:
                tqdm.write(f'Error while saving response for {text_filename}: {e}')

    pbar = tqdm(total=len(corpus))
    for batch in load_batches(corpus, batch_size):
        start_time = time.time()

        # Create batch request
        batch_request = service.new_batch_http_request()
        for request_id, text in batch:
            if len(text) > PERSPECTIVE_API_LEN_LIMIT:
                log_failure(failures_file, request_id,
                            f'Document length was {len(text)}, limit is {PERSPECTIVE_API_LEN_LIMIT}')
                num_failures += 1
                continue

            batch_request.add(perspective_request(text, service), callback=response_callback, request_id=request_id)

        # Make request
        batch_request.execute()

        # Update progress bar
        pbar.set_description(f"Failures: {num_failures}")
        pbar.update(batch_size)

        # Rate limit to 1 batch request per second
        request_time = time.time() - start_time
        if request_time < 1:
            time.sleep(1 - request_time)


@click.command()
@click.argument('corpus', required=True)
@click.option('responses_dir', required=True)
@click.option('--api_key', required=True, help='Google API key with Perspective API access.')
@click.option('--failures_file', default=None, help='CSV file to log API failures to.')
@click.option('--requests_per_second', default=25, help='Requests per second to the Perspective API.')
def main(corpus, responses_dir, failures_file, api_key, requests_per_second):
    # Load corpus
    corpus = Path(corpus)
    if not corpus.exists():
        raise click.FileError("Corpus path does not exist")
    elif corpus.is_file():
        # Read corpus into memory if it's a single file
        corpus = corpus.open().readlines()
    elif corpus.is_dir():
        # Create list of files of it's a directory
        corpus = list(corpus.iterdir())

    responses_dir = Path(responses_dir)
    if not responses_dir.exists():
        responses_dir.mkdir(parents=True, exist_ok=False)

    failures_file = Path(failures_file)
    if failures_file.exists():
        raise click.FileError("Failures file already exists")

    # Generate API client object dynamically based on service name and version
    service = discovery.build('commentanalyzer', 'v1alpha1', developerKey=api_key)

    # Make requests
    tqdm.write("Requesting from Perspective API...")
    request(corpus, responses_dir, failures_file, service, requests_per_second)


if __name__ == '__main__':
    main()
