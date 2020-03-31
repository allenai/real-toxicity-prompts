import csv
import json
import time
from pathlib import Path
from typing import List, Union, Optional, Iterable, Tuple

import click
from googleapiclient import discovery
from tqdm.auto import tqdm

from utils.constants import PERSPECTIVE_API_ATTRIBUTES, PERSPECTIVE_API_LEN_LIMIT, PERSPECTIVE_API_KEY


def perspective_service(api_key: str):
    # Generate API client object dynamically based on service name and version
    return discovery.build('commentanalyzer', 'v1alpha1', developerKey=api_key)


_SERVICE = perspective_service(PERSPECTIVE_API_KEY) if PERSPECTIVE_API_KEY else None
_last_request = 0


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


def request_batch(batch: Union[Iterable[str], Iterable[Tuple[str, str]]],
                  service=_SERVICE) -> Iterable[Tuple[str, Optional[dict], Optional[str]]]:
    # Rate limit to 1 batch request per second
    global _last_request
    now = time.time()
    request_time = now - _last_request
    if request_time < 1:
        time.sleep(1 - request_time)
    _last_request = now

    request_ids: List[str] = []
    responses: List[Optional[dict]] = []
    exceptions: List[Optional[str]] = []

    def response_callback(request_id, response, exception):
        nonlocal request_ids
        nonlocal responses
        nonlocal exceptions

        request_ids.append(request_id)
        responses.append(response)
        exceptions.append(exception)

    # Create batch request
    batch_request = service.new_batch_http_request()
    for request_id, text in batch:
        if len(text) > PERSPECTIVE_API_LEN_LIMIT:
            responses.append(None)
            exceptions.append(
                f'{request_id}: document length was {len(text)}, limit is {PERSPECTIVE_API_LEN_LIMIT}'
            )
            continue

        batch_request.add(perspective_request(text, service), callback=response_callback, request_id=request_id)

    # Make request
    try:
        batch_request.execute()
    except Exception as e:
        for request_id, _ in batch:
            responses.append(None)
            exceptions.append(str(e))

    return zip(request_ids, responses, exceptions)


def request(corpus: Union[List[Path], List[str]],
            api_key: str,
            requests_per_second: int,
            responses_file: Optional[Path] = None,
            failures_file: Optional[Path] = None) -> Optional[List[dict]]:
    # Generate API client object dynamically based on service name and version
    service = discovery.build('commentanalyzer', 'v1alpha1', developerKey=api_key)

    num_failures = 0
    responses: List[dict] = []

    pbar = tqdm(total=len(corpus))
    for batch in load_batches(corpus, requests_per_second):
        for request_id, response, exception in request_batch(batch, service):
            # Handle exception
            if exception:
                if failures_file:
                    log_failure(failures_file, request_id, str(exception))
                num_failures += 1
                continue

            # Save response
            if responses_file:
                try:
                    with response_path(responses_file, request_id).open('w') as f:
                        print(json.dumps(response), file=f)
                except OSError as e:
                    tqdm.write(f'Error while saving response for {request_id}: {e}')
            else:
                responses.append(response)

        # Update progress bar
        pbar.set_description(f"Failures: {num_failures}")
        pbar.update(requests_per_second)

    # Return list of responses if no directory is specified
    if not responses_file:
        return responses


@click.command()
@click.option('--corpus', required=True)
@click.option('--responses_file', required=True, help='jsonl file containing responses')
@click.option('--api_key', required=True, help='Google API key with Perspective API access.')
@click.option('--failures_file', default=None, help='CSV file to log API failures to.')
@click.option('--requests_per_second', default=25, help='Requests per second to the Perspective API.')
def main(corpus, responses_file, failures_file, api_key, requests_per_second):
    # Load corpus
    corpus = Path(corpus)
    if not corpus.exists():
        raise click.ClickException("Corpus path does not exist")
    elif corpus.is_file():
        # Read corpus into memory if it's a single file
        corpus = corpus.open().readlines()
    elif corpus.is_dir():
        # Create list of files of it's a directory
        corpus = list(corpus.iterdir())

    responses_file = Path(responses_file)
    if responses_file.exists():
        raise click.ClickException("Responses file already exists")

    failures_file = Path(failures_file)
    if failures_file.exists():
        raise click.ClickException("Failures file already exists")

    # Make requests
    tqdm.write("Requesting from Perspective API...")
    request(corpus, api_key, requests_per_second, responses_file=responses_file, failures_file=failures_file)


if __name__ == '__main__':
    main()
