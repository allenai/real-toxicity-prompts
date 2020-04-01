import csv
import json
import time
from pathlib import Path
from typing import List, Union, Optional, Iterable, Tuple, Dict

import click
from googleapiclient import discovery
from tqdm.auto import tqdm

from utils.constants import PERSPECTIVE_API_ATTRIBUTES, PERSPECTIVE_API_LEN_LIMIT, PERSPECTIVE_API_KEY


def perspective_service(api_key: str):
    # Generate API client object dynamically based on service name and version
    return discovery.build('commentanalyzer', 'v1alpha1', developerKey=api_key)


def perspective_request(text: str, service):
    analyze_request = {
        'comment': {'text': text},
        'requestedAttributes': {attr: {} for attr in PERSPECTIVE_API_ATTRIBUTES},
        'spanAnnotations': True,
    }
    return service.comments().analyze(body=analyze_request)


# def response_path(response_dir: Path, id_: str) -> Path:
#     return response_dir / (id_ + ".json")


def log_failure(failures_file: Optional[Path], id_: str, message: str = None):
    if not failures_file:
        return

    row = [id_, message or '']
    with failures_file.open('a') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def load_batches(docs: Union[List[Path], List[str]], batch_size: int) -> Iterable[Tuple[str, str]]:
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


def request_batch(batch: Iterable[Tuple[str, str]], service) -> Dict[str, Tuple[Optional[dict], Optional[str]]]:
    responses = {}

    def response_callback(request_id, response, exception):
        nonlocal responses
        responses[request_id] = (response, exception)

    # Create batch
    batch_request = service.new_batch_http_request()
    for request_id, text in batch:
        if len(text) > PERSPECTIVE_API_LEN_LIMIT:
            responses[request_id] = (
                None, f'{request_id}: document length was {len(text)}, limit is {PERSPECTIVE_API_LEN_LIMIT}'
            )
            continue

        batch_request.add(perspective_request(text, service), callback=response_callback, request_id=request_id)

    # Make API request
    batch_request.execute()
    return responses


def request(corpus: Union[List[Path], List[str]],
            responses_file: Optional[Path] = None,
            failures_file: Optional[Path] = None,
            api_key: str = PERSPECTIVE_API_KEY,
            requests_per_second: int = 25) -> List[dict]:
    # Generate API client object dynamically based on service name and version
    service = discovery.build('commentanalyzer', 'v1alpha1', developerKey=api_key)

    last_request = -1  # satisfies initial condition
    responses: List[Optional[dict]] = []

    pbar = tqdm(total=len(corpus))
    for batch in load_batches(corpus, requests_per_second):
        # Rate limit to 1 batch request per second
        now = time.time()
        request_time = now - last_request
        if request_time < 1:
            time.sleep(1 - request_time)
        last_request = now

        batch_responses = request_batch(batch, service)
        for request_id, _ in batch:
            response, exception = batch_responses[request_id]

            # Handle exception
            if exception:
                log_failure(failures_file, request_id, str(exception))
                assert not response

            # Save response
            responses.append(response)
            if responses_file:
                try:
                    with responses_file.open('a') as f:
                        print(json.dumps(response), file=f)
                except OSError as e:
                    log_failure(failures_file, request_id, f'Error saving file: {str(e)}')

        pbar.update(requests_per_second)

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
    request(corpus, responses_file=responses_file, failures_file=failures_file, api_key=api_key,
            requests_per_second=requests_per_second)


if __name__ == '__main__':
    main()
