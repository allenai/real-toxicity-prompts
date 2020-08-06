import collections
import json
from pathlib import Path
from typing import Union, Tuple, List, Optional, Iterable

import click
from tqdm.auto import tqdm

from utils.constants import PERSPECTIVE_API_KEY
from utils.perspective_api import PerspectiveAPI
from utils.utils import first, load_jsonl, batchify

Document = Union[Path, str, Tuple[str, str]]


def prepare_batch(batch: List[Document], offset: int):
    if isinstance(batch[0], Path):
        return [(file.name, file.read_text()) for file in batch]
    elif isinstance(batch[0], str):
        request_ids = [str(i) for i in range(offset, offset + len(batch))]
        return list(zip(request_ids, batch))
    elif isinstance(batch[0], tuple):
        assert len(batch[0]) == 2 and isinstance(batch[0][0], str) and isinstance(batch[0][1], str)
        return batch
    else:
        raise RuntimeError(f'Unexpected element type ({type(batch[0])} in batch for: {batch[0]}')


def perspective_api_request(corpus: Union[Iterable[Document], str],
                            responses_file: Optional[Path] = None,
                            api_key: str = PERSPECTIVE_API_KEY,
                            requests_per_second: int = 25,
                            pbar: tqdm = None):
    if isinstance(corpus, str):
        corpus = [corpus]

    # Set up api
    perspective_api = PerspectiveAPI(api_key)

    # Set up progress bar
    total = len(corpus) if isinstance(corpus, collections.abc.Sequence) else None
    pbar = pbar or tqdm(total=total, dynamic_ncols=True)

    i = 0
    num_failures = 0
    for batch in batchify(corpus, requests_per_second):
        batch = prepare_batch(batch, offset=i)
        for request_id, (response, exception) in perspective_api.request(batch):
            assert response or exception
            response_dict = {
                'request_id': request_id,
                'success': not exception,
                'response': response,
                'error': str(exception) if exception else None
            }

            # Save response
            if responses_file:
                with responses_file.open('a') as f:
                    print(json.dumps(response_dict), file=f)

            if exception:
                num_failures += 1

        i += requests_per_second
        pbar.update(len(batch))
        pbar.set_description(f'Perspective API ({num_failures} failures)')


@click.command()
@click.option('--corpus', required=True)
@click.option('--responses_file', required=True, help='jsonl file containing responses')
@click.option('--api_key', default=PERSPECTIVE_API_KEY, help='Google API key with Perspective API access.')
@click.option('--requests_per_second', default=25, help='Requests per second to the Perspective API.')
def main(corpus, responses_file, api_key, requests_per_second):
    # Load corpus
    corpus = Path(corpus)
    if not corpus.exists():
        raise FileNotFoundError("Corpus path does not exist")
    elif corpus.is_file():
        # Read corpus into memory if it's a single file
        print('Reading file into memory...')
        if corpus.suffix == '.jsonl':
            print("File type: jsonl")
            corpus = list(load_jsonl(corpus))

            # Create a list of tuples (request_id, str)
            if isinstance(corpus[0], str):
                print("Adding request ids to input file")
                request_ids = map(str, range(len(corpus)))
                corpus = list(zip(request_ids, corpus))
            elif isinstance(corpus[0], dict):
                print("Using request ids from file")
                corpus = [first(line.items()) for line in corpus]
            else:
                raise RuntimeError('jsonl file not in expected format')
        else:
            print("File type: txt")
            corpus = corpus.open().readlines()
    elif corpus.is_dir():
        # FIXME: script no longer handles directories / lists of paths
        # Create list of files of it's a directory
        print('Loading list of files in corpus...')
        corpus = list(corpus.iterdir())

    responses_file = Path(responses_file)
    if responses_file.exists():
        raise FileExistsError("Responses file already exists")

    # Make requests
    print('Requesting from Perspective API')
    perspective_api_request(corpus, responses_file=responses_file, api_key=api_key,
                            requests_per_second=requests_per_second)


if __name__ == '__main__':
    main()
