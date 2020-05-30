from pathlib import Path

import click

from utils.constants import PERSPECTIVE_API_KEY
from utils.perspective_api import perspective_api_request
from utils.utils import first, load_jsonl


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
