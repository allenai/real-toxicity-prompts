import json
import time
import collections
from pathlib import Path
import socket
from typing import List, Union, Optional, Iterable, Tuple

import httplib2
from googleapiclient import discovery
from googleapiclient.discovery import Resource
from tqdm.auto import tqdm

from utils.constants import PERSPECTIVE_API_ATTRIBUTES, PERSPECTIVE_API_LEN_LIMIT, PERSPECTIVE_API_KEY
from utils.utils import batchify

Document = Union[Path, str, Tuple[str, str]]


class PerspectiveAPI:
    def __init__(self, api_key: str = PERSPECTIVE_API_KEY):
        self.service = self._make_service(api_key)
        self.last_request_time = -1  # satisfies initial condition

    def request(self, batch: List[Tuple[str, str]]) -> List[Tuple[str, Tuple[Optional[dict], Optional[str]]]]:
        # Rate limit to 1 batch request per second
        time_since_last_request = time.time() - self.last_request_time
        if time_since_last_request < 1:
            time.sleep(1 - time_since_last_request)
        self.last_request_time = time.time()

        # Keys guaranteed in insertion order
        responses = {request_id: None for request_id, _ in batch}

        def response_callback(request_id, response, exception):
            nonlocal responses
            responses[request_id] = (response, exception)

        # Create batch
        batch_request = self.service.new_batch_http_request()
        for request_id, text in batch:
            if len(text) > PERSPECTIVE_API_LEN_LIMIT:
                responses[request_id] = (
                    None, f'{request_id}: document length was {len(text)}, limit is {PERSPECTIVE_API_LEN_LIMIT}'
                )
                continue

            batch_request.add(self._make_request(text, self.service), callback=response_callback, request_id=request_id)

        # Make API request
        try:
            batch_request.execute()
        except (httplib2.HttpLib2Error, socket.timeout) as e:
            print("Error while executing request with ids:", *responses.keys())
            print(e)
            print("Returning errors for batch. Please retry this request again later.")
            responses = {request_id: (None, str(e)) for request_id, _ in batch}

        # Return list of tuples of (request_id, (response, exception)) in same order as request_id in input
        # (dict keys have insertion order guarantee in Python 3.7+)
        return list(responses.items())

    @staticmethod
    def _make_service(api_key: str):
        # Generate API client object dynamically based on service name and version
        return discovery.build('commentanalyzer', 'v1alpha1', developerKey=api_key)

    @staticmethod
    def _make_request(text: str, service: Resource):
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {attr: {} for attr in PERSPECTIVE_API_ATTRIBUTES},
            'spanAnnotations': True,
        }
        return service.comments().analyze(body=analyze_request)


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