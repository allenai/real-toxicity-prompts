import time
from typing import List, Union, Optional, Tuple, Dict, Any

from googleapiclient import discovery
from googleapiclient.discovery import Resource
from googleapiclient.errors import HttpError

from utils.constants import PERSPECTIVE_API_ATTRIBUTES, PERSPECTIVE_API_KEY


def unpack_scores(response_json: dict) -> Optional[Tuple[dict, dict]]:
    if not response_json:
        return None

    attribute_scores = response_json['attributeScores'].items()

    summary_scores = {}
    span_scores = {}
    for attribute, scores in attribute_scores:
        attribute = attribute.lower()

        # Save summary score
        assert scores['summaryScore']['type'] == 'PROBABILITY'
        summary_scores[attribute] = scores['summaryScore']['value']

        # Save span scores
        for span_score_dict in scores['spanScores']:
            assert span_score_dict['score']['type'] == 'PROBABILITY'
            span = (span_score_dict['begin'], span_score_dict['end'])
            span_scores.setdefault(span, {})[attribute] = span_score_dict['score']['value']

    return summary_scores, span_scores


class PerspectiveAPI:
    def __init__(self, api_key: str = PERSPECTIVE_API_KEY, rate_limit: int = 25):
        self.service = self._make_service(api_key)
        self.last_request_time = -1  # satisfies initial condition
        self.rate_limit = rate_limit
        self.next_uid = 0

    def request(self, texts: Union[str, List[str]]) -> List[Tuple[Optional[Dict[str, Any]], Optional[HttpError]]]:
        if isinstance(texts, str):
            texts = [texts]

        # Rate limit to 1 batch request per second
        assert len(texts) <= self.rate_limit
        time_since_last_request = time.time() - self.last_request_time
        if time_since_last_request < 1:
            time.sleep(1 - time_since_last_request)
        self.last_request_time = time.time()

        # Keys guaranteed in insertion order (Python 3.7+)
        responses = {str(uid): None for uid in range(self.next_uid, self.next_uid + len(texts))}
        self.next_uid += len(texts)

        def response_callback(request_id, response, exception):
            nonlocal responses
            responses[request_id] = (response, exception)

        # Make API request
        batch_request = self.service.new_batch_http_request()
        for uid, text in zip(responses.keys(), texts):
            batch_request.add(self._make_request(text, self.service), callback=response_callback, request_id=uid)
        batch_request.execute()

        # Unpack responses
        scores, errors = zip(*responses.values())
        scores = map(unpack_scores, scores)

        return list(zip(scores, errors))

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


def test_perspective_api():
    api = PerspectiveAPI()

    text_success = "Testing"
    text_error = 'x' * (20480 + 1)

    score_1, error_1 = api.request(text_success)[0]
    assert score_1 and not error_1

    score_2, error_2 = api.request(text_error)[0]
    assert not score_2 and isinstance(error_2, HttpError)

    multi_score, multi_error = zip(*api.request([text_success, text_error]))
    assert multi_score == (score_1, score_2)
    assert tuple(map(str, multi_error)) == tuple(map(str, (error_1, error_2)))
