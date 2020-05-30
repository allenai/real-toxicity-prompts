import json
from itertools import islice
from pathlib import Path

import click
from tqdm import tqdm

from scripts.data_preprocessing.create_db import unpack_scores
from utils.constants import PERSPECTIVE_API_ATTRIBUTES_LOWER
from utils.utils import load_jsonl

ATTRIBUTES_SET = set(PERSPECTIVE_API_ATTRIBUTES_LOWER)


def create_rows(response_json: dict):
    response, id_ = response_json['response'], response_json['request_id']

    summary_scores_raw, span_scores_raw = unpack_scores(response)
    doc_score = {'id': id_, **summary_scores_raw}

    span_scores = []
    for span, attribute_span_scores in span_scores_raw.items():
        # All attributes should have values for the same spans (line-by-line)
        assert ATTRIBUTES_SET == attribute_span_scores.keys()
        begin, end = span
        span_score = {'id': id_, 'begin': begin, 'end': end, **attribute_span_scores}
        span_scores.append(span_score)

    return doc_score, span_scores


@click.command()
@click.option('--responses_file', required=True)
@click.option('--total', default=None, type=int)
@click.argument('out_dir')
def main(responses_file: str, out_dir: str, total: int):
    out_dir = Path(out_dir)
    out_dir.mkdir()

    with open(out_dir / 'doc_scores.jsonl', 'w') as fd, open(out_dir / 'span_scores.jsonl', 'w') as fs:
        response_iter = load_jsonl(responses_file)
        for line in tqdm(islice(response_iter, total), total=total):
            if not line['success']:
                continue

            doc_score, span_scores = create_rows(line)

            print(json.dumps(doc_score), file=fd)
            print(*map(json.dumps, span_scores), file=fs, sep='\n')

    try:
        next(response_iter)
        print('File had more lines which were skipped')
    except StopIteration:
        pass


if __name__ == '__main__':
    main()
