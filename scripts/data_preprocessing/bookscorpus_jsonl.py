import json

from utils.utils import load_jsonl
from utils.perspective_api import unpack_scores


def save_jsonl(perspective_file, corpus_file, out_file):
    with open(corpus_file) as f_in, open(out_file, 'a') as f_out:
        for text, response in zip(f_in, load_jsonl(perspective_file)):
            if not response['success']:
                continue

            id_ = int(response['request_id'])
            summary_scores, span_scores = unpack_scores(response['response'])
            json.dump({'line': id_, 'text': text, **summary_scores}, f_out)
            f_out.write('\n')
