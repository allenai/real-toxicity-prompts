import json

from utils.utils import load_jsonl, first
from utils.perspective_api import unpack_scores


def save_jsonl(perspective_file, corpus_file, out_file):
    with open(out_file, 'a') as f_out:
        for text, response in zip(load_jsonl(corpus_file), load_jsonl(perspective_file)):
            if not response['success']:
                continue

            id_, text = first(text.items())
            assert id_ == response['request_id']
            id_ = int(id_)

            summary_scores, span_scores = unpack_scores(response['response'])
            json.dump({'line': id_, 'text': text, **summary_scores}, f_out)
            f_out.write('\n')
