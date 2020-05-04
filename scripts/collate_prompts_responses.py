import json

import click
import pandas as pd
from tqdm import tqdm

from scripts.create_db import unpack_scores
from utils.constants import PERSPECTIVE_API_ATTRIBUTES_LOWER
from utils.utils import batchify


def format_response(text: str, response: dict):
    if not response:
        response = {x: None for x in PERSPECTIVE_API_ATTRIBUTES_LOWER}
    return {'text': text, **response}


@click.command()
@click.option('--dataset_file', required=True)
@click.option('--generations_file', required=True)
@click.option('--perspective_file', required=True)
@click.option('--out_file', required=True)
def collate_generations(dataset_file: str, generations_file: str, perspective_file: str, out_file: str):
    dataset = pd.read_pickle(dataset_file)

    with open(generations_file) as f:
        generations = [json.loads(line) for line in tqdm(f)]
    num_gen_per_prompt = len(generations) // len(dataset)
    dataset['generation'] = list(batchify(generations, num_gen_per_prompt))

    scores = {}
    with open(perspective_file) as f:
        for line in tqdm(f):
            response = json.loads(line)
            col, idx = response['request_id'].split('-')
            if response['response']:
                summary_scores, _ = unpack_scores(response['response'])
            else:
                summary_scores = None

            if col not in scores:
                scores[col] = [summary_scores]
            else:
                assert int(idx) == len(scores[col])
                scores[col].append(summary_scores)

    for attr, scores_for_attr in scores.items():
        if attr == 'generation':
            scores_for_attr = list(batchify(scores_for_attr, num_gen_per_prompt))
        dataset[f'{attr}_response'] = scores_for_attr

    with open(out_file, 'a') as f:
        for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
            out = {
                'filename': row['filename'],
                'begin': row['begin'],
                'end': row['end'],
                'prompt': format_response(row['prompt'], row['prompt_response']),
                'continuation': format_response(row['continuation'], row['continuation_response']),
                'generations': [format_response(generation, response) for generation, response in
                                zip(row['generation'], row['generation_response'])]
            }
            print(json.dumps(out), file=f)


if __name__ == '__main__':
    collate_generations()
