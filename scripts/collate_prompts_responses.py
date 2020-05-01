import json

import click
from tqdm import tqdm

from scripts.create_db import unpack_scores
from utils.constants import PERSPECTIVE_API_ATTRIBUTES_LOWER
from utils.utils import batchify


def format_responses(response: dict, text: str):
    if not response:
        response = {x: None for x in PERSPECTIVE_API_ATTRIBUTES_LOWER}
    return {'text': text, **response}


@click.command()
@click.option('--dataset_file', required=True)
@click.option('--generations_file', required=True)
@click.option('--perspective_file', required=True)
@click.option('--out_file', required=True)
def collate_generations(dataset_file: str, generations_file: str, perspective_file: str, out_file: str):
    with open(dataset_file) as f:
        dataset = [json.loads(line) for line in tqdm(f)]

    scores = []
    with open(perspective_file) as f:
        for line in tqdm(f):
            response = json.loads(line)
            if response['response']:
                summary_scores, _ = unpack_scores(response['response'])
            else:
                summary_scores = None
            scores.append(summary_scores)

    with open(generations_file) as f:
        generations = [json.loads(line) for line in tqdm(f)]
    gens_per_prompt = len(generations) // len(dataset)

    with open(out_file, 'a') as f:
        rows_iter = zip(dataset, batchify(generations, gens_per_prompt), batchify(scores, gens_per_prompt))
        for data, generations_batch, scores_batch, in tqdm(rows_iter, total=len(dataset)):
            out = {
                'filename': data['filename'],
                'begin': data['begin'],
                'end': data['end'],
                'prompt': data['prompt'],
                'continuation': data['continuation'],
                'generations': [format_responses(scores, gen) for scores, gen in zip(scores_batch, generations_batch)]
            }
            print(json.dumps(out), file=f)


if __name__ == '__main__':
    collate_generations()
