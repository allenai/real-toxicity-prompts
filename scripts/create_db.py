import json
from itertools import chain
from pathlib import Path
from typing import List, Union, Tuple

import click
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from tqdm import tqdm

from utils.constants import PERSPECTIVE_API_ATTRIBUTES_LOWER
from utils.db import Response, SpanScore

ATTRIBUTES_SET = set(PERSPECTIVE_API_ATTRIBUTES_LOWER)
BATCH_SIZE = 10_000
NUM_WORKERS = 8
LIMIT = None

Session = sessionmaker()


def unpack_scores(response_json: dict) -> Tuple[dict, dict]:
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


def create_rows(response_file: Path) -> List[Union[Response, SpanScore]]:
    with response_file.open() as f:
        response_json = json.load(f)

    rows = []
    filename = response_file.name.split('.json')[0]
    summary_scores, span_scores = unpack_scores(response_json)

    response = Response(filename=filename, **summary_scores)
    rows.append(response)

    for span, attribute_span_scores in span_scores.items():
        # All attributes should have values for the same spans (line-by-line)
        assert ATTRIBUTES_SET == attribute_span_scores.keys()
        begin, end = span
        span_score = SpanScore(filename=filename, begin=begin, end=end, **attribute_span_scores)
        rows.append(span_score)

    return rows


class PerspectiveDataset(Dataset):
    def __init__(self, data_dir: Path):
        super().__init__()
        self.files = list(data_dir.iterdir())

    def __getitem__(self, idx):
        response_file = self.files[idx]
        try:
            rows = create_rows(response_file)
        except:
            return []
        return rows

    def __len__(self):
        return len(self.files)


@click.command()
@click.argument('database_path')
@click.argument('responses_dir')
def main(database_path: str, responses_dir: str):
    responses_dir = Path(responses_dir)
    if not responses_dir.is_dir():
        raise click.ClickException('Responses directory does not exist')

    database_path = Path(database_path)
    if database_path.exists():
        raise click.ClickException('Database already exists')

    engine = create_engine(f'sqlite:///{database_path}', echo=False)

    # Create schemas
    tqdm.write('Creating schemas...')
    Response.metadata.create_all(engine)
    SpanScore.metadata.create_all(engine)

    # Start session
    Session.configure(bind=engine)
    session = Session()

    # Create dataloader
    tqdm.write("Loading list of files...")
    dataset = PerspectiveDataset(responses_dir)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=False, sampler=sampler, num_workers=NUM_WORKERS,
                            collate_fn=lambda x: x)

    tqdm.write(f'Number of files: {len(dataset)}, batch size: {BATCH_SIZE}, limit: {LIMIT}')

    tqdm.write(f'Reading responses into database...')
    try:
        # Add scores to our database
        for i, rows_batch in enumerate(tqdm(dataloader)):
            # Break if we will exceed our limit
            if LIMIT and i + BATCH_SIZE >= LIMIT:
                break

            rows = chain.from_iterable(rows_batch)
            session.bulk_save_objects(rows)
            session.commit()
            session.expunge_all()

    except Exception as e:
        print(e)
        session.rollback()
    finally:
        session.close()


if __name__ == '__main__':
    main()