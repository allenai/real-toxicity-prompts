from pathlib import Path
from typing import List, Union

import click
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from utils.constants import PERSPECTIVE_API_ATTRIBUTES_LOWER
from utils.perspective_api import unpack_scores
from utils.webtext_db import DocScore, SpanScore
from utils.utils import load_jsonl

ATTRIBUTES_SET = set(PERSPECTIVE_API_ATTRIBUTES_LOWER)

Session = sessionmaker()


def create_rows(response_json: dict, id_: str) -> List[Union[DocScore, SpanScore]]:
    rows = []
    summary_scores, span_scores = unpack_scores(response_json)

    response = DocScore(id=id_, **summary_scores)
    rows.append(response)

    for span, attribute_span_scores in span_scores.items():
        # All attributes should have values for the same spans (line-by-line)
        assert ATTRIBUTES_SET == attribute_span_scores.keys()
        begin, end = span
        span_score = SpanScore(id=id_, begin=begin, end=end, **attribute_span_scores)
        rows.append(span_score)

    return rows


@click.command()
@click.option('--responses_file', required=True)
@click.option('--total', default=None, type=int)
@click.argument('database_file')
def main(responses_file: str, database_file: str, total: int):
    responses_file = Path(responses_file)
    if not responses_file.is_file():
        raise click.ClickException('Responses file does not exist')

    database_file = Path(database_file)
    if database_file.exists():
        raise click.ClickException('Database already exists')

    engine = create_engine(f'sqlite:///{database_file}', echo=False)

    # Create schemas
    tqdm.write('Creating schemas...')
    DocScore.metadata.create_all(engine)
    SpanScore.metadata.create_all(engine)

    # Start session
    Session.configure(bind=engine)
    session = Session()

    tqdm.write(f'Reading responses into database...')
    try:
        # Add scores to our database
        for line in tqdm(load_jsonl(responses_file), total=total):
            if not line['success']:
                continue

            rows = create_rows(line['response'], line['request_id'])

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
