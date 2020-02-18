import json
from itertools import islice
from pathlib import Path
from typing import List, Union

import click
from sqlalchemy import Column, Float, String, Integer, ForeignKey
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from tqdm import trange, tqdm

BATCH_SIZE = 100000
LIMIT = None

Base = declarative_base()
Session = sessionmaker()


class Response(Base):
    __tablename__ = 'responses'
    span_scores = relationship('SpanScore')

    filename = Column(String, primary_key=True)
    insult = Column(Float)
    severe_toxicity = Column(Float)
    toxicity = Column(Float)
    profanity = Column(Float)
    sexually_explicit = Column(Float)
    flirtation = Column(Float)
    identity_attack = Column(Float)
    threat = Column(Float)

    def __repr__(self):
        return f"<Response<filename={self.filename}>"


class SpanScore(Base):
    __tablename__ = 'span_scores'

    filename = Column(String, ForeignKey('responses.filename'), primary_key=True)
    attribute = Column(String, primary_key=True)
    begin = Column(Integer, primary_key=True)
    end = Column(Integer, primary_key=True)
    value = Column(Float)

    def __repr__(self):
        return f"<SpanScore<filename={self.filename}, type={self.attribute}>"


def create_rows(response_file: Path) -> List[Union[Response, SpanScore]]:
    with response_file.open() as f:
        response_json = json.load(f)

    rows = []
    filename = response_file.name.split('.json')[0]
    attribute_scores = response_json['attributeScores'].items()

    summary_scores = {}
    for attribute, scores in attribute_scores:
        attribute = attribute.lower()

        # Save summary score
        assert scores['summaryScore']['type'] == 'PROBABILITY'
        summary_scores[attribute] = scores['summaryScore']['value']

        # Save span scores
        for span_score_dict in scores['spanScores']:
            assert span_score_dict['score']['type'] == 'PROBABILITY'
            span_score = SpanScore(filename=filename,
                                   attribute=attribute,
                                   begin=span_score_dict['begin'],
                                   end=span_score_dict['end'],
                                   value=span_score_dict['score']['value'])
            rows.append(span_score)

    response = Response(filename=filename, **summary_scores)
    rows.append(response)

    return rows


@click.command()
@click.argument('database_path')
@click.argument('responses_dir')
def main(database_path: str, responses_dir: str):
    responses_dir = Path(responses_dir)
    if not responses_dir.is_dir():
        raise click.FileError('Responses directory does not exist')

    engine = create_engine(f'sqlite:///{database_path}', echo=False)

    # Create schemas
    tqdm.write('Creating schemas...')
    Response.metadata.create_all(engine)
    SpanScore.metadata.create_all(engine)

    # Start session
    Session.configure(bind=engine)
    session = Session()

    tqdm.write('Counting responses... ', end='')
    num_responses = sum(1 for _ in responses_dir.iterdir())
    responses_dir_iter = responses_dir.iterdir()
    tqdm.write(f'total: {num_responses}')

    tqdm.write(f'Batch size: {BATCH_SIZE}, limit: {LIMIT}')
    tqdm.write(f'Reading responses into database...')
    try:
        # Add scores to our database
        for i in trange(0, num_responses, BATCH_SIZE):
            # Break if we will exceed our limit
            if LIMIT and i + BATCH_SIZE >= LIMIT:
                break

            # Get batch of rows
            rows = []
            for file in islice(responses_dir_iter, BATCH_SIZE):
                try:
                    rows.extend(create_rows(file))
                except Exception as e:
                    print("Error while creating response:", e)

            # If there are no more responses, break
            if not rows:
                break

            session.bulk_save_objects(rows)

        session.commit()
    except Exception as e:
        print(e)
        session.rollback()
    finally:
        session.close()


if __name__ == '__main__':
    main()
