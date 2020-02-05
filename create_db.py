import json
from itertools import islice
from pathlib import Path

import click
from sqlalchemy import Column, Float, String
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tqdm import trange

BATCH_SIZE = 100000
LIMIT = None

Base = declarative_base()
Session = sessionmaker()


class Response(Base):
    __tablename__ = 'responses'

    id = Column(String, primary_key=True)
    insult = Column(Float)
    severe_toxicity = Column(Float)
    toxicity = Column(Float)
    profanity = Column(Float)
    sexually_explicit = Column(Float)
    flirtation = Column(Float)
    identity_attack = Column(Float)
    threat = Column(Float)

    def __repr__(self):
        return f"<Response<id={self.id}>"


def create_response(response_file: Path) -> Response:
    with response_file.open() as f:
        response_json = json.load(f)

    id = response_file.name.split('.')[0]
    attributes = {k.lower(): v['summaryScore']['value'] for k, v in response_json['attributeScores'].items()}

    return Response(id=id, **attributes)

@click.command()
@click.argument('database_path')
@click.argument('responses_dir')
def main(database_path: str, responses_dir: str):
    engine = create_engine(f'sqlite:///{database_path}', echo=False)

    # Create schema for Response
    Response.metadata.create_all(engine)

    # Start session
    Session.configure(bind=engine)
    session = Session()

    responses_dir = Path(responses_dir)
    num_responses = sum(1 for _ in responses_dir.iterdir())
    responses_dir_iter = responses_dir.iterdir()

    try:
        # Create a sample response
        for i in trange(0, num_responses, BATCH_SIZE):
            # Break if we will exceed our limit
            if LIMIT and i + BATCH_SIZE >= LIMIT:
                break

            # Get batch of responses
            responses = []
            for file in islice(responses_dir_iter, BATCH_SIZE):
                # Skip chunked files
                if "chunk" in file.name:
                    continue

                try:
                    responses.append(create_response(file))
                except Exception as e:
                    print("Error while creating response:", e)

            # If there are no more responses, break
            if not responses:
                break

            session.bulk_save_objects(responses)

        session.commit()
    except Exception as e:
        print(e)
        session.rollback()
    finally:
        session.close()


if __name__ == '__main__':
    main()
