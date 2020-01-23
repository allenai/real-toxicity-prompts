from pathlib import Path
from tqdm import trange
from itertools import islice, count
import json
from constants import PERSPECTIVE_API_RESPONSE_DIR, DATA_DIR
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Float, String

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


def main():
    database_path = DATA_DIR / 'perspective_api_responses.db'

    engine = create_engine(f'sqlite:///{database_path}', echo=False)

    # Create schema for Response
    Response.metadata.create_all(engine)

    # Start session
    Session.configure(bind=engine)
    session = Session()

    num_files = sum(1 for _ in PERSPECTIVE_API_RESPONSE_DIR.iterdir())
    response_dir_iter = PERSPECTIVE_API_RESPONSE_DIR.iterdir()

    try:
        # Create a sample response
        for i in trange(0, num_files, BATCH_SIZE):
            # Break if we will exceed our limit
            if LIMIT and i + BATCH_SIZE >= LIMIT:
                break

            # Get batch of responses
            responses = []
            for file in islice(response_dir_iter, BATCH_SIZE):
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
