from pathlib import Path
import json
from constants import PERSPECTIVE_API_RESPONSE_DIR, DATA_DIR
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Float, String

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


if __name__ == '__main__':
    database_path = DATA_DIR / 'test.db'

    engine = create_engine(f'sqlite:///{database_path}', echo=True)

    # Create schema for Response
    Response.metadata.create_all(engine)

    # Start session
    Session.configure(bind=engine)
    session = Session()

    # Create a sample response
    sample_response_file = next(PERSPECTIVE_API_RESPONSE_DIR.iterdir())
    sample_response = create_response(sample_response_file)

    try:
        session.add(sample_response)
        session.commit()
    except:
        session.rollback()
    finally:
        session.close()
