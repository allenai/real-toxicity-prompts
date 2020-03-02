from sqlalchemy import Column, Float, String, Integer, ForeignKey, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from utils.constants import PERSPECTIVE_DB

Base = declarative_base()


class Response(Base):
    __tablename__ = 'responses'
    span_scores = relationship('SpanScore')

    # Metadata
    filename = Column(String, primary_key=True)

    # Attributes
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

    # Metadata
    filename = Column(String, ForeignKey('responses.filename'), primary_key=True)
    begin = Column(Integer, primary_key=True)
    end = Column(Integer, primary_key=True)

    # Attributes
    insult = Column(Float)
    severe_toxicity = Column(Float)
    toxicity = Column(Float)
    profanity = Column(Float)
    sexually_explicit = Column(Float)
    flirtation = Column(Float)
    identity_attack = Column(Float)
    threat = Column(Float)

    def __repr__(self):
        return f"<SpanScore<filename={self.filename}, begin={self.begin}, end={self.end}>"


def perspective_db_engine() -> Engine:
    return create_engine(f'sqlite:///{PERSPECTIVE_DB}', echo=False)
