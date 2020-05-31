from sqlalchemy import Column, Float, String, Integer, ForeignKey, create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from utils.constants import OPENWEBTEXT_DB

Base = declarative_base()
Session = sessionmaker()


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


def openwebtext_db_engine(**kwargs) -> Engine:
    return create_engine(f'sqlite:///{OPENWEBTEXT_DB}', **kwargs)


def openwebtext_db_session(**kwargs) -> Session:
    engine = openwebtext_db_engine(**kwargs)
    session = Session(bind=engine)
    return session


def primary_key(table: Base):
    tuple(pk.name for pk in inspect(table).primary_key)
