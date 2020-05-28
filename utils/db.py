from sqlalchemy import Column, Float, String, Integer, ForeignKey, create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from utils.constants import PERSPECTIVE_DB

Base = declarative_base()
Session = sessionmaker()


class Doc(Base):
    __tablename__ = 'docs'
    doc_scores = relationship('DocScore')
    span_scores = relationship('SpanScore')

    # Metadata
    id = Column(Integer, primary_key=True)
    location = Column(String)

    # Text
    text = Column(String)


class DocScore(Base):
    __tablename__ = 'doc_scores'

    # Metadata
    id = Column(Integer, ForeignKey('docs.id'), primary_key=True)

    # Attributes
    toxicity = Column(Float)
    severe_toxicity = Column(Float)
    identity_attack = Column(Float)
    insult = Column(Float)
    threat = Column(Float)
    profanity = Column(Float)
    sexually_explicit = Column(Float)
    flirtation = Column(Float)

    def __repr__(self):
        return f"<DocScore<id={self.id}>"


class SpanScore(Base):
    __tablename__ = 'span_scores'

    # Metadata
    id = Column(Integer, ForeignKey('docs.id'), primary_key=True)
    begin = Column(Integer, primary_key=True)
    end = Column(Integer, primary_key=True)

    # Attributes
    toxicity = Column(Float)
    severe_toxicity = Column(Float)
    identity_attack = Column(Float)
    insult = Column(Float)
    threat = Column(Float)
    profanity = Column(Float)
    sexually_explicit = Column(Float)
    flirtation = Column(Float)

    def __repr__(self):
        return f"<SpanScore<id={self.id}, begin={self.begin}, end={self.end}>"


def perspective_db_engine(**kwargs) -> Engine:
    return create_engine(f'sqlite:///{PERSPECTIVE_DB}', **kwargs)


def perspective_db_session(**kwargs) -> Session:
    engine = perspective_db_engine(**kwargs)
    session = Session(bind=engine)
    return session


def primary_key(table: Base):
    tuple(pk.name for pk in inspect(table).primary_key)
