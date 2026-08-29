import os
import logging
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

from sqlalchemy.orm import sessionmaker, declarative_base
import datetime

Base = declarative_base()

class ChatSession(Base):
    __tablename__ = 'chat_sessions'
    
    session_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    memory_type = Column(String, default="buffer")   # buffer, window, summary
    memory_k = Column(Integer, default=4)             # used for window memory
    summary = Column(String, default="")              # used for summary memory
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # --- New columns for backend selection ---
    # memory_backend controls WHERE state is stored:
    #   "sql"            -> SQLChatMessageHistory (langchain_community) — default
    #   "in_memory"      -> InMemoryChatMessageHistory (langchain_core)
    #   "memory_saver"   -> MemorySaver / older alias of InMemorySaver (LangGraph)
    #   "postgres_saver" -> PostgresSaver (LangGraph checkpoint-postgres)
    memory_backend = Column(String, default="sql")

    # thread_id is the key used by LangGraph checkpointers.
    # Defaults to session_id but stored explicitly for clarity.
    thread_id = Column(String, default=None)

def get_connection_string() -> str:
    """
    Returns the PostgreSQL connection string.
    Raises an error if the URL is missing or is not a PostgreSQL URL.
    The database must already exist — no auto-creation is performed.
    """
    
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Please define a valid postgresql:// URL in your .env file."
        )
    if not DATABASE_URL.startswith("postgresql"):
        raise RuntimeError(
            f"Invalid DATABASE_URL: '{DATABASE_URL}'. Only PostgreSQL is supported. "
            "Please set a valid postgresql:// URL in your .env file."
        )
    return DATABASE_URL

def get_engine():
    """Returns a SQLAlchemy engine for the PostgreSQL database."""
    return create_engine(get_connection_string())

def init_db():
    """Initializes database tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized.")

SessionLocal = sessionmaker(autocommit=False, autoflush=False)

def get_db():
    """FastAPI dependency for database sessions."""
    engine = get_engine()
    SessionLocal.configure(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()