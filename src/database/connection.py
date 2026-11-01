
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from config import DATABASE_URL
from src.database.models import Base


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)


@event.listens_for(engine, "connect")
def enable_sqlite_foreign_keys(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def create_database():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
