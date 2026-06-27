"""
DB session management.

DEMO MODE: if DATABASE_URL isn't set to a real Postgres instance, this
falls back to a local SQLite file so the app runs with zero setup for a
hackathon demo. Geometry is stored as a GeoJSON string column rather than
a PostGIS geometry type in that fallback mode (SQLite has no PostGIS).

PRODUCTION PATH: set DATABASE_URL to a Postgres+PostGIS instance (Railway,
Supabase, etc. as noted in the original plan) and swap the geometry column
to Geometry(...) from GeoAlchemy2 - db/models.py notes exactly where.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import get_settings

settings = get_settings()
DB_URL = settings.DATABASE_URL

engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
