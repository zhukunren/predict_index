"""SQLAlchemy setup and SQLite-level immutability constraints."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, event, text
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings
from .models import Base


class Database:
    def __init__(self, settings: Settings) -> None:
        connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(
            settings.database_url,
            future=True,
            connect_args=connect_args,
        )
        if settings.database_url.startswith("sqlite"):
            event.listen(self.engine, "connect", self._enable_sqlite_foreign_keys)
        self._session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
            future=True,
        )

    @staticmethod
    def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)
        if self.engine.dialect.name == "sqlite":
            self._install_sqlite_immutability_triggers()

    def dispose(self) -> None:
        """Release pooled connections, which is required before Windows file cleanup."""

        self.engine.dispose()

    def _install_sqlite_immutability_triggers(self) -> None:
        statements = (
            """
            CREATE TRIGGER IF NOT EXISTS prediction_ledger_no_update
            BEFORE UPDATE ON prediction_ledger
            BEGIN
                SELECT RAISE(ABORT, 'prediction ledger is immutable');
            END;
            """,
            """
            CREATE TRIGGER IF NOT EXISTS prediction_ledger_no_delete
            BEFORE DELETE ON prediction_ledger
            BEGIN
                SELECT RAISE(ABORT, 'prediction ledger is immutable');
            END;
            """,
            """
            CREATE TRIGGER IF NOT EXISTS outcome_resolution_no_update
            BEFORE UPDATE ON outcome_resolutions
            BEGIN
                SELECT RAISE(ABORT, 'outcome resolution is immutable');
            END;
            """,
            """
            CREATE TRIGGER IF NOT EXISTS outcome_resolution_no_delete
            BEFORE DELETE ON outcome_resolutions
            BEGIN
                SELECT RAISE(ABORT, 'outcome resolution is immutable');
            END;
            """,
        )
        with self.engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
