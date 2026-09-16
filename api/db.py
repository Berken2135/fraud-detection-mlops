"""PostgreSQL logging of every prediction the API serves.

Logging failures never fail a prediction request — availability of the
prediction itself matters more than the audit trail for this endpoint.
"""
from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, create_engine, func, text
from sqlalchemy.dialects.postgresql import JSONB

from src import config

# connect_timeout keeps a Postgres outage from turning into slow /predict
# responses - a refused/unreachable connection fails fast instead of hanging.
engine = create_engine(config.DATABASE_URL, pool_pre_ping=True, connect_args={"connect_timeout": 1})
metadata = MetaData()

predictions_table = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("features", JSONB, nullable=False),
    Column("prediction", Integer, nullable=False),
    Column("fraud_probability", Float, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("model_version", String, nullable=False),
)


def init_db() -> None:
    """Creates the predictions table if it doesn't exist yet.

    Swallows connection errors so the API can still start (and serve
    predictions) while Postgres is unavailable or still starting up.
    """
    try:
        metadata.create_all(engine)
    except Exception as exc:  # noqa: BLE001 - deliberate: DB unavailability must not block API startup
        print(f"WARNING: could not initialize database (table creation skipped): {exc}")


def log_predictions(rows: list[dict]) -> None:
    """Bulk-inserts prediction records. Swallows errors so a DB outage
    doesn't take down the /predict endpoint."""
    if not rows:
        return
    try:
        with engine.begin() as conn:
            conn.execute(predictions_table.insert(), rows)
    except Exception as exc:  # noqa: BLE001 - deliberate: logging must never break predictions
        print(f"WARNING: failed to log predictions to database: {exc}")


def is_connected() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
