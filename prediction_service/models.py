"""Database records for releases, immutable predictions, and publications."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class AdminUser(Base):
    __tablename__ = "admin_users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ModelRelease(Base):
    __tablename__ = "model_releases"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    algorithm_id: Mapped[str] = mapped_column(String(120), nullable=False)
    source_bundle_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    config_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    parent_snapshot_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    source: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    data_as_of: Mapped[str] = mapped_column(String(8), nullable=False)
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    features_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    features_path: Mapped[str] = mapped_column(Text, nullable=False)
    manifest_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PredictionLedger(Base):
    __tablename__ = "prediction_ledger"
    __table_args__ = (
        UniqueConstraint("release_id", "signal_date", name="uq_prediction_release_date"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    release_id: Mapped[str] = mapped_column(
        ForeignKey("model_releases.id"), nullable=False, index=True
    )
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("market_snapshots.id"), nullable=False, index=True
    )
    signal_date: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    target_date: Mapped[str | None] = mapped_column(String(8), nullable=True)
    base_close: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_return: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_label: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_close: Mapped[float] = mapped_column(Float, nullable=False)
    raw_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    calibrated_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    return_calibration_scale: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class OutcomeResolution(Base):
    __tablename__ = "outcome_resolutions"

    prediction_id: Mapped[str] = mapped_column(
        ForeignKey("prediction_ledger.id"), primary_key=True
    )
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("market_snapshots.id"), nullable=False
    )
    target_trade_date: Mapped[str] = mapped_column(String(8), nullable=False)
    actual_return: Mapped[float] = mapped_column(Float, nullable=False)
    actual_close: Mapped[float] = mapped_column(Float, nullable=False)
    correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    outcome_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Publication(Base):
    __tablename__ = "publications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("market_snapshots.id"), nullable=False, index=True
    )
    release_id: Mapped[str] = mapped_column(
        ForeignKey("model_releases.id"), nullable=False
    )
    public_csv_path: Mapped[str] = mapped_column(Text, nullable=False)
    public_csv_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RefreshJob(Base):
    __tablename__ = "refresh_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    trigger: Mapped[str] = mapped_column(String(30), nullable=False)
    requested_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    snapshot_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ShadowRun(Base):
    """A non-public model run evaluated against the same immutable snapshot."""

    __tablename__ = "shadow_runs"
    __table_args__ = (
        UniqueConstraint("snapshot_id", "release_id", name="uq_shadow_snapshot_release"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    engine: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    release_id: Mapped[str] = mapped_column(
        ForeignKey("model_releases.id"), nullable=False, index=True
    )
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("market_snapshots.id"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    requested_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    result_csv_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_csv_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    manifest_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    actor: Mapped[str | None] = mapped_column(String(120), nullable=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TradingDay(Base):
    __tablename__ = "trading_calendar"

    date: Mapped[str] = mapped_column(String(8), primary_key=True)
    is_open: Mapped[bool] = mapped_column(Boolean, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
