"""Immutable ledger, refresh orchestration, and verified CSV generation."""

from __future__ import annotations

import json
import io
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import numpy as np
from sqlalchemy import select, update

from .archive import (
    ArchiveArtifact,
    frame_to_csv_bytes,
    sha256_bytes,
    sha256_file,
    write_daily_archive,
    read_features,
)
from .calendar import MarketCalendar, SHANGHAI
from .metrics import statistics, clean_records
from .config import Settings
from .database import Database
from .engine import (
    HistoricalMarketDataDriftError,
    calculate_results,
    calculate_results_with_diagnostics,
    calculate_bilstm_shadow_results,
    canonicalize_features,
    data_as_of,
    feature_close_by_date,
    merge_append_only_features,
    model_configuration,
    outcome_payload,
    payload_fingerprint,
    prediction_payload,
    public_csv_bytes,
    public_frame,
    release_id,
    shadow_settings,
    target_dates_by_signal,
)
from .models import (
    AdminUser,
    AuditLog,
    MarketSnapshot,
    ModelRelease,
    OutcomeResolution,
    PredictionLedger,
    Publication,
    RefreshJob,
    ShadowRun,
    utcnow,
)
from .security import ensure_bootstrap_admin


class ServiceNotReadyError(RuntimeError):
    pass


class ModelReleaseMismatchError(RuntimeError):
    pass


class PredictionDriftError(RuntimeError):
    pass


class NoNewMarketDataError(RuntimeError):
    """Raised when a refresh has no new trading-day data to publish."""


@dataclass(frozen=True, slots=True)
class CandidatePrediction:
    signal_date: str
    prediction_id: str
    base_close: float
    payload: dict[str, Any]
    fingerprint: str
    outcome: dict[str, Any] | None
    outcome_fingerprint: str | None
    is_new: bool
    target_date: str | None
    generated_at: datetime
    origin: str


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    snapshot_id: str
    release_id: str
    data_as_of: str
    result: pd.DataFrame
    public_frame: pd.DataFrame
    csv_bytes: bytes
    csv_sha256: str
    archive: ArchiveArtifact | None = None


CalculationFunction = Callable[[pd.DataFrame, Settings], pd.DataFrame]
DiagnosticsCalculationFunction = Callable[
    [pd.DataFrame, Settings, Path], tuple[pd.DataFrame, pd.DataFrame]
]


class PredictionService:
    """Owns the append-only data/forecast ledger and public publication state."""

    def __init__(
        self,
        settings: Settings,
        *,
        database: Database | None = None,
        calculation_function: CalculationFunction | None = None,
        diagnostics_calculation_function: DiagnosticsCalculationFunction | None = None,
        shadow_calculation_function: CalculationFunction | None = None,
    ) -> None:
        self.settings = settings
        self.database = database or Database(settings)
        self.calculation_function = calculation_function or calculate_results
        self.diagnostics_calculation_function = (
            diagnostics_calculation_function
            if diagnostics_calculation_function is not None
            else (
                calculate_results_with_diagnostics
                if calculation_function is None
                else None
            )
        )
        self.shadow_calculation_function = (
            shadow_calculation_function or calculate_bilstm_shadow_results
        )
        self._operation_lock = threading.RLock()
        self._calendar_lock = threading.Lock()
        self._trading_day_cache: dict[str, bool] = {}
        self.calendar = MarketCalendar(self.database)

    def initialize(self, *, bootstrap: bool = True) -> None:
        self.database.initialize()
        with self.database.session() as session:
            ensure_bootstrap_admin(session, self.settings)
            self._ensure_release(session)

        if bootstrap and self._active_publication() is None and self.settings.local_feature_path.exists():
            features = read_features(self.settings.local_feature_path)
            self.publish_from_features(
                features,
                source="bootstrap",
                raw_frames=None,
                actor="system",
            )

    def _ensure_release_for_settings(self, session, settings: Settings) -> ModelRelease:
        identifier, config, config_sha256, source_sha256 = release_id(settings)
        release = session.get(ModelRelease, identifier)
        if release is not None:
            return release
        release = ModelRelease(
            id=identifier,
            algorithm_id=str(config["algorithm_id"]),
            source_bundle_sha256=source_sha256,
            config_sha256=config_sha256,
            config_json=json.dumps(config, ensure_ascii=False, sort_keys=True),
        )
        session.add(release)
        session.flush()
        return release

    def _ensure_release(self, session) -> ModelRelease:
        """Ensure the default release exists without selecting a shadow release."""

        publication = session.scalar(
            select(Publication)
            .where(Publication.is_active.is_(True))
            .order_by(Publication.published_at.desc())
        )
        if publication is not None:
            release = session.get(ModelRelease, publication.release_id)
            if release is not None:
                return release
        return self._ensure_release_for_settings(session, self.settings)

    def _active_release(self, session) -> ModelRelease:
        return self._ensure_release(session)

    def _assert_runtime_release(self, release: ModelRelease) -> None:
        if self.settings.model_bundle_dir:
            from .model_registry import ModelBundle
            from .forecast_models import PRODUCTION_KEY
            bundle = ModelBundle(self.settings.model_bundle_dir)
            if release.id != bundle.manifest["releases"][PRODUCTION_KEY]["release_id"]:
                raise ModelReleaseMismatchError("配置中的模型组合尚未正式激活，请使用模型组合发布命令。")
            return
        identifier, _, _, _ = release_id(self.settings)
        if release.id != identifier:
            raise ModelReleaseMismatchError(
                "当前预测代码或参数与已发布模型版本不一致；"
                "请创建新的模型发布版本，不能重写历史预测。"
            )

    def release_upgrade_required(self) -> bool:
        _, _, release = self._load_active_context()
        if self.settings.model_bundle_dir:
            try:
                self._assert_runtime_release(release)
                return False
            except (ModelReleaseMismatchError, ValueError, OSError):
                return True
        identifier, _, _, _ = release_id(self.settings)
        return release.id != identifier

    def promote_compatible_release(self, *, actor: str | None) -> VerifiedArtifact | None:
        """Create a new immutable release only after exact historical parity.

        A source/configuration change never updates the old release. It first
        recomputes the active archive against the old ledger. Only a complete
        match can be promoted into a new release with its own prediction rows.
        """

        if self.settings.model_bundle_dir:
            raise ModelReleaseMismatchError("固定模型组合须通过独立版本发布命令更新，不能使用基线兼容发布入口。")
        with self._operation_lock:
            publication, snapshot, old_release = self._load_active_context()
            identifier, config, config_sha256, source_sha256 = release_id(self.settings)
            if old_release.id == identifier:
                return None
            self._assert_archive_integrity(snapshot, publication)
            features = read_features(snapshot.features_path)
            # Validate current code output against the frozen old-release ledger.
            self._candidate_plan(
                features=features,
                release=old_release,
                snapshot_id=snapshot.id,
                allow_new=False,
            )
            with self.database.session() as session:
                existing = session.get(ModelRelease, identifier)
                if existing is None:
                    session.add(
                        ModelRelease(
                            id=identifier,
                            algorithm_id=str(config["algorithm_id"]),
                            source_bundle_sha256=source_sha256,
                            config_sha256=config_sha256,
                            config_json=json.dumps(config, ensure_ascii=False, sort_keys=True),
                        )
                    )
                session.add(
                    AuditLog(
                        actor=actor,
                        action="release_promoted_after_parity_check",
                        detail=json.dumps(
                            {
                                "from_release": old_release.id,
                                "to_release": identifier,
                                "snapshot_id": snapshot.id,
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
            return self.publish_from_features(
                features,
                source="release_promotion",
                raw_frames=None,
                actor=actor,
                release_id_override=identifier,
            )

    def _active_publication(self) -> Publication | None:
        with self.database.session() as session:
            return session.scalar(
                select(Publication)
                .where(Publication.is_active.is_(True))
                .order_by(Publication.published_at.desc())
            )

    def _load_active_context(self) -> tuple[Publication, MarketSnapshot, ModelRelease]:
        with self.database.session() as session:
            publication = session.scalar(
                select(Publication)
                .where(Publication.is_active.is_(True))
                .order_by(Publication.published_at.desc())
            )
            if publication is None:
                raise ServiceNotReadyError("尚无可对外发布的预测快照。")
            snapshot = session.get(MarketSnapshot, publication.snapshot_id)
            release = session.get(ModelRelease, publication.release_id)
            if snapshot is None or release is None:
                raise ServiceNotReadyError("当前发布快照的元数据不完整。")
            return publication, snapshot, release

    @staticmethod
    def _signal_date(row: pd.Series) -> str:
        return f"{int(row['trade_date']):08d}"

    def _existing_predictions(self, session, release_id: str, signal_dates: list[str]) -> dict[str, PredictionLedger]:
        if not signal_dates:
            return {}
        rows = session.scalars(
            select(PredictionLedger).where(
                PredictionLedger.release_id == release_id,
                PredictionLedger.signal_date.in_(signal_dates),
            )
        ).all()
        return {row.signal_date: row for row in rows}

    def _existing_outcomes(self, session, prediction_ids: list[str]) -> dict[str, OutcomeResolution]:
        if not prediction_ids:
            return {}
        rows = session.scalars(
            select(OutcomeResolution).where(OutcomeResolution.prediction_id.in_(prediction_ids))
        ).all()
        return {row.prediction_id: row for row in rows}

    def _candidate_plan(
        self,
        *,
        features: pd.DataFrame,
        release: ModelRelease,
        snapshot_id: str,
        allow_new: bool,
        result: pd.DataFrame | None = None,
        inherit_provenance: bool = True,
    ) -> tuple[pd.DataFrame, list[CandidatePrediction]]:
        if result is None:
            result = self.calculation_function(features, self.settings)
        close_by_date = feature_close_by_date(features)
        targets = target_dates_by_signal(features)
        signal_dates = [self._signal_date(row) for _, row in result.iterrows()]

        with self.database.session() as session:
            existing = self._existing_predictions(session, release.id, signal_dates)
            outcomes = self._existing_outcomes(
                session, [row.id for row in existing.values()]
            )
            max_existing_date = session.scalar(
                select(PredictionLedger.signal_date)
                .where(PredictionLedger.release_id == release.id)
                .order_by(PredictionLedger.signal_date.desc())
                .limit(1)
            )
            previous_rows = session.scalars(
                select(PredictionLedger)
                .where(PredictionLedger.signal_date.in_(signal_dates))
                .order_by(PredictionLedger.created_at)
            ).all()
        provenance = {}
        for previous in previous_rows:
            provenance.setdefault((previous.signal_date, previous.prediction_fingerprint), previous)

        plan: list[CandidatePrediction] = []
        for _, row in result.iterrows():
            signal_date = self._signal_date(row)
            base_close = close_by_date.get(signal_date)
            if base_close is None:
                raise PredictionDriftError(f"信号日 {signal_date} 缺少收盘价。")
            payload = prediction_payload(row, base_close=base_close)
            fingerprint = payload_fingerprint(payload)
            stored = existing.get(signal_date)
            if stored is None:
                if not allow_new and max_existing_date is not None:
                    raise PredictionDriftError(f"重算结果出现未登记预测：{signal_date}")
                if max_existing_date is not None and signal_date <= max_existing_date:
                    raise PredictionDriftError(
                        f"预测账本存在历史缺口：{signal_date} 不能作为新预测插入。"
                    )
                prediction_id = str(uuid.uuid4())
                is_new = True
                stored_outcome = None
            else:
                if stored.prediction_fingerprint != fingerprint:
                    raise PredictionDriftError(
                        f"预测漂移：信号日 {signal_date} 的预测字段与账本不一致。"
                    )
                prediction_id = stored.id
                is_new = False
                stored_outcome = outcomes.get(stored.id)

            target_date = targets.get(signal_date)
            if target_date is None and pd.notna(row.get("real_pct_change")):
                raise PredictionDriftError(f"信号日 {signal_date} 无法定位实际结果交易日。")
            outcome = outcome_payload(
                row,
                target_date=target_date or "",
                base_close=base_close,
            )
            outcome_fingerprint = payload_fingerprint(outcome) if outcome else None
            if stored_outcome is not None:
                if outcome is None or stored_outcome.outcome_fingerprint != outcome_fingerprint:
                    raise PredictionDriftError(
                        f"实际结果漂移：信号日 {signal_date} 与既有结算不一致。"
                    )

            previous = stored or (
                provenance.get((signal_date, fingerprint)) if inherit_provenance else None
            )
            generated_at = previous.created_at if previous is not None else utcnow()
            if generated_at.tzinfo is None:
                generated_at = generated_at.replace(tzinfo=timezone.utc)
            forecast_target = target_date or self.calendar.next_session(signal_date)
            origin = "backfill"
            if forecast_target:
                opening = datetime.strptime(forecast_target, "%Y%m%d").replace(hour=9, minute=30, tzinfo=SHANGHAI)
                if generated_at < opening:
                    origin = "live"
            else:
                origin = "unknown"
            plan.append(
                CandidatePrediction(
                    signal_date=signal_date,
                    prediction_id=prediction_id,
                    base_close=base_close,
                    payload=payload,
                    fingerprint=fingerprint,
                    outcome=outcome,
                    outcome_fingerprint=outcome_fingerprint,
                    is_new=is_new,
                    target_date=forecast_target,
                    generated_at=generated_at,
                    origin=origin,
                )
            )
        return result, plan

    def _persist_prediction_plan(
        self,
        session,
        *,
        plan: list[CandidatePrediction],
        release: ModelRelease,
        snapshot_id: str,
    ) -> None:
        """Insert immutable forecasts and one-time outcomes for one release."""

        for item in plan:
            if not item.is_new:
                continue
            payload = item.payload
            session.add(
                PredictionLedger(
                    id=item.prediction_id,
                    release_id=release.id,
                    snapshot_id=snapshot_id,
                    signal_date=item.signal_date,
                    target_date=item.target_date,
                    base_close=item.base_close,
                    predicted_return=float(payload["predicted_return"]),
                    predicted_label=int(payload["predicted_label"]),
                    predicted_close=float(payload["predicted_close"]),
                    raw_confidence=float(payload["raw_confidence"]),
                    calibrated_confidence=float(payload["calibrated_confidence"]),
                    return_calibration_scale=(
                        float(payload["return_calibration_scale"])
                        if payload["return_calibration_scale"] is not None
                        else None
                    ),
                    prediction_fingerprint=item.fingerprint,
                    payload_json=json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    created_at=item.generated_at,
                )
            )

        session.flush()
        existing_outcomes = self._existing_outcomes(
            session, [item.prediction_id for item in plan]
        )
        for item in plan:
            if item.outcome is None or item.prediction_id in existing_outcomes:
                continue
            outcome = item.outcome
            session.add(
                OutcomeResolution(
                    prediction_id=item.prediction_id,
                    snapshot_id=snapshot_id,
                    target_trade_date=str(outcome["target_trade_date"]),
                    actual_return=float(outcome["actual_return"]),
                    actual_close=float(outcome["actual_close"]),
                    correct=bool(outcome["correct"]),
                    outcome_fingerprint=str(item.outcome_fingerprint),
                    payload_json=json.dumps(outcome, ensure_ascii=False, sort_keys=True),
                )
            )

    def _render_artifact(
        self,
        *,
        result: pd.DataFrame,
        plan: list[CandidatePrediction],
        snapshot_id: str,
        release_id: str,
        as_of: str,
        metadata: bool = True,
        stored_metadata: pd.DataFrame | None = None,
    ) -> VerifiedArtifact:
        prediction_ids = {item.signal_date: item.prediction_id for item in plan}
        output_frame = public_frame(
            result,
            snapshot_id=snapshot_id,
            release_id=release_id,
            data_as_of_date=as_of,
            prediction_ids=prediction_ids,
        )
        if metadata:
            by_date = {item.signal_date: item for item in plan}
            dates = output_frame["信号日期"].astype(str)
            output_frame["预测目标交易日"] = dates.map(lambda day: by_date[day].target_date)
            output_frame["记录来源"] = dates.map(lambda day: by_date[day].origin)
            output_frame["预测生成时间"] = dates.map(lambda day: by_date[day].generated_at.astimezone(SHANGHAI).isoformat(timespec="seconds"))
            if stored_metadata is not None:
                frozen = stored_metadata.set_index(stored_metadata["信号日期"].astype(str))
                for column in ("预测目标交易日", "记录来源", "预测生成时间"):
                    output_frame[column] = dates.map(frozen[column])
        output_bytes = public_csv_bytes(output_frame)
        return VerifiedArtifact(
            snapshot_id=snapshot_id,
            release_id=release_id,
            data_as_of=as_of,
            result=result,
            public_frame=output_frame,
            csv_bytes=output_bytes,
            csv_sha256=sha256_bytes(output_bytes),
        )

    def _settle_outstanding(self, session, features: pd.DataFrame, release_id: str, snapshot_id: str) -> None:
        """Resolve old pending rows even when downtime exceeds the export window."""
        session.flush()
        pending = session.scalars(
            select(PredictionLedger)
            .outerjoin(OutcomeResolution, OutcomeResolution.prediction_id == PredictionLedger.id)
            .where(PredictionLedger.release_id == release_id, OutcomeResolution.prediction_id.is_(None))
        ).all()
        targets = target_dates_by_signal(features)
        closes = feature_close_by_date(features)
        for prediction in pending:
            target = targets.get(prediction.signal_date)
            if target is None:
                continue
            actual = closes[target] / closes[prediction.signal_date] - 1.0
            payload = outcome_payload(pd.Series({"real_pct_change": actual, "correct": bool(prediction.predicted_label == int(actual > 0))}), target_date=target, base_close=prediction.base_close)
            session.add(OutcomeResolution(
                prediction_id=prediction.id, snapshot_id=snapshot_id, target_trade_date=target,
                actual_return=float(payload["actual_return"]), actual_close=float(payload["actual_close"]),
                correct=bool(payload["correct"]), outcome_fingerprint=payload_fingerprint(payload),
                payload_json=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            ))

    def recompute_active(self) -> VerifiedArtifact:
        """Recompute from the archived snapshot and fail closed on any drift."""

        if self.settings.model_bundle_dir:
            from .portfolio import recompute_portfolio
            return recompute_portfolio(self)
        with self._operation_lock:
            publication, snapshot, release = self._load_active_context()
            self._assert_runtime_release(release)
            self._assert_archive_integrity(snapshot, publication)
            features_path = Path(snapshot.features_path)
            if not features_path.exists():
                raise ServiceNotReadyError("当前快照的特征归档文件不存在。")
            features = read_features(features_path)
            result, plan = self._candidate_plan(
                features=features,
                release=release,
                snapshot_id=snapshot.id,
                allow_new=False,
            )
            artifact = self._render_artifact(
                result=result,
                plan=plan,
                snapshot_id=snapshot.id,
                release_id=release.id,
                as_of=snapshot.data_as_of,
                metadata=json.loads(Path(snapshot.manifest_path).read_text(encoding="utf-8")).get("format_version", 1) >= 2,
                stored_metadata=pd.read_csv(publication.public_csv_path, encoding="utf-8-sig", dtype={"预测目标交易日": "string"}),
            )
            if artifact.csv_sha256 != publication.public_csv_sha256:
                raise PredictionDriftError(
                    "重算 CSV 哈希与已发布快照不一致，已拒绝返回候选结果。"
                )
            return artifact

    def _assert_archive_integrity(
        self,
        snapshot: MarketSnapshot,
        publication: Publication | None,
    ) -> None:
        features_path = Path(snapshot.features_path)
        manifest_path = Path(snapshot.manifest_path)
        if not features_path.exists() or not manifest_path.exists():
            raise PredictionDriftError("归档快照文件缺失。")
        if sha256_file(features_path) != snapshot.features_sha256:
            raise PredictionDriftError("归档特征文件哈希不匹配。")

        self._assert_manifest_integrity(manifest_path)

        if publication is not None:
            public_path = Path(publication.public_csv_path)
            if (
                not public_path.exists()
                or sha256_file(public_path) != publication.public_csv_sha256
            ):
                raise PredictionDriftError("已发布 CSV 哈希不匹配。")

    def read_published(self, context: tuple[Publication, MarketSnapshot, ModelRelease] | None = None) -> VerifiedArtifact:
        """Serve the immutable publication independently of the installed model."""
        publication, snapshot, release = context or self._load_active_context()
        try:
            self._assert_archive_integrity(snapshot, publication)
            content = Path(publication.public_csv_path).read_bytes()
            if sha256_bytes(content) != publication.public_csv_sha256:
                raise PredictionDriftError("已发布 CSV 哈希不匹配。")
            frame = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", float_precision="round_trip")
            if len(frame) != publication.row_count or frame.empty:
                raise PredictionDriftError("已发布 CSV 记录数不匹配。")
        except (OSError, ValueError, KeyError) as exc:
            raise PredictionDriftError("发布文件无法读取或格式无效。") from exc
        return VerifiedArtifact(snapshot.id, release.id, snapshot.data_as_of, pd.DataFrame(), frame, content, publication.public_csv_sha256)

    def health(self, now: datetime | None = None, *, artifact: VerifiedArtifact | None = None) -> dict[str, Any]:
        try:
            artifact = artifact or self.read_published()
        except (ServiceNotReadyError, PredictionDriftError) as exc:
            return {"status": "unavailable", "message": str(exc), "csv_available": False}
        expected = self.calendar.expected_as_of(now or utcnow(), self.settings.scheduled_refresh_hour, self.settings.scheduled_refresh_minute)
        status = "unknown" if expected is None else ("stale" if artifact.data_as_of < expected else "ok")
        return {
            "status": status, "csv_available": True,
            "data_as_of": artifact.data_as_of, "expected_as_of": expected,
            "snapshot_id": artifact.snapshot_id,
            "message": {"ok": "发布数据已更新", "stale": "发布数据落后于应有交易日", "unknown": "交易日历尚未确认"}[status],
        }

    def recover_interrupted_jobs(self) -> None:
        """Called only after the server acquires its exclusive instance lock."""
        with self.database.session() as session:
            for model in (RefreshJob, ShadowRun):
                for job in session.scalars(select(model).where(model.status.in_(("queued", "running")))).all():
                    job.status = "interrupted"
                    job.finished_at = utcnow()
                    job.message = "上次服务退出时任务中断，可重新提交。"

    @staticmethod
    def _assert_manifest_integrity(manifest_path: Path) -> None:
        if not manifest_path.exists():
            raise PredictionDriftError("归档清单文件缺失。")

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise PredictionDriftError("归档清单无法读取。") from exc
        expected_files = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(expected_files, dict):
            raise PredictionDriftError("归档清单缺少文件哈希。")
        archive_root = manifest_path.parent
        for relative_path, expected_hash in expected_files.items():
            candidate = archive_root / str(relative_path)
            if not candidate.resolve().is_relative_to(archive_root.resolve()) or not candidate.is_file() or sha256_file(candidate) != expected_hash:
                raise PredictionDriftError(f"归档文件哈希不匹配：{relative_path}")

    def _current_canonical_features(self) -> tuple[pd.DataFrame | None, MarketSnapshot | None]:
        publication = self._active_publication()
        if publication is None:
            return None, None
        with self.database.session() as session:
            snapshot = session.get(MarketSnapshot, publication.snapshot_id)
        if snapshot is None:
            raise ServiceNotReadyError("当前快照不存在。")
        path = Path(snapshot.features_path)
        if not path.exists():
            raise ServiceNotReadyError("当前快照特征文件不存在。")
        self._assert_archive_integrity(snapshot, publication)
        return read_features(path), snapshot

    def publish_from_features(
        self,
        feature_frame: pd.DataFrame,
        *,
        source: str,
        raw_frames: dict[str, pd.DataFrame] | None,
        actor: str | None,
        release_id_override: str | None = None,
    ) -> VerifiedArtifact:
        """Verify history, append only new values, archive, and atomically publish."""

        if self.settings.model_bundle_dir:
            if release_id_override is not None:
                raise ModelReleaseMismatchError("模型组合使用显式发布入口，不能覆盖单个版本。")
            from .portfolio import publish_portfolio
            return publish_portfolio(self, feature_frame, source=source, raw_frames=raw_frames,
                                     actor=actor, fetch_context=source == "tushare")
        with self._operation_lock:
            candidate_features = canonicalize_features(feature_frame)
            current_features, parent_snapshot = self._current_canonical_features()
            if current_features is None:
                canonical_features = candidate_features
                allow_new = True
            else:
                canonical_features = merge_append_only_features(
                    current_features, candidate_features
                )
                allow_new = True

            as_of = data_as_of(canonical_features)
            known_sessions = self.calendar.sessions(str(canonical_features["trade_date"].iloc[0]).replace("-", ""), as_of)
            if known_sessions is not None:
                actual_dates = set(canonical_features["trade_date"].str.replace("-", "", regex=False))
                missing = set(known_sessions) - actual_dates
                if missing:
                    raise ValueError(f"行情缺少交易日：{', '.join(sorted(missing)[:5])}")
            snapshot_id = str(uuid.uuid4())
            with self.database.session() as session:
                release = (
                    session.get(ModelRelease, release_id_override)
                    if release_id_override is not None
                    else self._active_release(session)
                )
                if release is None:
                    raise ServiceNotReadyError("找不到目标模型发布版本。")
            self._assert_runtime_release(release)
            diagnostics = pd.DataFrame()
            diagnostics_path = self.settings.root_dir / f".diagnostics-{snapshot_id}.csv"
            if self.diagnostics_calculation_function is not None:
                result, diagnostics = self.diagnostics_calculation_function(
                    canonical_features,
                    self.settings,
                    diagnostics_path,
                )
            else:
                result = self.calculation_function(canonical_features, self.settings)
            result, plan = self._candidate_plan(
                features=canonical_features,
                release=release,
                snapshot_id=snapshot_id,
                allow_new=allow_new,
                result=result,
            )
            artifact = self._render_artifact(
                result=result,
                plan=plan,
                snapshot_id=snapshot_id,
                release_id=release.id,
                as_of=as_of,
            )

            input_bytes = frame_to_csv_bytes(canonical_features)
            try:
                archive = write_daily_archive(
                    archive_root=self.settings.archive_dir,
                    snapshot_id=snapshot_id,
                    data_as_of=as_of,
                    features=canonical_features,
                    public_csv=artifact.csv_bytes,
                    raw_frames=raw_frames,
                    extra_files=(
                        {"diagnostics.csv": frame_to_csv_bytes(diagnostics)}
                        if not diagnostics.empty
                        else None
                    ),
                    manifest={
                        "source": source,
                        "parent_snapshot_id": parent_snapshot.id if parent_snapshot else None,
                        "data_as_of": as_of,
                        "release_id": release.id,
                        "source_bundle_sha256": release.source_bundle_sha256,
                        "config_sha256": release.config_sha256,
                        "input_sha256": sha256_bytes(input_bytes),
                        "public_csv_sha256": artifact.csv_sha256,
                        "prediction_fingerprints": {
                            item.signal_date: item.fingerprint for item in plan
                        },
                    },
                )
            finally:
                diagnostics_path.unlink(missing_ok=True)

            with self.database.session() as session:
                session.add(
                    MarketSnapshot(
                        id=snapshot_id,
                        parent_snapshot_id=parent_snapshot.id if parent_snapshot else None,
                        source=source,
                        status="published",
                        data_as_of=as_of,
                        input_sha256=sha256_bytes(input_bytes),
                        features_sha256=archive.features_sha256,
                        features_path=str(archive.features_path.resolve()),
                        manifest_path=str(archive.manifest_path.resolve()),
                    )
                )
                session.flush()
                self._persist_prediction_plan(
                    session,
                    plan=plan,
                    release=release,
                    snapshot_id=snapshot_id,
                )
                self._settle_outstanding(session, canonical_features, release.id, snapshot_id)

                session.execute(
                    update(Publication)
                    .where(Publication.is_active.is_(True))
                    .values(is_active=False)
                )
                session.add(
                    Publication(
                        id=str(uuid.uuid4()),
                        snapshot_id=snapshot_id,
                        release_id=release.id,
                        public_csv_path=str(archive.results_path.resolve()),
                        public_csv_sha256=artifact.csv_sha256,
                        row_count=len(artifact.public_frame),
                        is_active=True,
                    )
                )
                session.add(
                    AuditLog(
                        actor=actor,
                        action="publish_snapshot",
                        detail=json.dumps(
                            {
                                "snapshot_id": snapshot_id,
                                "source": source,
                                "data_as_of": as_of,
                                "rows": len(artifact.public_frame),
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
            return VerifiedArtifact(
                snapshot_id=artifact.snapshot_id,
                release_id=artifact.release_id,
                data_as_of=artifact.data_as_of,
                result=artifact.result,
                public_frame=artifact.public_frame,
                csv_bytes=artifact.csv_bytes,
                csv_sha256=artifact.csv_sha256,
                archive=archive,
            )

    def fetch_tushare_features(self) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Fetch only in a worker; public API requests never call this method."""

        from 数据拉取脚本_tushare import (
            DEFAULT_TUSHARE_MIN_INTERVAL_SECONDS,
            RetryConfig,
            fetch_all,
        )

        retry_config = RetryConfig(
            retries=self.settings.retry_count,
            sleep_seconds=self.settings.retry_sleep_seconds,
            rate_limit_sleep_seconds=self.settings.rate_limit_sleep_seconds,
        )
        token = self._configured_tushare_token()
        end_date = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d")
        datasets = fetch_all(
            self.settings.history_start_date,
            end_date,
            token=token,
            output_dir=self.settings.root_dir / "transient_fetch",
            save=False,
            retry_config=retry_config,
            tushare_min_interval_by_api={
                **DEFAULT_TUSHARE_MIN_INTERVAL_SECONDS,
                "index_global": self.settings.index_global_min_interval,
            },
        )
        features = datasets["merged_features"]
        if features.empty:
            raise RuntimeError("Tushare 未返回可用的合并特征。")
        raw_frames = {
            name: frame
            for name, frame in datasets.items()
            if name != "merged_features" and isinstance(frame, pd.DataFrame)
        }
        return features, raw_frames

    def _configured_tushare_token(self) -> str:
        token = self.settings.tushare_token
        if token:
            return token
        raise RuntimeError("服务未在 config.ini 的 [Tushare] 令牌中设置 Tushare token。")

    def is_sse_trading_day(self, business_date: str) -> bool:
        """Use Tushare's SSE calendar, falling back to weekdays on API failure."""

        parsed = pd.Timestamp(business_date)
        if parsed.weekday() >= 5:
            return False
        with self._calendar_lock:
            cached = self._trading_day_cache.get(business_date)
            if cached is not None:
                return cached
        try:
            from 数据拉取脚本_tushare import get_pro

            calendar = get_pro(self._configured_tushare_token()).trade_cal(
                exchange="SSE",
                start_date=business_date,
                end_date=business_date,
                fields="cal_date,is_open",
            )
            if calendar is None or calendar.empty or "is_open" not in calendar:
                is_open = True
            else:
                is_open = bool(pd.to_numeric(calendar["is_open"], errors="coerce").iloc[0])
        except Exception:
            # A calendar lookup outage must not make the service miss a real
            # trading day. The later data-growth guard still prevents a publish
            # on holidays or stale provider data.
            is_open = True
        with self._calendar_lock:
            self._trading_day_cache[business_date] = is_open
        return is_open

    def refresh_from_tushare(self, *, actor: str | None) -> VerifiedArtifact:
        features, raw_frames = self.fetch_tushare_features()
        self.refresh_calendar()
        current_features, _ = self._current_canonical_features()
        if current_features is not None:
            merged = merge_append_only_features(current_features, features)
            if data_as_of(merged) <= data_as_of(current_features):
                raise NoNewMarketDataError("行情数据截止日未增长，跳过发布。")
        if self.settings.model_bundle_dir and self._active_publication() is not None:
            self._assert_runtime_release(self._load_active_context()[2])
        elif self._active_publication() is not None and self.release_upgrade_required():
            self.promote_compatible_release(actor=actor)
        return self.publish_from_features(
            features,
            source="tushare",
            raw_frames=raw_frames,
            actor=actor,
        )

    def refresh_calendar(self) -> None:
        from 数据拉取脚本_tushare import get_pro
        try:
            now = pd.Timestamp.now(tz="Asia/Shanghai")
            frame = get_pro(self._configured_tushare_token()).trade_cal(
                exchange="SSE", start_date=self.settings.history_start_date,
                end_date=(now + pd.Timedelta(days=370)).strftime("%Y%m%d"), fields="cal_date,is_open",
            )
            self.calendar.store(frame)
        except Exception as exc:
            self.record_audit("system", "calendar_refresh_failed", f"{type(exc).__name__}: 日历更新失败，继续使用已缓存日历。")

    def _shadow_runtime_settings(self) -> Settings:
        return shadow_settings(self.settings)

    def request_bilstm_shadow(
        self,
        snapshot_id: str,
        *,
        actor: str | None,
    ) -> tuple[ShadowRun, bool]:
        """Create or requeue one non-public causal BiLSTM run for a snapshot."""

        with self.database.session() as session:
            snapshot = session.get(MarketSnapshot, snapshot_id)
            if snapshot is None:
                raise FileNotFoundError("找不到影子运行所需的行情快照。")
            shadow_runtime = self._shadow_runtime_settings()
            release = self._ensure_release_for_settings(session, shadow_runtime)
            existing = session.scalar(
                select(ShadowRun).where(
                    ShadowRun.snapshot_id == snapshot_id,
                    ShadowRun.release_id == release.id,
                )
            )
            if existing is not None:
                if existing.status in {"queued", "running", "succeeded"}:
                    return existing, False
                existing.status = "queued"
                existing.requested_by = actor
                existing.message = None
                existing.started_at = None
                existing.finished_at = None
                existing.result_csv_path = None
                existing.result_csv_sha256 = None
                existing.manifest_path = None
                existing.row_count = None
                return existing, True
            run = ShadowRun(
                id=str(uuid.uuid4()),
                engine="bilstm_causal",
                release_id=release.id,
                snapshot_id=snapshot_id,
                status="queued",
                requested_by=actor,
            )
            session.add(run)
            session.flush()
            return run, True

    def run_bilstm_shadow(self, run_id: str) -> ShadowRun:
        """Calculate a shadow run without mutating public publication state."""

        with self.database.session() as session:
            run = session.get(ShadowRun, run_id)
            if run is None:
                raise FileNotFoundError("找不到影子运行任务。")
            if run.status == "succeeded":
                return run
            snapshot = session.get(MarketSnapshot, run.snapshot_id)
            release = session.get(ModelRelease, run.release_id)
            publication = session.scalar(
                select(Publication)
                .where(Publication.snapshot_id == run.snapshot_id)
                .order_by(Publication.published_at.desc())
            )
            if snapshot is None or release is None:
                raise ServiceNotReadyError("影子运行缺少快照或模型版本。")
            run.status = "running"
            run.started_at = utcnow()
            run.message = None
            snapshot_id = snapshot.id
            release_id_value = release.id

        self._assert_archive_integrity(snapshot, publication)
        features = read_features(snapshot.features_path)
        shadow_runtime = self._shadow_runtime_settings()
        current_release_id, _, _, _ = release_id(shadow_runtime)
        if release_id_value != current_release_id:
            raise ModelReleaseMismatchError(
                "影子模型代码或参数已变化；请为当前快照创建新的影子运行。"
            )

        result = self.shadow_calculation_function(features, self.settings)
        result, plan = self._candidate_plan(
            features=features,
            release=release,
            snapshot_id=snapshot_id,
            allow_new=True,
            result=result,
            inherit_provenance=False,
        )
        artifact = self._render_artifact(
            result=result,
            plan=plan,
            snapshot_id=snapshot_id,
            release_id=release.id,
            as_of=snapshot.data_as_of,
        )
        archive = write_daily_archive(
            archive_root=self.settings.shadow_archive_dir,
            snapshot_id=run_id,
            data_as_of=snapshot.data_as_of,
            features=features,
            public_csv=artifact.csv_bytes,
            raw_frames=None,
            manifest={
                "source": "shadow_bilstm_causal",
                "market_snapshot_id": snapshot_id,
                "market_snapshot_features_sha256": snapshot.features_sha256,
                "release_id": release.id,
                "source_bundle_sha256": release.source_bundle_sha256,
                "config_sha256": release.config_sha256,
                "public_csv_sha256": artifact.csv_sha256,
                "prediction_fingerprints": {
                    item.signal_date: item.fingerprint for item in plan
                },
            },
        )

        with self.database.session() as session:
            run = session.get(ShadowRun, run_id)
            release = session.get(ModelRelease, release_id_value)
            if run is None or release is None:
                raise ServiceNotReadyError("影子运行在写入前丢失。")
            self._persist_prediction_plan(
                session,
                plan=plan,
                release=release,
                snapshot_id=snapshot_id,
            )
            run.status = "succeeded"
            run.result_csv_path = str(archive.results_path.resolve())
            run.result_csv_sha256 = artifact.csv_sha256
            run.manifest_path = str(archive.manifest_path.resolve())
            run.row_count = len(artifact.public_frame)
            run.message = "BiLSTM 影子预测已完成，未影响公开发布。"
            run.finished_at = utcnow()
            session.add(
                AuditLog(
                    actor=run.requested_by,
                    action="bilstm_shadow_succeeded",
                    detail=json.dumps(
                        {
                            "shadow_run_id": run.id,
                            "snapshot_id": snapshot_id,
                            "release_id": release.id,
                            "rows": run.row_count,
                        },
                        ensure_ascii=False,
                    ),
                )
            )
            return run

    def mark_bilstm_shadow_failed(self, run_id: str, message: str) -> None:
        with self.database.session() as session:
            run = session.get(ShadowRun, run_id)
            if run is None:
                return
            run.status = "failed"
            run.message = message
            run.finished_at = utcnow()
            session.add(
                AuditLog(
                    actor=run.requested_by,
                    action="bilstm_shadow_failed",
                    detail=json.dumps({"shadow_run_id": run_id, "message": message}, ensure_ascii=False),
                )
            )

    def _shadow_run_metrics(self, run: ShadowRun) -> dict[str, Any] | None:
        if (
            run.status != "succeeded"
            or not run.result_csv_path
            or not run.result_csv_sha256
            or not run.manifest_path
        ):
            return None
        result_path = Path(run.result_csv_path)
        manifest_path = Path(run.manifest_path)
        if not result_path.exists() or sha256_file(result_path) != run.result_csv_sha256:
            raise PredictionDriftError("BiLSTM 影子结果 CSV 哈希不匹配。")
        self._assert_manifest_integrity(manifest_path)
        frame = pd.read_csv(result_path, encoding="utf-8-sig")
        return statistics(frame)

    def shadow_dashboard_data(self) -> dict[str, Any]:
        with self.database.session() as session:
            run = session.scalar(
                select(ShadowRun).where(ShadowRun.engine == "bilstm_causal")
                .order_by(ShadowRun.created_at.desc()).limit(1)
            )
        if run is None:
            return {
                "enabled": self.settings.bilstm_shadow_enabled,
                "run": None,
                "metrics": None,
                "integrity_error": None,
            }
        try:
            metrics = self._shadow_run_metrics(run)
            integrity_error = None
        except PredictionDriftError as exc:
            metrics = None
            integrity_error = str(exc)
        return {
            "enabled": self.settings.bilstm_shadow_enabled,
            "run": run,
            "metrics": metrics,
            "integrity_error": integrity_error,
        }

    def get_shadow_run(self, run_id: str) -> ShadowRun | None:
        with self.database.session() as session:
            return session.get(ShadowRun, run_id)

    def shadow_result_file(self, run_id: str) -> Path:
        run = self.get_shadow_run(run_id)
        if run is None or not run.result_csv_path:
            raise FileNotFoundError("找不到 BiLSTM 影子结果。")
        self._shadow_run_metrics(run)
        return Path(run.result_csv_path)

    def history_frame(self, release_id: str, data_as_of: str) -> pd.DataFrame:
        with self.database.session() as session:
            pairs = session.execute(
                select(PredictionLedger, OutcomeResolution)
                .join(OutcomeResolution, OutcomeResolution.prediction_id == PredictionLedger.id)
                .where(PredictionLedger.release_id == release_id, OutcomeResolution.target_trade_date <= data_as_of)
                .order_by(OutcomeResolution.target_trade_date)
            ).all()
        records = []
        for prediction, outcome in pairs:
            generated = prediction.created_at.replace(tzinfo=timezone.utc) if prediction.created_at.tzinfo is None else prediction.created_at
            opening = datetime.strptime(outcome.target_trade_date, "%Y%m%d").replace(hour=9, minute=30, tzinfo=SHANGHAI)
            records.append({
                "结果类型": "循环验证", "信号日期": int(prediction.signal_date),
                "预测目标交易日": outcome.target_trade_date,
                "预测方向": "上涨" if prediction.predicted_label else "下跌",
                "预测次日涨跌幅": prediction.predicted_return,
                "预测次日收盘价": prediction.predicted_close,
                "置信度": prediction.calibrated_confidence,
                "次日实际涨跌幅": outcome.actual_return,
                "方向预测正确": outcome.correct,
                "记录来源": "live" if generated < opening else "backfill",
                "预测生成时间": generated.astimezone(SHANGHAI).isoformat(timespec="seconds"),
            })
        return pd.DataFrame(records, columns=["结果类型", "信号日期", "预测目标交易日", "预测方向", "预测次日涨跌幅", "预测次日收盘价", "置信度", "次日实际涨跌幅", "方向预测正确", "记录来源", "预测生成时间"])

    def performance_data(self, days: int = 60) -> dict[str, Any]:
        if not 1 <= days <= 5000:
            raise ValueError("统计天数必须在 1 到 5000 之间。")
        publication, snapshot, release = self._load_active_context()
        self.read_published((publication, snapshot, release))
        history = self.history_frame(release.id, snapshot.data_as_of)
        return {
            "snapshot_id": snapshot.id,
            "model_release": release.id,
            "algorithm_id": release.algorithm_id,
            "data_as_of": snapshot.data_as_of,
            "metrics": {key: value for key, value in statistics(history, days).items() if key != "records"},
            "recent_20": {key: value for key, value in statistics(history, 20).items() if key != "records"},
        }

    def dashboard_data(self, days: int = 60) -> dict[str, Any]:
        if not 1 <= days <= 5000:
            raise ValueError("统计天数必须在 1 到 5000 之间。")
        publication, snapshot, release = self._load_active_context()
        artifact = self.read_published((publication, snapshot, release))
        frame = artifact.public_frame
        history = self.history_frame(release.id, snapshot.data_as_of)
        metrics = statistics(history, days)
        pending = frame.loc[frame["结果类型"].eq("次日预测")]
        latest = clean_records(pending)[-1] if len(pending) else None
        if latest and not latest.get("预测目标交易日"):
            latest["预测目标交易日"] = self.calendar.next_session(snapshot.data_as_of)
        veto = self._veto_attribution(snapshot)
        with self.database.session() as session:
            jobs = session.scalars(
                select(RefreshJob).order_by(RefreshJob.created_at.desc()).limit(10)
            ).all()
            publications = session.scalars(
                select(Publication).order_by(Publication.published_at.desc()).limit(30)
            ).all()
        return {
            "publication": publication,
            "snapshot": snapshot,
            "release": release,
            "model_name": json.loads(release.config_json).get("model_name", "原生产基线"),
            "portfolio_enabled": bool(self.settings.model_bundle_dir),
            "release_upgrade_required": self.release_upgrade_required(),
            "rows": list(reversed(metrics["records"])),
            "latest": latest,
            "metrics": metrics,
            "days": days,
            "health": self.health(artifact=artifact),
            "veto": veto,
            "shadow": self.shadow_dashboard_data(),
            "jobs": jobs,
            "publications": publications,
            "scheduled": self.settings.scheduled_refresh_enabled,
            "schedule_time": f"{self.settings.scheduled_refresh_hour:02d}:{self.settings.scheduled_refresh_minute:02d}",
        }

    def list_archives(self) -> list[tuple[Publication, MarketSnapshot]]:
        with self.database.session() as session:
            publications = session.scalars(
                select(Publication).order_by(Publication.published_at.desc())
            ).all()
            snapshots = {
                snapshot.id: snapshot
                for snapshot in session.scalars(select(MarketSnapshot)).all()
            }
        return [
            (publication, snapshots[publication.snapshot_id])
            for publication in publications
            if publication.snapshot_id in snapshots
        ]

    def archive_file(self, snapshot_id: str, filename: str) -> Path:
        if filename not in {"results.csv", "features.csv", "diagnostics.csv", "manifest.json"}:
            raise FileNotFoundError("不允许下载该归档文件。")
        with self.database.session() as session:
            snapshot = session.get(MarketSnapshot, snapshot_id)
            publication = session.scalar(
                select(Publication)
                .where(Publication.snapshot_id == snapshot_id)
                .order_by(Publication.published_at.desc())
            )
        if snapshot is None:
            raise FileNotFoundError("找不到归档快照。")
        self._assert_archive_integrity(snapshot, publication)
        path = Path(snapshot.manifest_path).parent / filename
        if not path.exists():
            raise FileNotFoundError("归档文件不存在。")
        return path

    def _veto_attribution(self, snapshot: MarketSnapshot) -> dict[str, Any] | None:
        diagnostics_path = Path(snapshot.manifest_path).parent / "diagnostics.csv"
        if not diagnostics_path.exists():
            return None
        diagnostics = pd.read_csv(diagnostics_path, encoding="utf-8-sig")
        required = {
            "veto_applied",
            "predicted_label",
            "base_predicted_label",
            "real_pct_change",
            "correct",
        }
        if diagnostics.empty or not required.issubset(diagnostics.columns):
            return None
        diagnostics = diagnostics.loc[diagnostics["real_pct_change"].notna()].sort_values("trade_date").reset_index(drop=True)
        veto = pd.to_numeric(diagnostics["veto_applied"], errors="coerce").fillna(0).eq(1)

        def summarize(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
            selected = frame.loc[mask].copy()
            total = len(frame)
            if selected.empty:
                return {
                    "rows": 0,
                    "trigger_rate": 0.0 if total else None,
                    "accuracy": None,
                    "accuracy_lift": None,
                    "directional_return_lift": None,
                }
            realized = pd.to_numeric(selected["real_pct_change"], errors="coerce")
            final_label = pd.to_numeric(selected["predicted_label"], errors="coerce")
            base_label = pd.to_numeric(selected["base_predicted_label"], errors="coerce")
            final_directional = np.where(final_label.gt(0), realized, -realized)
            base_directional = np.where(base_label.gt(0), realized, -realized)
            final_correct = selected["correct"].astype("boolean")
            base_correct = base_label.eq((realized > 0).astype(int)).astype("boolean")
            return {
                "rows": len(selected),
                "trigger_rate": len(selected) / total if total else None,
                "accuracy": float(final_correct.mean()),
                "accuracy_lift": float(final_correct.mean() - base_correct.mean()),
                "directional_return_lift": float(
                    np.nansum(final_directional - base_directional)
                ),
            }

        return {
            "all": summarize(diagnostics, veto),
            "recent_20": summarize(diagnostics.tail(20), veto.tail(20)),
        }

    def record_audit(self, actor: str | None, action: str, detail: str | None = None) -> None:
        with self.database.session() as session:
            session.add(AuditLog(actor=actor, action=action, detail=detail))

    def admin_is_configured(self) -> bool:
        with self.database.session() as session:
            return session.scalar(select(AdminUser.id).limit(1)) is not None

    def shutdown(self) -> None:
        self.database.dispose()


class RefreshManager:
    """Single-worker refresh queue so manual and scheduled jobs cannot overlap."""

    def __init__(
        self,
        service: PredictionService,
        *,
        on_success: Callable[[VerifiedArtifact, str | None], None] | None = None,
    ) -> None:
        self.service = service
        self.on_success = on_success
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prediction-refresh")
        self._lock = threading.Lock()

    def submit(
        self,
        *,
        trigger: str,
        actor: str | None,
        idempotency_key: str | None = None,
    ) -> tuple[str, bool]:
        with self._lock:
            with self.service.database.session() as session:
                stored_trigger = (
                    f"{trigger}:{idempotency_key}" if idempotency_key else trigger
                )
                if idempotency_key:
                    attempts = session.scalars(
                        select(RefreshJob)
                        .where(RefreshJob.trigger == stored_trigger)
                        .order_by(RefreshJob.created_at.desc())
                    ).all()
                    if attempts:
                        existing = attempts[0]
                        finished = existing.finished_at or existing.created_at
                        finished = finished.replace(tzinfo=timezone.utc) if finished.tzinfo is None else finished
                        publication = self.service._active_publication()
                        snapshot = session.get(MarketSnapshot, publication.snapshot_id) if publication else None
                        complete = snapshot is not None and snapshot.data_as_of >= idempotency_key
                        if complete or existing.status in {"queued", "running"} or len(attempts) >= 6 or utcnow() - finished < timedelta(minutes=10):
                            return existing.id, False
                running = session.scalar(
                    select(RefreshJob).where(RefreshJob.status.in_(("queued", "running")))
                )
                if running is not None:
                    return running.id, False
                job = RefreshJob(
                    id=str(uuid.uuid4()),
                    trigger=stored_trigger,
                    requested_by=actor,
                    status="queued",
                )
                session.add(job)
                job_id = job.id
            self._executor.submit(self._run, job_id)
            return job_id, True

    def _run(self, job_id: str) -> None:
        with self.service.database.session() as session:
            job = session.get(RefreshJob, job_id)
            if job is None:
                return
            job.status = "running"
            job.started_at = utcnow()
            job.heartbeat_at = utcnow()
            actor = job.requested_by
            trigger = job.trigger
        heartbeat_stop = threading.Event()

        def heartbeat() -> None:
            while not heartbeat_stop.wait(10):
                try:
                    with self.service.database.session() as session:
                        current = session.get(RefreshJob, job_id)
                        if current is not None and current.status == "running":
                            current.heartbeat_at = utcnow()
                except Exception:
                    break

        pulse = threading.Thread(target=heartbeat, daemon=True, name="refresh-heartbeat")
        pulse.start()
        try:
            if trigger == "promotion":
                artifact = self.service.promote_compatible_release(actor=actor) or self.service.read_published()
            else:
                artifact = self.service.refresh_from_tushare(actor=actor)
        except (HistoricalMarketDataDriftError, PredictionDriftError) as exc:
            status = "drift_detected"
            message = str(exc)
            snapshot_id = None
        except NoNewMarketDataError as exc:
            status = "skipped"
            message = str(exc)
            snapshot_id = None
        except Exception as exc:  # pragma: no cover - guarded by integration tests at boundary.
            status = "failed"
            message = f"{exc}\n{traceback.format_exc(limit=8)}"
            snapshot_id = None
        else:
            status = "succeeded"
            message = "刷新完成并通过历史预测校验。"
            snapshot_id = artifact.snapshot_id
            if self.on_success is not None:
                try:
                    self.on_success(artifact, actor)
                except Exception as exc:  # Shadow scheduling must not revoke publication.
                    self.service.record_audit(
                        actor,
                        "bilstm_shadow_schedule_failed",
                        f"{type(exc).__name__}: {exc}",
                    )
        finally:
            heartbeat_stop.set()
            pulse.join(timeout=2)
        with self.service.database.session() as session:
            job = session.get(RefreshJob, job_id)
            if job is not None:
                job.status = status
                job.snapshot_id = snapshot_id
                job.message = message
                job.finished_at = utcnow()
            session.add(
                AuditLog(
                    actor=None,
                    action=f"refresh_{status}",
                    detail=json.dumps({"job_id": job_id, "message": message}, ensure_ascii=False),
                )
            )

    def get(self, job_id: str) -> RefreshJob | None:
        with self.service.database.session() as session:
            return session.get(RefreshJob, job_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


class ShadowManager:
    """Run the expensive BiLSTM research engine away from public publication."""

    def __init__(self, service: PredictionService) -> None:
        self.service = service
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bilstm-shadow")
        self._lock = threading.Lock()

    def submit(self, *, snapshot_id: str, actor: str | None, force: bool = False) -> tuple[str | None, bool]:
        if not force and not self.service.settings.bilstm_shadow_enabled:
            return None, False
        with self._lock:
            run, created = self.service.request_bilstm_shadow(snapshot_id, actor=actor)
            if created:
                self._executor.submit(self._run, run.id)
            return run.id, created

    def _run(self, run_id: str) -> None:
        try:
            self.service.run_bilstm_shadow(run_id)
        except Exception as exc:  # A shadow failure must never affect public publication.
            self.service.mark_bilstm_shadow_failed(
                run_id,
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}",
            )

    def get(self, run_id: str) -> ShadowRun | None:
        return self.service.get_shadow_run(run_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
