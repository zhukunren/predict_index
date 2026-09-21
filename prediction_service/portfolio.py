"""One shared snapshot, four immutable forecast streams, one public model."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import uuid

import pandas as pd
from sqlalchemy import select, update

from .archive import read_features, frame_to_csv_bytes, sha256_bytes, write_daily_archive
from .calendar import SHANGHAI
from .engine import canonicalize_features, merge_append_only_features, data_as_of
from .forecast_models import MODELS, PRODUCTION_KEY, CONTEXT_NAMES
from .model_registry import ModelBundle, canonical_json, read_frame
from .models import ModelRelease, MarketSnapshot, Publication, ShadowRun, AuditLog, utcnow


def context_for_snapshot(service, bundle, snapshot):
    if snapshot:
        service._assert_archive_integrity(snapshot, None)
        manifest = json.loads(Path(snapshot.manifest_path).read_text(encoding="utf-8"))
        if manifest.get("model_bundle_id") == bundle.manifest["bundle_id"]:
            directory = Path(snapshot.manifest_path).parent / "raw"
            return {name: read_frame(directory / f"context_{name}.csv") for name in CONTEXT_NAMES}
    return bundle.context()


def releases_for_bundle(bundle):
    releases = {}
    for model in MODELS:
        item = bundle.manifest["releases"][model.key]
        config = item["configuration"]
        releases[model.key] = ModelRelease(
            id=item["release_id"], algorithm_id=config["algorithm_id"],
            source_bundle_sha256=sha256_bytes(canonical_json(config["runtime"]["sources"])),
            config_sha256=sha256_bytes(canonical_json(config)),
            config_json=canonical_json(config).decode(),
        )
    return releases


def publish_portfolio(service, feature_frame, *, source, actor, raw_frames=None,
                      fetch_context=False, activate=False):
    from .service import ModelReleaseMismatchError, NoNewMarketDataError

    with service._operation_lock:
        bundle = ModelBundle(service.settings.model_bundle_dir)
        releases = releases_for_bundle(bundle)
        current, parent = service._current_canonical_features()
        features = canonicalize_features(feature_frame)
        if current is not None:
            features = merge_append_only_features(current, features)
        as_of = data_as_of(features)
        active = service._active_publication()
        production = releases[PRODUCTION_KEY]
        if active and active.release_id != production.id and not activate:
            raise ModelReleaseMismatchError("请先显式激活模型组合，再执行日常刷新。")
        if active and active.release_id == production.id and parent.data_as_of == as_of:
            return service.read_published()
        earliest = datetime.strptime(as_of, "%Y%m%d").replace(hour=20, tzinfo=SHANGHAI)
        if utcnow() < earliest:
            raise NoNewMarketDataError("当日期权模型最早在上海时间 20:00 后计算发布。")
        target = service.calendar.next_session(as_of)
        if target is None:
            raise ValueError("无法确认下一交易日，模型组合暂不发布。")
        dates = features.trade_date.str.replace("-", "", regex=False)
        sessions = service.calendar.sessions(dates.iloc[0], as_of)
        if sessions is not None and set(sessions) - set(dates):
            raise ValueError("共享行情缺少交易日，模型组合暂不发布。")
        context = context_for_snapshot(service, bundle, parent)
        raw, receipts = {}, []
        if fetch_context:
            from .model_context import extend_model_context
            from 数据拉取脚本_tushare import get_pro
            context, raw, receipts = extend_model_context(context, features, get_pro(service._configured_tushare_token()))
        results = bundle.calculate(features, context)
        return persist_portfolio(service, bundle, features, results, context, releases,
                                 parent=parent, source=source, actor=actor,
                                 raw_frames={**(raw_frames or {}), **raw}, receipts=receipts)


def persist_portfolio(service, bundle, features, results, context, releases, *,
                      parent, source, actor, raw_frames=None, receipts=()):
    """Validate every stream before any database publication is changed."""
    from .service import PredictionDriftError

    snapshot_id = str(uuid.uuid4())
    as_of = data_as_of(features)
    artifacts, plans = {}, {}
    production_dates = results[PRODUCTION_KEY].trade_date.tolist()
    for model in MODELS:
        result = results[model.key]
        if result.trade_date.tolist() != production_dates or int(result.trade_date.iloc[-1]) != int(as_of):
            raise PredictionDriftError("各模型预测日期必须完全一致并覆盖最新信号日。")
        _, plans[model.key] = service._candidate_plan(
            features=features, release=releases[model.key], snapshot_id=snapshot_id,
            allow_new=True, result=result, inherit_provenance=False,
        )
        # Fixed public contract: 60 settled trading days plus the next forecast.
        artifacts[model.key] = service._render_artifact(
            result=result.tail(61), plan=plans[model.key], snapshot_id=snapshot_id,
            release_id=releases[model.key].id, as_of=as_of,
        )
    # Stamps describe when all model computations completed, including in slow runs.
    generated_at = utcnow()
    for key, plan in plans.items():
        refreshed = []
        for item in plan:
            if item.is_new:
                opening = datetime.strptime(item.target_date, "%Y%m%d").replace(hour=9, minute=30, tzinfo=SHANGHAI) if item.target_date else None
                item = replace(item, generated_at=generated_at,
                               origin="live" if opening and generated_at < opening else "backfill")
            refreshed.append(item)
        plans[key] = refreshed
        artifacts[key] = service._render_artifact(result=results[key].tail(61), plan=refreshed,
                                                  snapshot_id=snapshot_id, release_id=releases[key].id, as_of=as_of)
    metadata = {
        model.key: {"name": model.name, "release_id": releases[model.key].id,
                    "role": "production" if model.key == PRODUCTION_KEY else "shadow",
                    "results_file": f"{model.key}_results.csv"}
        for model in MODELS
    }
    extra_files = {f"{key}_results.csv": artifact.csv_bytes for key, artifact in artifacts.items()}
    extra_files.update({f"{key}_replay.csv": frame_to_csv_bytes(result) for key, result in results.items()})
    extra_files["model_bundle.json"] = canonical_json(bundle.manifest)
    public = artifacts[PRODUCTION_KEY]
    archive = write_daily_archive(
        archive_root=service.settings.archive_dir, snapshot_id=snapshot_id, data_as_of=as_of,
        features=features, public_csv=public.csv_bytes,
        raw_frames={**(raw_frames or {}), **{f"context_{name}": frame for name, frame in context.items()}},
        extra_files=extra_files,
        manifest={"source": source, "parent_snapshot_id": parent.id if parent else None,
                  "data_as_of": as_of, "release_id": releases[PRODUCTION_KEY].id,
                  "model_bundle_id": bundle.manifest["bundle_id"], "production_model": PRODUCTION_KEY,
                  "models": metadata, "receipts": list(receipts), "scheduled_time_shanghai": "20:15",
                  "activation_basis": "explicit_user_selection", "historical_gate_passed": False},
    )
    # Every release, all forecasts and the active pointer commit together.
    with service.database.session() as session:
        for release in releases.values():
            existing = session.get(ModelRelease, release.id)
            if existing is None:
                session.add(release)
            elif json.loads(existing.config_json) != json.loads(release.config_json):
                raise PredictionDriftError("登记模型参数与冻结版本不一致。")
        session.add(MarketSnapshot(
            id=snapshot_id, parent_snapshot_id=parent.id if parent else None,
            source=source, status="published", data_as_of=as_of,
            input_sha256=sha256_bytes(frame_to_csv_bytes(features)), features_sha256=archive.features_sha256,
            features_path=str(archive.features_path.resolve()), manifest_path=str(archive.manifest_path.resolve()),
        ))
        session.flush()
        for model in MODELS:
            release = releases[model.key]
            service._persist_prediction_plan(session, plan=plans[model.key], release=release, snapshot_id=snapshot_id)
            service._settle_outstanding(session, features, release.id, snapshot_id)
            if model.key != PRODUCTION_KEY:
                session.add(ShadowRun(
                    id=str(uuid.uuid4()), engine=release.algorithm_id, release_id=release.id,
                    snapshot_id=snapshot_id, status="succeeded", requested_by=actor,
                    result_csv_path=str(archive.directory / f"{model.key}_results.csv"),
                    result_csv_sha256=artifacts[model.key].csv_sha256,
                    manifest_path=str(archive.manifest_path), row_count=len(artifacts[model.key].public_frame),
                    started_at=generated_at, finished_at=utcnow(), message="与生产模型使用相同行情和统计口径。",
                ))
        previous = session.scalar(select(Publication).where(Publication.is_active.is_(True)))
        if previous and previous.release_id not in {item.id for item in releases.values()}:
            service._settle_outstanding(session, features, previous.release_id, snapshot_id)
        session.execute(update(Publication).where(Publication.is_active.is_(True)).values(is_active=False))
        session.add(Publication(id=str(uuid.uuid4()), snapshot_id=snapshot_id, release_id=releases[PRODUCTION_KEY].id,
                                public_csv_path=str(archive.results_path), public_csv_sha256=public.csv_sha256,
                                row_count=len(public.public_frame), is_active=True))
        session.add(AuditLog(actor=actor, action="model_portfolio_published", detail=canonical_json({
            "snapshot_id": snapshot_id, "from_release": previous.release_id if previous else None,
            "models": metadata, "bundle_id": bundle.manifest["bundle_id"],
            "historical_gate_passed": False, "activation_basis": "explicit_user_selection",
        }).decode()))
    return replace(public, archive=archive)


def recompute_portfolio(service):
    """Offline replay uses frozen context, verifies all streams and public bytes."""
    from .service import PredictionDriftError

    with service._operation_lock:
        publication, snapshot, release = service._load_active_context()
        service._assert_runtime_release(release)
        service._assert_archive_integrity(snapshot, publication)
        bundle = ModelBundle(service.settings.model_bundle_dir)
        features = read_features(snapshot.features_path)
        context = context_for_snapshot(service, bundle, snapshot)
        results = bundle.calculate(features, context)
        releases = releases_for_bundle(bundle)
        artifact = None
        for model in MODELS:
            _, plan = service._candidate_plan(features=features, release=releases[model.key], snapshot_id=snapshot.id,
                                               allow_new=False, result=results[model.key], inherit_provenance=False)
            stored_path = Path(snapshot.manifest_path).parent / f"{model.key}_results.csv"
            artifact = service._render_artifact(result=results[model.key].tail(61), plan=plan, snapshot_id=snapshot.id,
                                                 release_id=releases[model.key].id, as_of=snapshot.data_as_of,
                                                 stored_metadata=pd.read_csv(stored_path, encoding="utf-8-sig", dtype={"预测目标交易日": "string"}))
            if artifact.csv_bytes != stored_path.read_bytes():
                raise PredictionDriftError(f"{model.name}重算与归档不一致。")
            if model.key == PRODUCTION_KEY and artifact.csv_sha256 != publication.public_csv_sha256:
                raise PredictionDriftError("生产模型重算 CSV 与公开结果不一致。")
        return service.read_published()
