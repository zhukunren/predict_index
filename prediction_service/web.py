"""FastAPI application exposing public prediction data and a protected panel."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, Query
from filelock import FileLock, Timeout
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.cors import CORSMiddleware
from sqlalchemy import select

from .config import Settings
from .calendar import SHANGHAI
from .metrics import clean_records, finite
from .archive import frame_to_csv_bytes, sha256_bytes
from .model_comparison import comparison_data, comparison_frame, model_csv
from .models import RefreshJob
from .scheduler import DailyRefreshScheduler
from .security import LoginRateLimiter, csrf_token, validate_csrf, verify_password
from .service import (
    ModelReleaseMismatchError,
    PredictionDriftError,
    PredictionService,
    RefreshManager,
    ShadowManager,
    ServiceNotReadyError,
)


PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))
PUBLIC_JSON_FIELDS = {
    "signal_date": "信号日期",
    "predicted_next_day_return": "预测次日涨跌幅",
    "predicted_direction": "预测方向",
    "predicted_next_day_close": "预测次日收盘价",
    "confidence": "置信度",
    "actual_next_day_return": "次日实际涨跌幅",
    "direction_prediction_correct": "方向预测正确",
}
PUBLIC_DIRECTION_VALUES = {"上涨": "up", "下跌": "down"}


class AppContainer:
    def __init__(self, settings: Settings, service: PredictionService) -> None:
        self.settings = settings
        self.service = service
        self.shadow_manager = ShadowManager(service)
        self.refresh_manager = RefreshManager(
            service,
            on_success=self._submit_shadow_after_refresh,
        )
        self.scheduler = DailyRefreshScheduler(
            settings,
            self.refresh_manager,
            is_trading_day=service.is_sse_trading_day,
            expected_date=lambda now: service.calendar.expected_as_of(now, settings.scheduled_refresh_hour, settings.scheduled_refresh_minute),
        )
        self.login_limiter = LoginRateLimiter()

    def _submit_shadow_after_refresh(
        self,
        artifact,
        actor: str | None,
    ) -> None:
        self.shadow_manager.submit(snapshot_id=artifact.snapshot_id, actor=actor)

    def submit_active_shadow(self, *, actor: str | None, force: bool = False) -> tuple[str | None, bool]:
        _, snapshot, _ = self.service._load_active_context()
        return self.shadow_manager.submit(
            snapshot_id=snapshot.id,
            actor=actor,
            force=force,
        )


def _client_host(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _admin_username(request: Request) -> str | None:
    value = request.session.get("admin_username")
    return value if isinstance(value, str) and value else None


def _admin_redirect(request: Request) -> RedirectResponse | None:
    if _admin_username(request) is not None:
        return None
    return RedirectResponse(url="/admin/login", status_code=303)


def _context(request: Request, **kwargs: Any) -> dict[str, Any]:
    return {
        "request": request,
        "csrf_token": csrf_token(request.session),
        "admin_username": _admin_username(request),
        **kwargs,
    }


def _as_percent(value: float | None) -> str:
    return "-" if not finite(value) else f"{float(value):.2%}"


def _date(value: Any) -> str:
    if value is None or value == "":
        return "待确认"
    text = str(value).removesuffix(".0").replace("-", "")
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}" if len(text) == 8 and text.isdigit() else str(value)


def _local_time(value: datetime | None) -> str:
    if value is None:
        return "-"
    aware = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    return aware.astimezone(SHANGHAI).strftime("%m-%d %H:%M:%S")


TEMPLATES.env.filters["percent"] = _as_percent
TEMPLATES.env.filters["date"] = _date
TEMPLATES.env.filters["localtime"] = _local_time
TEMPLATES.env.filters["signed_percent"] = lambda value: f"{float(value):+.3%}" if finite(value) else "-"
TEMPLATES.env.filters["price"] = lambda value: f"{float(value):,.2f}" if finite(value) else "-"
TEMPLATES.env.filters["job_label"] = lambda value: {"queued": "排队中", "running": "运行中", "succeeded": "已完成", "failed": "失败", "skipped": "暂无新数据", "drift_detected": "数据差异", "interrupted": "已中断"}.get(value, value)


def create_app(
    settings: Settings | None = None,
    *,
    service: PredictionService | None = None,
    bootstrap: bool = True,
) -> FastAPI:
    runtime_settings = settings or Settings.from_config()
    prediction_service = service or PredictionService(runtime_settings)
    container = AppContainer(runtime_settings, prediction_service)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        instance_lock = FileLock(str(runtime_settings.root_dir / "service.lock"), timeout=0)
        try:
            instance_lock.acquire()
        except Timeout as exc:
            raise RuntimeError("该数据目录已被另一个服务实例使用。") from exc
        try:
            container.service.initialize(bootstrap=bootstrap)
            container.service.recover_interrupted_jobs()
            container.scheduler.start()
            if runtime_settings.bilstm_shadow_enabled:
                try:
                    container.submit_active_shadow(actor="system")
                except ServiceNotReadyError:
                    pass
            yield
        finally:
            container.scheduler.stop()
            container.refresh_manager.shutdown()
            container.shadow_manager.shutdown()
            container.service.shutdown()
            instance_lock.release()

    app = FastAPI(
        title="上证指数次日预测服务",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        SessionMiddleware,
        secret_key=runtime_settings.session_secret,
        session_cookie="prediction_admin_session",
        max_age=24 * 60 * 60,
        same_site="lax",
        https_only=runtime_settings.cookie_secure,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "HEAD"], expose_headers=["ETag", "X-Data-As-Of", "X-Snapshot-Id", "X-Model-Release"])
    app.mount("/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static")
    app.state.container = container

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/admin/", status_code=303)

    @app.get("/livez")
    def livez():
        return {"status": "ok"}

    @app.get("/healthz")
    def healthz():
        health = container.service.health()
        return JSONResponse(health, status_code=200 if health["status"] == "ok" else 503)

    @app.get("/api/v1/sh000001/status")
    def publication_status():
        return container.service.health()

    @app.get("/api/v1/sh000001/metrics")
    def prediction_metrics(days: int = Query(60, ge=1, le=5000)):
        try:
            return container.service.performance_data(days)
        except ServiceNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.exception_handler(FileNotFoundError)
    async def not_found(_request, _exc):
        return JSONResponse({"detail": "所请求的文件或记录不存在。"}, status_code=404)

    @app.exception_handler(PredictionDriftError)
    async def invalid_archive(_request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.api_route("/api/v1/sh000001/latest.json", methods=["GET", "HEAD"])
    @app.api_route("/api/v1/sh000001/latest.csv", methods=["GET", "HEAD"], include_in_schema=False)
    def latest_json(request: Request) -> Response:
        try:
            artifact = container.service.read_published()
        except (ServiceNotReadyError, ModelReleaseMismatchError, PredictionDriftError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        source_fields = list(PUBLIC_JSON_FIELDS.values())
        source_records = clean_records(artifact.public_frame.loc[:, source_fields])
        records = [
            {field: record[source] for field, source in PUBLIC_JSON_FIELDS.items()}
            for record in source_records
        ]
        for record in records:
            record["predicted_direction"] = PUBLIC_DIRECTION_VALUES[
                record["predicted_direction"]
            ]
        payload = {
            "code": 0,
            "data": {"msg": "success", "items": records},
            "status": 200,
        }
        content = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        headers = {
            "ETag": f'"{sha256_bytes(content)}"',
            "X-Snapshot-Id": artifact.snapshot_id,
            "X-Model-Release": artifact.release_id,
            "X-Data-As-Of": artifact.data_as_of,
            "Cache-Control": "no-cache",
        }
        supplied = request.headers.get("if-none-match", "")
        if any(tag.strip().removeprefix("W/") in {headers["ETag"], "*"} for tag in supplied.split(",")):
            return Response(status_code=304, headers=headers)
        if request.method == "HEAD":
            headers["Content-Length"] = str(len(content))
            return Response(headers=headers, media_type="application/json")
        return Response(
            content=content,
            media_type="application/json",
            headers=headers,
        )

    @app.get("/admin/login", include_in_schema=False)
    def login_page(request: Request):
        if _admin_username(request):
            return RedirectResponse(url="/admin/", status_code=303)
        return TEMPLATES.TemplateResponse(
            request,
            "login.html",
            _context(
                request,
                setup_required=not container.service.admin_is_configured(),
                error=None,
            ),
        )

    @app.post("/admin/login", include_in_schema=False)
    async def login(request: Request):
        form = await request.form()
        username = str(form.get("username", "")).strip()
        password = str(form.get("password", ""))
        if not validate_csrf(request.session, str(form.get("csrf_token", ""))):
            raise HTTPException(status_code=403, detail="CSRF 校验失败。")
        host = _client_host(request)
        if not container.login_limiter.is_allowed(host, username):
            return TEMPLATES.TemplateResponse(
                request,
                "login.html",
                _context(
                    request,
                    setup_required=not container.service.admin_is_configured(),
                    error="登录尝试过多，请稍后再试。",
                ),
                status_code=429,
            )
        with container.service.database.session() as session:
            authenticated = verify_password(session, username, password)
        if not authenticated:
            container.login_limiter.record_failure(host, username)
            container.service.record_audit(username or None, "login_failed", host)
            return TEMPLATES.TemplateResponse(
                request,
                "login.html",
                _context(
                    request,
                    setup_required=not container.service.admin_is_configured(),
                    error="账号或密码错误。",
                ),
                status_code=401,
            )
        container.login_limiter.clear(host, username)
        request.session.clear()
        request.session["admin_username"] = username
        csrf_token(request.session)
        container.service.record_audit(username, "login_succeeded", host)
        return RedirectResponse(url="/admin/", status_code=303)

    @app.post("/admin/logout", include_in_schema=False)
    async def logout(request: Request):
        actor = _admin_username(request)
        if actor is None:
            return RedirectResponse(url="/admin/login", status_code=303)
        form = await request.form()
        if not validate_csrf(request.session, str(form.get("csrf_token", ""))):
            raise HTTPException(status_code=403, detail="CSRF 校验失败。")
        request.session.clear()
        container.service.record_audit(actor, "logout", None)
        return RedirectResponse(url="/admin/login", status_code=303)

    @app.get("/admin/", include_in_schema=False)
    def dashboard(request: Request, days: int | None = Query(None, ge=1, le=5000), view: str = "overview"):
        redirect = _admin_redirect(request)
        if redirect:
            return redirect
        job_id = request.query_params.get("job")
        shadow_run_id = request.query_params.get("shadow")
        selected_days = days if days is not None else int(request.session.get("statistics_days", 60))
        selected_days = max(1, min(5000, selected_days))
        request.session["statistics_days"] = selected_days
        if view not in {"overview", "history", "jobs", "research"}:
            view = "overview"
        error = None
        try:
            data = container.service.dashboard_data(selected_days)
        except (ServiceNotReadyError, PredictionDriftError) as exc:
            data = None
            error = str(exc)
        job = container.refresh_manager.get(job_id) if job_id else None
        shadow_run = container.shadow_manager.get(shadow_run_id) if shadow_run_id else None
        with container.service.database.session() as session:
            jobs = session.scalars(select(RefreshJob).order_by(RefreshJob.created_at.desc()).limit(20)).all()
        return TEMPLATES.TemplateResponse(
            request,
            "dashboard.html",
            _context(
                request,
                data=data,
                job=job,
                shadow_run=shadow_run,
                service_ready=data is not None,
                error=error, days=selected_days, view=view, jobs=jobs,
                scheduled=runtime_settings.scheduled_refresh_enabled,
                active_jobs=[item.id for item in jobs if item.status in {"queued", "running"}],
            ),
        )

    @app.get("/admin/models", include_in_schema=False)
    def models_page(request: Request, days: int | None = Query(None, ge=1, le=5000),
                    sample: Literal["all", "live"] = "all"):
        redirect = _admin_redirect(request)
        if redirect:
            return redirect
        selected_days = days if days is not None else max(1, min(5000, int(request.session.get("statistics_days", 60))))
        request.session["statistics_days"] = selected_days
        error = None
        try:
            comparison = comparison_data(container.service, selected_days, sample)
        except (ServiceNotReadyError, PredictionDriftError) as exc:
            comparison, error = None, str(exc)
        return TEMPLATES.TemplateResponse(request, "models.html", _context(
            request, view="models", comparison=comparison, days=selected_days, sample=sample, error=error,
        ))

    @app.get("/admin/api/models", include_in_schema=False)
    def models_api(request: Request, days: int = Query(60, ge=1, le=5000),
                   sample: Literal["all", "live"] = "all"):
        redirect = _admin_redirect(request)
        if redirect:
            return redirect
        return comparison_data(container.service, days, sample)

    @app.get("/admin/models/comparison.csv", include_in_schema=False)
    def comparison_download(request: Request, days: int = Query(60, ge=1, le=5000),
                            sample: Literal["all", "live"] = "all"):
        redirect = _admin_redirect(request)
        if redirect:
            return redirect
        data = comparison_data(container.service, days, sample)
        if data is None:
            raise HTTPException(status_code=503, detail="模型组合尚未激活。")
        return Response(frame_to_csv_bytes(comparison_frame(data)), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="model_comparison_{sample}_{days}.csv"',
                                 "Cache-Control": "no-store"})

    @app.get("/admin/models/{model_key}/latest.csv", include_in_schema=False)
    def model_download(request: Request, model_key: str):
        redirect = _admin_redirect(request)
        if redirect:
            return redirect
        return Response(model_csv(container.service, model_key), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{model_key}_latest.csv"',
                                 "Cache-Control": "no-store"})

    @app.post("/admin/refresh", include_in_schema=False)
    async def refresh(request: Request):
        actor = _admin_username(request)
        if actor is None:
            return RedirectResponse(url="/admin/login", status_code=303)
        form = await request.form()
        if not validate_csrf(request.session, str(form.get("csrf_token", ""))):
            raise HTTPException(status_code=403, detail="CSRF 校验失败。")
        job_id, created = container.refresh_manager.submit(trigger="manual", actor=actor)
        container.service.record_audit(
            actor,
            "refresh_requested" if created else "refresh_joined_existing_job",
            job_id,
        )
        return RedirectResponse(url=f"/admin/?view=jobs&job={job_id}", status_code=303)

    @app.post("/admin/promote", include_in_schema=False)
    async def promote(request: Request):
        actor = _admin_username(request)
        if actor is None:
            return RedirectResponse(url="/admin/login", status_code=303)
        form = await request.form()
        if not validate_csrf(request.session, str(form.get("csrf_token", ""))):
            raise HTTPException(status_code=403, detail="CSRF 校验失败。")
        job_id, _ = container.refresh_manager.submit(trigger="promotion", actor=actor)
        return RedirectResponse(url=f"/admin/?view=jobs&job={job_id}", status_code=303)

    @app.post("/admin/shadow/bilstm", include_in_schema=False)
    async def run_bilstm_shadow(request: Request):
        actor = _admin_username(request)
        if actor is None:
            return RedirectResponse(url="/admin/login", status_code=303)
        form = await request.form()
        if not validate_csrf(request.session, str(form.get("csrf_token", ""))):
            raise HTTPException(status_code=403, detail="CSRF 校验失败。")
        run_id, created = container.submit_active_shadow(actor=actor, force=True)
        if run_id is None:
            raise HTTPException(status_code=503, detail="当前没有可用于影子计算的发布快照。")
        container.service.record_audit(
            actor,
            "bilstm_shadow_requested" if created else "bilstm_shadow_joined_existing_run",
            run_id,
        )
        return RedirectResponse(url=f"/admin/?view=research&shadow={run_id}", status_code=303)

    @app.get("/admin/jobs/{job_id}", include_in_schema=False)
    def job_status(request: Request, job_id: str) -> Response:
        redirect = _admin_redirect(request)
        if redirect:
            return redirect
        job = container.refresh_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="找不到刷新任务。")
        return JSONResponse(
            {
                "id": job.id,
                "status": job.status,
                "snapshot_id": job.snapshot_id,
                "message": job.message,
                "created_at": job.created_at.isoformat(),
                "finished_at": job.finished_at.isoformat() if job.finished_at else None,
            }
        )

    @app.get("/admin/shadow-runs/{run_id}", include_in_schema=False)
    def shadow_status(request: Request, run_id: str) -> Response:
        redirect = _admin_redirect(request)
        if redirect:
            return redirect
        run = container.shadow_manager.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="找不到 BiLSTM 影子任务。")
        return JSONResponse(
            {
                "id": run.id,
                "engine": run.engine,
                "status": run.status,
                "snapshot_id": run.snapshot_id,
                "message": run.message,
                "created_at": run.created_at.isoformat(),
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            }
        )

    @app.get("/admin/shadow-runs/{run_id}/results.csv", include_in_schema=False)
    def shadow_result_download(request: Request, run_id: str):
        redirect = _admin_redirect(request)
        if redirect:
            return redirect
        path = container.service.shadow_result_file(run_id)
        return FileResponse(
            path,
            media_type="text/csv",
            filename=f"bilstm_shadow_{run_id}.csv",
        )

    @app.get("/admin/archives", include_in_schema=False)
    def archives(request: Request):
        redirect = _admin_redirect(request)
        if redirect:
            return redirect
        return TEMPLATES.TemplateResponse(
            request,
            "archives.html",
            _context(request, archives=container.service.list_archives()),
        )

    @app.get("/admin/archives/{snapshot_id}/{filename}", include_in_schema=False)
    def archive_download(request: Request, snapshot_id: str, filename: str):
        redirect = _admin_redirect(request)
        if redirect:
            return redirect
        path = container.service.archive_file(snapshot_id, filename)
        media_type = "application/json" if filename.endswith(".json") else "text/csv"
        return FileResponse(path, media_type=media_type, filename=filename)

    return app
