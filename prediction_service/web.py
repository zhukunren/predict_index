"""FastAPI application exposing public CSV data and a password-protected panel."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .config import Settings
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
    return "-" if value is None else f"{value:.2%}"


TEMPLATES.env.filters["percent"] = _as_percent


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
        container.service.initialize(bootstrap=bootstrap)
        container.scheduler.start()
        if runtime_settings.bilstm_shadow_enabled:
            try:
                container.submit_active_shadow(actor="system")
            except ServiceNotReadyError:
                pass
        try:
            yield
        finally:
            container.scheduler.stop()
            container.refresh_manager.shutdown()
            container.shadow_manager.shutdown()
            container.service.shutdown()

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
    app.mount("/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static")
    app.state.container = container

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/admin/", status_code=303)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        try:
            publication, snapshot, _ = container.service._load_active_context()
        except ServiceNotReadyError:
            return {"status": "starting"}
        return {
            "status": "ok",
            "snapshot_id": snapshot.id,
            "data_as_of": snapshot.data_as_of,
            "publication_id": publication.id,
        }

    @app.get("/api/v1/sh000001/latest.csv")
    def latest_csv() -> Response:
        try:
            artifact = container.service.recompute_active()
        except (ServiceNotReadyError, ModelReleaseMismatchError, PredictionDriftError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        headers = {
            "Content-Disposition": 'attachment; filename="sh000001_latest.csv"',
            "ETag": f'"{artifact.csv_sha256}"',
            "X-Snapshot-Id": artifact.snapshot_id,
            "X-Model-Release": artifact.release_id,
            "X-Data-As-Of": artifact.data_as_of,
            "Cache-Control": "no-cache",
        }
        return Response(
            content=artifact.csv_bytes,
            media_type="text/csv; charset=utf-8",
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
    def dashboard(request: Request):
        redirect = _admin_redirect(request)
        if redirect:
            return redirect
        job_id = request.query_params.get("job")
        shadow_run_id = request.query_params.get("shadow")
        try:
            data = container.service.dashboard_data()
        except ServiceNotReadyError:
            data = None
        job = container.refresh_manager.get(job_id) if job_id else None
        shadow_run = container.shadow_manager.get(shadow_run_id) if shadow_run_id else None
        return TEMPLATES.TemplateResponse(
            request,
            "dashboard.html",
            _context(
                request,
                data=data,
                job=job,
                shadow_run=shadow_run,
                service_ready=data is not None,
            ),
        )

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
        return RedirectResponse(url=f"/admin/?job={job_id}", status_code=303)

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
        return RedirectResponse(url=f"/admin/?shadow={run_id}", status_code=303)

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
