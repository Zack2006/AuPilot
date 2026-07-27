"""FastAPI composition root for AurumPilot. / AurumPilot FastAPI 组合根。

Purpose / 文件用途: assemble the modular monolith, versioned API, CORS and error mapping.
Inputs / 输入: environment settings plus HTTP requests containing validated Pydantic payloads.
Outputs / 输出: JSON-only analysis, ledger, snapshot, forecast and analytics APIs.
Business invariants / 业务约束: no broker integration, order execution, shorting or leverage endpoints exist.
Side effects / 副作用: domain services may append settings, transaction, forecast and audit files.
Fallback behavior / 降级: formal market, model and RAG paths fail explicitly; no Mock data substitutes formal output.
"""

from __future__ import annotations

import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api import advice, cycles, health, macro, market, model_validation, portfolio, runtime_settings, technical, transactions
from backend.app.core.config import get_settings
from backend.app.core.exceptions import AurumPilotError, DataCorruptionError, NotFoundError
from backend.app.core.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize a missing market cache once; normal restarts stay offline."""
    logger.info("Starting %s in %s mode", settings.app_name, settings.app_env)
    from backend.app.api.dependencies import market_service

    try:
        created = await asyncio.to_thread(market_service().ensure_initial_cache)
        if created:
            logger.info("Published the initial verified Databento daily cache")
    except Exception as exc:
        logger.warning("Initial Databento cache remains unavailable: %s", type(exc).__name__)
    yield
    logger.info("Stopping %s", settings.app_name)


app = FastAPI(
    title=f"{settings.app_name} API", version="2.0.0",
    description="Local gold technical outlook and independent official macro risk; no order execution.", lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=[origin.strip() for origin in settings.frontend_origin.split(",")],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

api = APIRouter(prefix="/api/v1")
for router in (
    health.router, market.router, portfolio.router, transactions.router,
    macro.router, cycles.router, runtime_settings.router, model_validation.router,
    technical.router, advice.router,
):
    api.include_router(router)
app.include_router(api)


@app.exception_handler(AurumPilotError)
async def application_error_handler(_: Request, exc: AurumPilotError) -> JSONResponse:
    status = 404 if isinstance(exc, NotFoundError) else 500 if isinstance(exc, DataCorruptionError) else 503
    return JSONResponse(status_code=status, content={"error": exc.__class__.__name__, "detail": str(exc)})


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": "InvalidOperation", "detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    if request.url.path.startswith("/api/v1/settings/"):
        return JSONResponse(
            status_code=422,
            content={"error": "InvalidSettingsRequest", "detail": "The settings request is invalid."},
        )
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})

