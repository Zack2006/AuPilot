"""Local-only data-source and credential settings routes."""

from fastapi import APIRouter, status

from backend.app.api.dependencies import local_settings_service, market_service
from backend.app.schemas.runtime_settings import (
    CredentialDeleteResponse,
    DataSourceBatchUpdate,
    DataSourceSettingsResponse,
    DataSourceUpdate,
    DataSourceVerificationResponse,
)


router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/sources", response_model=DataSourceSettingsResponse)
def sources() -> DataSourceSettingsResponse:
    return local_settings_service().list_sources()


@router.put("/sources", response_model=DataSourceSettingsResponse)
def update_sources(
    payload: DataSourceBatchUpdate,
) -> DataSourceSettingsResponse:
    return local_settings_service().update_sources(payload.sources)


@router.put("/sources/{source_id}", response_model=DataSourceSettingsResponse)
def update_source(source_id: str, payload: DataSourceUpdate) -> DataSourceSettingsResponse:
    return local_settings_service().update_source(source_id, payload)


@router.delete(
    "/sources/{source_id}/credential",
    response_model=CredentialDeleteResponse,
    status_code=status.HTTP_200_OK,
)
def delete_credential(source_id: str) -> CredentialDeleteResponse:
    return local_settings_service().delete_credential(source_id)


@router.post("/sources/{source_id}/verify", response_model=DataSourceVerificationResponse)
def verify_source(source_id: str) -> DataSourceVerificationResponse:
    if source_id != "databento":
        raise ValueError("Automated verification is not implemented for this data source")
    return DataSourceVerificationResponse(**market_service().verify_access())
