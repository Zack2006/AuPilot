"""Controlled internal publication boundary for ex-ante dual-model predictions."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd

from backend.app.core.config import Settings
from backend.app.core.exceptions import FormalValidationUnavailableError
from backend.app.repositories.formal_prediction_repository import FormalPredictionRepository
from backend.app.schemas.formal_model import FormalDualModelOutput, FormalPredictionCreate, FormalPredictionRecord
from backend.app.services.formal_model_registry import FormalModelRegistry
from backend.app.services.market_data_service import MarketDataService


class FormalPredictionPublicationService:
    """Attach trusted runtime identity to model output and append it without historical overwrite."""

    def __init__(
        self,
        settings: Settings,
        market_data: MarketDataService,
        *,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.registry = FormalModelRegistry(settings)
        self.predictions = FormalPredictionRepository(
            settings.storage_dir / "predictions" / "formal_predictions.jsonl"
        )
        self.market_data = market_data
        self.clock = clock or (lambda: datetime.now(UTC))
        self.id_factory = id_factory or (lambda: uuid4().hex[:12])

    def publish(self, output: FormalDualModelOutput) -> FormalPredictionRecord:
        manifest, manifest_sha256, reason = self.registry.validated_manifest()
        if manifest is None or manifest_sha256 is None:
            raise FormalValidationUnavailableError(reason or "FORMAL_MODEL_MANIFEST_INVALID")

        generated_at = self.clock()
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise ValueError("Publication clock must be timezone-aware")
        generated_at = generated_at.astimezone(UTC)
        if generated_at < manifest.activated_at_utc:
            raise FormalValidationUnavailableError("FORMAL_MODEL_NOT_ACTIVATED")
        if output.prediction_for_date <= generated_at.date():
            raise ValueError("Formal predictions must be published before the target UTC date")
        if output.prediction_for_date < manifest.out_of_sample_start_date:
            raise ValueError("Prediction date is before the manifest out-of-sample boundary")

        market = self.market_data.get_history(refresh=False)
        latest_bar = market.frame.iloc[-1]
        market_as_of = pd.Timestamp(latest_bar["ts_event_utc"])
        if market_as_of.tzinfo is None:
            raise FormalValidationUnavailableError("DATABENTO_MARKET_TIMESTAMP_INVALID")
        market_as_of_utc = market_as_of.tz_convert(UTC).to_pydatetime()
        if market_as_of_utc <= manifest.training_data_cutoff_utc:
            raise FormalValidationUnavailableError("FORMAL_MARKET_DATA_NOT_AFTER_TRAINING_CUTOFF")
        if output.prediction_for_date <= market_as_of_utc.date():
            raise ValueError("Prediction target must be after the current Databento market snapshot")
        if market_as_of_utc > generated_at:
            raise FormalValidationUnavailableError("DATABENTO_MARKET_TIMESTAMP_IN_FUTURE")

        existing = self.predictions.read_all()
        same_date = [item for item in existing if item.prediction_for_date == output.prediction_for_date]
        if same_date and output.revision_reason is None:
            raise ValueError("A same-date formal revision requires a revision_reason")
        if not same_date and output.revision_reason is not None:
            raise ValueError("revision_reason is only valid for an existing prediction date")
        supersedes = same_date[-1].prediction_id if same_date else None

        turning_sha = manifest.artifact("turning_point").sha256
        price_sha = manifest.artifact("price").sha256
        prediction = FormalPredictionCreate(
            prediction_id=f"pred_{output.prediction_for_date:%Y%m%d}_{self.id_factory()}",
            generated_at_utc=generated_at,
            prediction_for_date=output.prediction_for_date,
            market_data_as_of_utc=market_as_of_utc,
            training_data_cutoff_utc=manifest.training_data_cutoff_utc,
            model_version=manifest.model_version,
            feature_schema_version=manifest.feature_schema_version,
            model_manifest_sha256=manifest_sha256,
            turning_point_model_sha256=turning_sha,
            price_model_sha256=price_sha,
            market_data_sha256=market.metadata.content_sha256,
            code_sha=manifest.code_sha,
            predicted_close=output.predicted_close,
            predicted_low=output.predicted_low,
            predicted_high=output.predicted_high,
            expected_direction=output.expected_direction,
            direction_confidence=output.direction_confidence,
            turning_point=output.turning_point,
            turning_point_confidence=output.turning_point_confidence,
            target_exposure=output.target_exposure,
            supersedes_prediction_id=supersedes,
            revision_reason=output.revision_reason,
        )
        return self.predictions.append(prediction)
