class AurumPilotError(Exception):
    """Base class for expected application errors."""


class ModelUnavailableError(AurumPilotError):
    """Raised when a configured prediction model cannot be loaded."""


class RAGUnavailableError(AurumPilotError):
    """Raised when the future RAG adapter has not been configured."""


class MarketDataUnavailableError(AurumPilotError):
    """Raised when verified Databento gold data is unavailable."""


class TechnicalSidecarUnavailableError(AurumPilotError):
    """Raised when the verified V4 model sidecar cannot return a valid response."""


class TechnicalIssuanceUnavailableError(AurumPilotError):
    """Raised when no valid persisted V4 technical issuance is available."""


class FormalValidationUnavailableError(AurumPilotError):
    """Raised when formal out-of-sample validation inputs are unavailable."""


class NotFoundError(AurumPilotError):
    """Raised when a requested file-backed record does not exist."""


class DataCorruptionError(AurumPilotError):
    """Raised when persisted JSON exists but cannot be decoded safely."""
