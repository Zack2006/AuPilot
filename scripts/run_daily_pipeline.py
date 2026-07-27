"""Run one deterministic recommendation pipeline using the demo portfolio."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.api.dependencies import recommendation_orchestrator
from backend.app.core.config import get_settings
from backend.app.schemas.portfolio import PortfolioInput

if __name__ == "__main__":
    settings = get_settings()
    payload = json.loads((settings.storage_dir / "users" / "demo_portfolio.json").read_text(encoding="utf-8"))
    result = recommendation_orchestrator().generate(PortfolioInput.model_validate(payload))
    print(result.model_dump_json(indent=2))
