from fastapi import APIRouter

from backend.app.api.dependencies import cycle_service
from backend.app.schemas.cycle import TacticalCycle, TacticalCycleCreate, TacticalCycleUpdate

router = APIRouter(prefix="/cycles", tags=["cycles"])


@router.get("", response_model=list[TacticalCycle])
def list_cycles() -> list[TacticalCycle]:
    return cycle_service().list_cycles()


@router.get("/current", response_model=TacticalCycle | None)
def current_cycle() -> TacticalCycle | None:
    return cycle_service().current()


@router.post("", response_model=TacticalCycle, status_code=201)
def create_cycle(payload: TacticalCycleCreate) -> TacticalCycle:
    return cycle_service().create(payload)


@router.patch("/{cycle_id}", response_model=TacticalCycle)
def update_cycle(cycle_id: str, payload: TacticalCycleUpdate) -> TacticalCycle:
    return cycle_service().update(cycle_id, payload)
