from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from auth.deps import get_current_user
from database.db import get_db
from database.models import User
from services.portfolio_risk import resolve_max_open_r  # noqa: F401 — canonical home is services/portfolio_risk.py; re-exported here so existing `from api.settings import ...` imports keep working

router = APIRouter(prefix="/api/settings", tags=["settings"])


# resolve_max_open_r now lives in services/portfolio_risk.py (shared by the
# API and services layers without a services->api import); it is
# re-exported at this module's top for existing import sites.


class SettingsResponse(BaseModel):
    account_size: float
    risk_pct: float
    max_positions: int
    atr_stop_mult: float
    r_multiple: float
    # NULL = "not chosen": the open-R budget is then DERIVED as
    # max_positions x risk_pct. `max_open_r_basis` says which one is in play so
    # a derived number is never presented as a deliberate ceiling.
    max_open_r: float | None = None
    max_open_r_effective: float | None = None
    max_open_r_basis: str = ""
    use_evidence_regimes: bool = False


class SettingsUpdate(BaseModel):
    account_size: float | None = Field(default=None, gt=0)
    risk_pct: float | None = Field(default=None, ge=0.1, le=5)
    max_positions: int | None = Field(default=None, ge=1, le=50)
    atr_stop_mult: float | None = Field(default=None, ge=0.5, le=6)
    r_multiple: float | None = Field(default=None, ge=0.5, le=10)
    # Sent as null to CLEAR the chosen ceiling and go back to the derived
    # budget — so this one is read from model_fields_set, not exclude_none.
    max_open_r: float | None = Field(default=None, ge=0.1, le=100)
    use_evidence_regimes: bool | None = None


def _to_response(user: User) -> SettingsResponse:
    chosen, basis = resolve_max_open_r(user)
    return SettingsResponse(
        account_size=user.account_size,
        risk_pct=user.risk_pct,
        max_positions=user.max_positions,
        atr_stop_mult=user.atr_stop_mult,
        r_multiple=user.r_multiple,
        max_open_r=getattr(user, "max_open_r", None),
        max_open_r_effective=chosen,
        max_open_r_basis=basis,
        use_evidence_regimes=bool(getattr(user, "use_evidence_regimes", False)),
    )


@router.get("", response_model=SettingsResponse)
def get_settings_endpoint(user: User = Depends(get_current_user)):
    return _to_response(user)


@router.put("", response_model=SettingsResponse)
def update_settings_endpoint(
    req: SettingsUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    sent = req.model_fields_set
    for field, value in req.model_dump(exclude_none=True).items():
        setattr(user, field, value)
    # `max_open_r` is nullable on purpose: an explicit null means "stop using a
    # chosen ceiling, go back to the derived budget", which exclude_none above
    # would silently swallow.
    if "max_open_r" in sent:
        user.max_open_r = req.max_open_r
    db.commit()
    db.refresh(user)
    return _to_response(user)
