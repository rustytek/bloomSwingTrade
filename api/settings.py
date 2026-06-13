from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from auth.deps import get_current_user
from database.db import get_db
from database.models import User

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsResponse(BaseModel):
    account_size: float
    risk_pct: float
    max_positions: int
    atr_stop_mult: float
    r_multiple: float


class SettingsUpdate(BaseModel):
    account_size: float | None = Field(default=None, gt=0)
    risk_pct: float | None = Field(default=None, ge=0.1, le=5)
    max_positions: int | None = Field(default=None, ge=1, le=25)
    atr_stop_mult: float | None = Field(default=None, ge=0.5, le=6)
    r_multiple: float | None = Field(default=None, ge=0.5, le=10)


def _to_response(user: User) -> SettingsResponse:
    return SettingsResponse(
        account_size=user.account_size,
        risk_pct=user.risk_pct,
        max_positions=user.max_positions,
        atr_stop_mult=user.atr_stop_mult,
        r_multiple=user.r_multiple,
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
    for field, value in req.model_dump(exclude_none=True).items():
        setattr(user, field, value)
    db.commit()
    db.refresh(user)
    return _to_response(user)
