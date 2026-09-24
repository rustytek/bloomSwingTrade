from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from auth.deps import get_current_user
from database.db import get_db
from database.models import User
from services.weekly_plan import build_weekly_plan

router = APIRouter(prefix="/api/weekly-plan", tags=["weekly-plan"])


@router.get("")
async def weekly_plan(
    force: bool = Query(False, description="Rebuild the underlying Playbook payload (bypass its 15-minute cache)"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """The guided weekly check: regime -> positions -> new trades -> orders.

    `orders[]` is the contract the Trade page consumes: stable ids
    ("buy:AAPL" / "sell:MSFT" / "trim:NVDA"), whole-share limit orders, a
    `recommended` flag with a `skip_reason` when false, a plain-English `why[]`,
    and the `plan` intent to persist on fill.
    """
    return await build_weekly_plan(db, user, force=force)
