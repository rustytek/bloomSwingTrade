from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from auth.deps import get_current_user
from database.db import get_db
from database.models import User
from services.today import build_today

router = APIRouter(prefix="/api/today", tags=["today"])


@router.get("")
async def today(
    force: bool = Query(False, description="Bypass the 15-minute cache"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return await build_today(db, user.id, force=force)
