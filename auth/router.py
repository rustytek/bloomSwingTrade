from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from pydantic import BaseModel
from database.db import get_db
from database.models import User
from auth.utils import verify_password, hash_password, create_access_token
from auth.deps import get_current_user, get_current_admin

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Schemas ─────────────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str
    is_admin: bool


class UserResponse(BaseModel):
    id: int
    username: str
    email: str | None
    is_admin: bool
    created_at: datetime
    last_login: datetime | None
    has_litellm_key: bool = False

    class Config:
        from_attributes = True


class CreateUserRequest(BaseModel):
    username: str
    password: str
    email: str | None = None
    is_admin: bool = False
    litellm_api_key: str | None = None


class UpdateUserRequest(BaseModel):
    password: str | None = None
    email: str | None = None
    is_admin: bool | None = None
    litellm_api_key: str | None = None   # set the per-user LiteLLM key
    clear_litellm_key: bool = False      # set true to remove an existing key


# ── Routes ───────────────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form.username).first()
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    user.last_login = datetime.now(timezone.utc)
    db.commit()

    token = create_access_token({"sub": str(user.id)})
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        username=user.username,
        is_admin=user.is_admin,
    )


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)):
    return current_user


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(
    req: CreateUserRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")

    user = User(
        username=req.username,
        password_hash=hash_password(req.password),
        email=req.email,
        is_admin=req.is_admin,
        litellm_api_key=(req.litellm_api_key or None),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.get("/users", response_model=list[UserResponse])
def list_users(db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    return db.query(User).all()


@router.put("/users/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    req: UpdateUserRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if req.password is not None:
        user.password_hash = hash_password(req.password)
    if req.email is not None:
        user.email = req.email
    if req.is_admin is not None:
        user.is_admin = req.is_admin
    if req.clear_litellm_key:
        user.litellm_api_key = None
    elif req.litellm_api_key:
        user.litellm_api_key = req.litellm_api_key

    db.commit()
    db.refresh(user)
    return user


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    db.delete(user)
    db.commit()
