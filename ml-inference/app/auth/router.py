from typing import Optional
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, EmailStr

from .db import get_db, init_db
from .security import hash_password, verify_password, create_token, decode_token

router = APIRouter(prefix="/auth", tags=["auth"])

init_db()


class RegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    role: str
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    hospital_name: Optional[str] = None
    license_number: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UpdateProfileRequest(BaseModel):
    full_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    hospital_name: Optional[str] = None
    license_number: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


def _user_public(row) -> dict:
    return {
        "id": row["id"], "full_name": row["full_name"], "email": row["email"], "role": row["role"],
        "age": row["age"], "gender": row["gender"], "phone": row["phone"],
        "hospital_name": row["hospital_name"], "license_number": row["license_number"],
    }


@router.post("/register")
def register(req: RegisterRequest):
    if req.role not in ("doctor", "patient"):
        raise HTTPException(status_code=400, detail="role must be 'doctor' or 'patient'")

    with get_db() as conn:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (req.email,)).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail="An account with this email already exists")

        cursor = conn.execute(
            """INSERT INTO users (full_name, email, password_hash, role, age, gender, phone, hospital_name, license_number)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (req.full_name, req.email, hash_password(req.password), req.role, req.age, req.gender, req.phone, req.hospital_name, req.license_number),
        )
        user_id = cursor.lastrowid

    token = create_token(user_id, req.role)
    return {"token": token, "user": {"id": user_id, "full_name": req.full_name, "email": req.email, "role": req.role}}


@router.post("/login")
def login(req: LoginRequest):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (req.email,)).fetchone()

    if not row or not verify_password(req.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token(row["id"], row["role"])
    return {"token": token, "user": _user_public(row)}


def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = decode_token(token)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (payload["sub"],)).fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="User no longer exists")

    return _user_public(row)


@router.get("/me")
def me(authorization: Optional[str] = Header(None)):
    return get_current_user(authorization)


@router.patch("/me")
def update_me(req: UpdateProfileRequest, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    updates = {k: v for k, v in req.model_dump().items() if v is not None}

    if updates:
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        with get_db() as conn:
            conn.execute(f"UPDATE users SET {set_clause} WHERE id = ?", (*updates.values(), user["id"]))
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
        return _user_public(row)

    return user


@router.post("/change-password")
def change_password(req: ChangePasswordRequest, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)

    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
        if not verify_password(req.current_password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Current password is incorrect")

        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(req.new_password), user["id"]),
        )

    return {"status": "password updated"}