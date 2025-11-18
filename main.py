from datetime import datetime, timedelta
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlmodel import Field, Session, SQLModel, create_engine, select
from passlib.context import CryptContext
from jose import JWTError, jwt

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from fastapi_csrf_protect import CsrfProtect
from config import CsrfSettings

import os

load_dotenv('.env.local')

# Security Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


# Models
class UserBase(SQLModel):
    username: str = Field(index=True, unique=True)
    email: str = Field(index=True, unique=True)
    tanggal_lahir: datetime | None = None
    jenis_kelamin: str = Field(description="L/P")
    tinggi_badan: float | None = Field(default=None, description="Tinggi badan dalam cm")
    berat_badan: float | None = Field(default=None, description="Berat badan dalam kg")
    alergi: str | None = Field(default=None, description="Alergi terhadap")
    catatan_medis: str | None = Field(default=None, description="Catatan medis anda")


class User(UserBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    password: str  # Hashed password


class UserPublic(UserBase):
    id: int


class UserCreate(UserBase):
    password: str


class UserLogin(SQLModel):
    email: str
    password: str


class Token(SQLModel):
    access_token: str
    token_type: str
    user: UserPublic


class UserUpdate(SQLModel):
    username: str | None = None
    email: str | None = None
    tanggal_lahir: datetime | None = None
    jenis_kelamin: str | None = None
    tinggi_badan: float | None = None
    berat_badan: float | None = None
    alergi: str | None = None
    catatan_medis: str | None = None


# Database Configuration - MySQL
MYSQL_USER = os.environ.get("MYSQL_USER")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD")
MYSQL_HOST = os.environ.get("MYSQL_HOST")
MYSQL_PORT = os.environ.get("MYSQL_PORT")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE")

mysql_url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
engine = create_engine(mysql_url, echo=True)


def create_db_and_tables():
    """Membuat tabel otomatis berdasarkan model SQLModel"""
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

app = FastAPI(title="User Management API")

templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
def on_startup():
    """Migrasi otomatis saat aplikasi startup"""
    create_db_and_tables()


# Helper functions
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        session: Session = Depends(get_session)
) -> User:
    """Dependency untuk mendapatkan user yang sedang login dari token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = session.get(User, user_id)
    if user is None:
        raise credentials_exception
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


# Auth Endpoints
@app.post("/register", response_model=UserPublic, status_code=201)
def register(user: UserCreate, session: SessionDep):
    """Register user baru"""
    # Check if username exists
    existing_user = session.exec(
        select(User).where(User.username == user.username)
    ).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username sudah digunakan")

    # Check if email exists
    existing_email = session.exec(
        select(User).where(User.email == user.email)
    ).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="Email sudah digunakan")

    # Create user with hashed password
    user_data = user.model_dump()
    user_data["password"] = hash_password(user_data["password"])
    db_user = User.model_validate(user_data)

    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


@app.post("/login", response_model=Token)
def login(user_login: UserLogin, session: SessionDep):
    """Login dengan email dan password"""
    # Cari user berdasarkan email
    user = session.exec(
        select(User).where(User.email == user_login.email)
    ).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email atau password salah"
        )

    # Verify password
    if not verify_password(user_login.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email atau password salah"
        )

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=access_token_expires
    )

    # Return token dan data user
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserPublic.model_validate(user)
    )


@app.get("/me", response_model=UserPublic)
def get_current_user_info(current_user: CurrentUser):
    """Get informasi user yang sedang login"""
    return current_user


# User CRUD Endpoints
@app.get("/users/", response_model=list[UserPublic])
def read_users(
        session: SessionDep,
        current_user: CurrentUser,  # Harus login untuk akses endpoint ini
        offset: int = 0,
        limit: Annotated[int, Query(le=100)] = 100,
):
    """Get semua users (requires authentication)"""
    users = session.exec(select(User).offset(offset).limit(limit)).all()
    return users


@app.get("/users/{user_id}", response_model=UserPublic)
def read_user(user_id: int, session: SessionDep, current_user: CurrentUser):
    """Get user by ID (requires authentication)"""
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")
    return user


@app.patch("/users/{user_id}", response_model=UserPublic)
def update_user(
        user_id: int,
        user: UserUpdate,
        session: SessionDep,
        current_user: CurrentUser
):
    """Update user (requires authentication)"""
    # User hanya bisa update data dirinya sendiri, kecuali admin
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Anda hanya bisa update data diri sendiri"
        )

    db_user = session.get(User, user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")

    user_data = user.model_dump(exclude_unset=True)

    # Hash password if updated
    if "password" in user_data and user_data["password"]:
        user_data["password"] = hash_password(user_data["password"])

    db_user.sqlmodel_update(user_data)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


@app.delete("/users/{user_id}")
def delete_user(
        user_id: int,
        session: SessionDep,
        current_user: CurrentUser
):
    """Delete user (requires authentication)"""
    # User hanya bisa delete dirinya sendiri
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Anda hanya bisa delete akun sendiri"
        )

    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")
    session.delete(user)
    session.commit()
    return {"ok": True, "message": "User berhasil dihapus"}

@app.get("/register-page")
def register_page(request: Request):
    return templates.TemplateResponse("auth/register/index.html", {"request": request})

@app.get("/login-page")
def login_page(request: Request):
    return templates.TemplateResponse("auth/login/index.html", {"request": request})