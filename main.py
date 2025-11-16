from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select
from passlib.context import CryptContext

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Models
class UserBase(SQLModel):
    username: str = Field(index=True, unique=True)
    email: str = Field(index=True)
    nama_lengkap: str
    tinggi_badan: float | None = Field(default=None, description="Tinggi badan dalam cm")
    berat_badan: float | None = Field(default=None, description="Berat badan dalam kg")
    tanggal_lahir: str | None = None
    jenis_kelamin: str | None = Field(default=None, description="L/P")


class User(UserBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    password: str  # Hashed password


class UserPublic(UserBase):
    id: int


class UserCreate(UserBase):
    password: str


class UserUpdate(SQLModel):
    username: str | None = None
    email: str | None = None
    nama_lengkap: str | None = None
    tinggi_badan: float | None = None
    berat_badan: float | None = None
    tanggal_lahir: str | None = None
    jenis_kelamin: str | None = None
    password: str | None = None


# Database Configuration - MySQL
# Format: mysql+pymysql://username:password@host:port/database
MYSQL_USER = "root"
MYSQL_PASSWORD = "password"
MYSQL_HOST = "localhost"
MYSQL_PORT = "3306"
MYSQL_DATABASE = "user_db"

mysql_url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
engine = create_engine(mysql_url, echo=True)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

app = FastAPI(title="User Management API")


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


# Helper function
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


# Endpoints
@app.post("/users/", response_model=UserPublic, status_code=201)
def create_user(user: UserCreate, session: SessionDep):
    # Check if username exists
    existing_user = session.exec(
        select(User).where(User.username == user.username)
    ).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username sudah digunakan")

    # Create user with hashed password
    user_data = user.model_dump()
    user_data["password"] = hash_password(user_data["password"])
    db_user = User.model_validate(user_data)

    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


@app.get("/users/", response_model=list[UserPublic])
def read_users(
        session: SessionDep,
        offset: int = 0,
        limit: Annotated[int, Query(le=100)] = 100,
):
    users = session.exec(select(User).offset(offset).limit(limit)).all()
    return users


@app.get("/users/{user_id}", response_model=UserPublic)
def read_user(user_id: int, session: SessionDep):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")
    return user


@app.patch("/users/{user_id}", response_model=UserPublic)
def update_user(user_id: int, user: UserUpdate, session: SessionDep):
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
def delete_user(user_id: int, session: SessionDep):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")
    session.delete(user)
    session.commit()
    return {"ok": True, "message": "User berhasil dihapus"}