from datetime import datetime, timedelta
from typing import Annotated
import base64
import json
import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, status, Request, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Field, Session, SQLModel, create_engine, select
from passlib.context import CryptContext
from jose import JWTError, jwt

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Import untuk AI (pilih salah satu)
import google.generativeai as genai
from PIL import Image
import io

load_dotenv('.env.local')

# Security Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "abc123")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Google Gemini Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

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
    password: str


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


# Database Configuration
MYSQL_USER = os.environ.get("MYSQL_USER")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD")
MYSQL_HOST = os.environ.get("MYSQL_HOST")
MYSQL_PORT = os.environ.get("MYSQL_PORT")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE")

mysql_url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
engine = create_engine(mysql_url, echo=True)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

app = FastAPI(title="Food Nutrition API with WebSocket")

# CORS Configuration untuk Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001",
                   "http://localhost:5500", "http://127.0.0.1:5500"],  # Next.js ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
def on_startup():
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


async def get_user_from_token(token: str, session: Session) -> User | None:
    """Verify token dan return user"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            return None
        user = session.get(User, int(user_id))
        return user
    except JWTError:
        return None


def analyze_food_with_gemini(image_data: bytes, user_info: dict = None) -> dict:
    """
    Analisis makanan menggunakan Google Gemini Vision
    """
    try:
        # Load image
        image = Image.open(io.BytesIO(image_data))

        # Try multiple model versions
        model_names = [
            'models/gemini-2.5-flash-lite'
        ]

        model = None
        last_error = None

        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                # Test if model works
                break
            except Exception as e:
                last_error = str(e)
                continue

        if not model:
            return {
                "error": f"Tidak ada model Gemini yang tersedia. Last error: {last_error}",
                "nama_makanan": "Tidak dapat diidentifikasi",
                "pesan": "Silakan cek API key atau quota Gemini API Anda"
            }

        # Personalized prompt based on user info
        prompt = f"""Analisis gambar makanan ini dan berikan informasi nutrisi dalam format JSON.

Format JSON yang diharapkan:
{{
    "nama_makanan": "nama makanan yang terdeteksi",
    "porsi": "estimasi porsi (contoh: 1 porsi, 200g)",
    "kalori": kalori_dalam_angka,
    "nutrisi": {{
        "protein": "dalam gram",
        "karbohidrat": "dalam gram",
        "lemak": "dalam gram",
        "serat": "dalam gram"
    }},
    "vitamin_mineral": ["Vitamin A", "Zat Besi", "dll"],
    "rekomendasi": "saran berdasarkan profil pengguna"
}}
"""

        if user_info:
            prompt += f"\n\nProfil pengguna:"
            if user_info.get('tinggi_badan') and user_info.get('berat_badan'):
                bmi = user_info['berat_badan'] / ((user_info['tinggi_badan'] / 100) ** 2)
                prompt += f"\n- BMI: {bmi:.1f}"
            if user_info.get('alergi'):
                prompt += f"\n- Alergi: {user_info['alergi']}"
            if user_info.get('jenis_kelamin'):
                prompt += f"\n- Jenis Kelamin: {user_info['jenis_kelamin']}"

        prompt += "\n\nBerikan respons HANYA dalam format JSON yang valid, tanpa markdown atau teks tambahan."

        # Generate response
        response = model.generate_content([prompt, image])

        # Parse JSON response
        response_text = response.text.strip()
        # Remove markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]

        nutrition_data = json.loads(response_text.strip())
        return nutrition_data

    except Exception as e:
        return {
            "error": str(e),
            "nama_makanan": "Tidak dapat diidentifikasi",
            "pesan": "Terjadi kesalahan saat menganalisis gambar"
        }


# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)


manager = ConnectionManager()


# WebSocket Endpoint
@app.websocket("/ws/analyze/{token}")
async def websocket_analyze_food(websocket: WebSocket, token: str):
    """
    WebSocket endpoint untuk analisis makanan real-time
    Client mengirim: {type: "image", data: "base64_image_string"}
    Server mengirim: {type: "progress", message: "..."} atau {type: "result", data: {...}}
    """
    client_id = f"client_{id(websocket)}"

    # Verify token
    with Session(engine) as session:
        user = await get_user_from_token(token, session)
        if not user:
            await websocket.close(code=1008, reason="Invalid token")
            return

        user_info = {
            "tinggi_badan": user.tinggi_badan,
            "berat_badan": user.berat_badan,
            "alergi": user.alergi,
            "jenis_kelamin": user.jenis_kelamin
        }

    await manager.connect(websocket, client_id)

    try:
        await manager.send_message({
            "type": "connected",
            "message": f"Terhubung sebagai {user.username}"
        }, client_id)

        while True:
            # Terima data dari client
            data = await websocket.receive_json()

            if data.get("type") == "image":
                # Send progress
                await manager.send_message({
                    "type": "progress",
                    "message": "Menganalisis gambar..."
                }, client_id)

                # Decode base64 image
                image_base64 = data.get("data", "").split(",")[-1]  # Remove data:image/...;base64,
                image_bytes = base64.b64decode(image_base64)

                # Send progress
                await manager.send_message({
                    "type": "progress",
                    "message": "Menghubungi AI untuk analisis nutrisi..."
                }, client_id)

                # Analyze with AI
                result = analyze_food_with_gemini(image_bytes, user_info)

                # Send result
                await manager.send_message({
                    "type": "result",
                    "data": result,
                    "timestamp": datetime.now().isoformat()
                }, client_id)

            elif data.get("type") == "ping":
                await manager.send_message({"type": "pong"}, client_id)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        print(f"Client {client_id} disconnected")
    except Exception as e:
        await manager.send_message({
            "type": "error",
            "message": str(e)
        }, client_id)
        manager.disconnect(client_id)


# Auth Endpoints (sama seperti sebelumnya)
@app.post("/register", response_model=UserPublic, status_code=201)
def register(user: UserCreate, session: SessionDep):
    existing_user = session.exec(
        select(User).where(User.username == user.username)
    ).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username sudah digunakan")

    existing_email = session.exec(
        select(User).where(User.email == user.email)
    ).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="Email sudah digunakan")

    user_data = user.model_dump()
    user_data["password"] = hash_password(user_data["password"])
    db_user = User.model_validate(user_data)

    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


@app.post("/login", response_model=Token)
def login(user_login: UserLogin, session: SessionDep):
    user = session.exec(
        select(User).where(User.email == user_login.email)
    ).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email atau password salah"
        )

    if not verify_password(user_login.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email atau password salah"
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=access_token_expires
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserPublic.model_validate(user)
    )


@app.get("/list-models")
def list_available_models():
    """List semua model Gemini yang tersedia"""
    try:
        models = genai.list_models()
        available_models = []

        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available_models.append({
                    "name": model.name,
                    "display_name": model.display_name,
                    "description": model.description,
                    "methods": model.supported_generation_methods
                })

        return {
            "status": "success",
            "total": len(available_models),
            "models": available_models
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "hint": "Pastikan GEMINI_API_KEY sudah benar di .env.local"
        }


# REST API Endpoint untuk Testing (tanpa WebSocket)
@app.post("/analyze-food")
async def analyze_food_rest(
        request: Request,
        session: SessionDep,
        credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    REST API endpoint untuk testing di Postman
    Body: {"image": "base64_string_here"}
    Header: Authorization: Bearer {token}
    """
    try:
        # Verify token
        user = await get_user_from_token(credentials.credentials, session)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")

        # Get request body
        body = await request.json()
        image_base64 = body.get("image", "")

        if not image_base64:
            raise HTTPException(status_code=400, detail="Image data required")

        # Remove data URI prefix if exists
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        # Decode image
        image_bytes = base64.b64decode(image_base64)

        # User info
        user_info = {
            "tinggi_badan": user.tinggi_badan,
            "berat_badan": user.berat_badan,
            "alergi": user.alergi,
            "jenis_kelamin": user.jenis_kelamin
        }

        # Analyze
        result = analyze_food_with_gemini(image_bytes, user_info)

        return {
            "status": "success",
            "user": user.username,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check
@app.get("/")
def read_root():
    return {
        "message": "Food Nutrition API with WebSocket",
        "endpoints": {
            "rest_api": "/analyze-food (POST)",
            "websocket": "/ws/analyze/{token}"
        },
        "status": "running"
    }