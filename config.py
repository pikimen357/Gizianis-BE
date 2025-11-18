from pydantic import BaseModel

class CsrfSettings(BaseModel):
    secret_key: str = "super-secret"
    cookie_samesite: str = "lax"
    cookie_secure: bool = False
