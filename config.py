# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    GITHUB_TOKEN: str
    STUDENT_SECRET: str
    GITHUB_USERNAME: str
    STUDENT_EMAIL: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache()
def get_settings():
    return Settings()