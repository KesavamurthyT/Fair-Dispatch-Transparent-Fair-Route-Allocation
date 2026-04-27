"""
Application configuration using Pydantic Settings.
Loads from environment variables and .env file.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Database
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/fair_dispatch"
    
    # Application
    app_env: str = "development"
    debug: bool = True
    app_title: str = "Fair Dispatch System"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    
    # Workload Score Weights
    workload_weight_a: float = 1.0  # num_packages weight
    workload_weight_b: float = 0.5  # total_weight_kg weight
    workload_weight_c: float = 10.0  # route_difficulty_score weight
    workload_weight_d: float = 0.2  # estimated_time_minutes weight
    
    # Clustering Settings
    target_packages_per_route: int = 20
    
    # Route Difficulty Weights
    difficulty_weight_per_kg: float = 0.01
    difficulty_weight_per_stop: float = 0.1
    difficulty_base: float = 1.0
    
    # Time Estimation (minutes)
    time_per_package: float = 5.0
    time_per_stop: float = 3.0
    base_route_time: float = 30.0
    
    # LangGraph / LangSmith (optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: Optional[str] = None
    langchain_project: str = "fair-dispatch-dev"
    
    # Gemini API (optional)
    google_api_key: Optional[str] = None
    enable_gemini_explain: bool = False
    
    # Redis & Celery (Phase 10 - Async Architecture)
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: Optional[str] = None  # defaults to redis_url
    celery_result_backend: Optional[str] = None  # defaults to redis_url
    
    # Model Storage (safe JSON artifacts, NO pickle)
    model_storage_dir: str = "models/"
    
    # Feature Flags
    sync_allocation_enabled: bool = True  # Keep sync endpoint during migration
    
    @property
    def effective_broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url
    
    @property
    def effective_result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
