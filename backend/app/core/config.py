from pydantic_settings import BaseSettings
from typing import Optional, List, Literal
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class LLMProviderConfig(BaseModel):
    type: Literal["openai", "ollama"]
    base_url: str
    api_key: str | None = None
    chat_model: str
    embedding_model: str

class ModelConfig(BaseModel):
    id: str
    name: str
    type: Literal["openai", "ollama"]
    description: str
    provider_config: LLMProviderConfig

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "gpt-4o",
                "name": "GPT-4 (OpenAI)",
                "type": "openai",
                "description": "OpenAI's GPT-4 model"
            }
        }

class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG Web UI"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api"
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"
    ]
    
    # MySQL settings
    MYSQL_SERVER: str = os.getenv("MYSQL_SERVER", "localhost")
    MYSQL_USER: str = os.getenv("MYSQL_USER", "ragwebui")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "ragwebui")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "ragwebui")
    SQLALCHEMY_DATABASE_URI: Optional[str] = None

    @property
    def get_database_url(self) -> str:
        if self.SQLALCHEMY_DATABASE_URI:
            return self.SQLALCHEMY_DATABASE_URI
        return f"mysql+mysqlconnector://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_SERVER}/{self.MYSQL_DATABASE}"

    # JWT settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080

    # Vector DB settings
    CHROMA_DB_HOST: str = os.getenv("CHROMA_DB_HOST", "localhost")
    CHROMA_DB_PORT: int = int(os.getenv("CHROMA_DB_PORT", "8001"))

    # MinIO
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET_NAME: str = os.getenv("MINIO_BUCKET_NAME", "documents")

    # Ollama Embeddings Settings
    OLLAMA_EMBEDDINGS_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    
    # LLM Provider Settings
    OPENAI_PROVIDER: LLMProviderConfig = LLMProviderConfig(
        type="openai",
        base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    OLLAMA_PROVIDER: LLMProviderConfig = LLMProviderConfig(
        type="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="none",
        chat_model=os.getenv("OLLAMA_CHAT_MODEL", "deepseek-r1"),
        embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    )

    DEFAULT_LLM_PROVIDER: Literal["openai", "ollama"] = os.getenv("DEFAULT_LLM_PROVIDER", "ollama")

    # Vector Store Settings
    VECTOR_STORE_TYPE: str = "chroma"

    # Model settings
    # MODEL_CONFIGS: List[ModelConfig] = [
    #     ModelConfig(
    #         id="gpt-4",
    #         name="GPT-4 (OpenAI)",
    #         type="openai",
    #         description="OpenAI's GPT-4 model",
    #         provider_config=OPENAI_PROVIDER
    #     ),
    #     ModelConfig(
    #         id="deepseek-r1",
    #         name="Deepseek (Ollama)",
    #         type="ollama",
    #         description="Deepseek model via local Ollama",
    #         provider_config=OLLAMA_PROVIDER
    #     )
    # ]

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #     logger.info("Initializing settings")
    #     logger.info("MODEL_CONFIGS: %s", self.MODEL_CONFIGS)

settings = Settings() 