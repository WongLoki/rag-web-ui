from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class LLMFactory:
    @staticmethod
    def create_embeddings():
        """Create embeddings based on provider configuration"""
        provider = settings.DEFAULT_LLM_PROVIDER
        logger.info(f"Creating embeddings for provider: {provider}")
        
        if provider == "openai":
            config = settings.OPENAI_PROVIDER
            return OpenAIEmbeddings(
                openai_api_key=config.api_key,
                openai_api_base=config.base_url,
                model=config.embedding_model
            )
        else:  # ollama
            config = settings.OLLAMA_PROVIDER
            base_url = config.base_url
            if base_url.endswith('/v1'):
                base_url = base_url[:-3]
            
            logger.info(f"Creating Ollama embeddings with base_url: {base_url}")
            logger.info(f"Will request to: {base_url}/api/embeddings")
            logger.info(f"Ollama embedding model: {config.embedding_model}")
            
            return OllamaEmbeddings(
                base_url=base_url,
                model=config.embedding_model
            )

    @staticmethod
    def create_llm(model: str):
        """Create LLM based on provider configuration"""
        logger.info(f"Creating LLM for provider: {settings.DEFAULT_LLM_PROVIDER}")
        
        if settings.DEFAULT_LLM_PROVIDER == "openai":
            return ChatOpenAI(
                temperature=0,
                streaming=True,
                model=model,
                openai_api_key=settings.OPENAI_PROVIDER.api_key,
                openai_api_base=settings.OPENAI_PROVIDER.base_url
            )
        else:  # ollama
            base_url = settings.OLLAMA_PROVIDER.base_url
            if base_url.endswith('/v1'):
                base_url = base_url[:-3]
                
            return ChatOllama(
                temperature=0,
                model=model,
                base_url=base_url
            ) 