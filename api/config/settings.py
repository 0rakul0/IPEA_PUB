from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )

    qdrant_url: str
    collecion_name: str = "publicacoes_ipea"

    dense_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    sparse_model: str = "Qdrant/bm25"
    colbert_model: str = "colbert-ir/colbertv2.0"

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"

settings = Settings()