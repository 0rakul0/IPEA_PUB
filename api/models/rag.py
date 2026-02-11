from pydantic import BaseModel

class RAGRequest(BaseModel):
    query: str
    limit: int = 3

class RAGResponse(BaseModel):
    """pergunta e resposta"""
    query: str
    answer: str
    metadata: list[dict]


