"""aqui são os modelos de busca, ou seja,
 os modelos que serão usados para buscar os dados no banco de dados"""

from pydantic import BaseModel
from typing import List

class SearchRequest(BaseModel):
    """é o modelo que o usuario vai enviar para o endpoint de busca, ele contém o texto que o usuário quer buscar
     e o número de resultados que ele quer receber"""
    query: str
    limit: int = 3

class SearchResult(BaseModel):
    """é o modelo que o endpoint de busca vai retornar para o usuário,
     ele contém o id do documento, a pontuação de relevância e o texto do documento"""
    score: float
    text: str
    metadata: dict

class SearchResponse(BaseModel):
    """é o modelo que o endpoint de busca vai retornar para o usuário,
     ele contém uma lista de resultados de busca"""
    results: List[SearchResult]