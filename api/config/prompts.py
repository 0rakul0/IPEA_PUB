RAG_PROMPT = """
Com base exclusivamente no contexto abaixo, responda à pergunta de forma clara e objetiva.

Contexto:
{context}

Pergunta:
{query}

Diretrizes:
- Não utilize conhecimento externo ao contexto.
- Não invente dados ou valores.
- Caso a informação não esteja presente, indique explicitamente a limitação.

Resposta:
"""
