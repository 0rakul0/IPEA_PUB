import json
import os

import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def search_publicacoes(query: str, limit: int = 3, top_k: int | None = None):
    url = "http://localhost:8080/search"
    if top_k is not None:
        limit = top_k

    payload = {
        "query": query,
        "limit": limit
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


tools = [
    {
        "type": "function",
        "name": "search_publicacoes",
        "description": "Busca as publicações do IPEA relacionadas a um tema específico.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "O tema ou assunto para o qual deseja buscar publicações."
                },
                "limit": {
                    "type": "integer",
                    "description": "O número de publicações mais relevantes a serem retornadas (padrão é 3)."
                }
            },
            "required": ["query"]
        }
    }
]

input_list = [{
    "role": "user",
    "content": "Quais são as publicações mais relevantes do IPEA sobre políticas públicas para redução da desigualdade social?"
}]

response = client.responses.create(
    model="gpt-4o-mini",
    tools=tools,
    input=input_list,
)

input_list += response.output

for item in response.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)
        result = search_publicacoes(**args)

        texts = [r["text"] for r in result["results"]]

        input_list.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": json.dumps({"results": texts}, ensure_ascii=False)
        })


final_response = client.responses.create(
    model="gpt-4o-mini",
    input=input_list,
    tools=tools,
    instructions="""Responda à pergunta do usuário usando as informações das publicações retornadas pela função de busca. 
    Sintetize as informações de forma clara e objetiva, destacando os pontos mais relevantes relacionados a políticas públicas
     para redução da desigualdade social."""
)

print(final_response.output_text)

