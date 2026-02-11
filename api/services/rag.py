from openai import OpenAI
from api.config.settings import settings
from api.config.prompts import RAG_PROMPT
from api.models.rag import RAGResponse
from api.services.search import SearchService

class RagService:
    def __init__(self, search_service: SearchService):
        self.search_service = search_service
        self.openai = OpenAI(api_key=settings.openai_api_key)

    def generate_answer(self, query: str, limit: int=3):
        search_result = self.search_service.search(query, limit=limit)

        #montagem do contexto
        context = "\n\n".join(result.text for result in search_result.results)

        #prompt para o modelo
        prompt = RAG_PROMPT.format(context=context, query=query)

        # chama o modelo de linguagem para gerar a resposta
        response = self.openai.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        # inclui a resposta e os metadados dos resultados da busca na resposta final
        metadata = [{**result.metadata,
                     "score":result.score,
                     } for result in search_result.results
        ]

        return RAGResponse(
            query=query,
            answer=response.choices[0].message.content,
            metadata=metadata,
        )