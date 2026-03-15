import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import document_router, rag_router, search_router

app = FastAPI(title="API de busca das publicações do IPEA")


# 🔥 CONFIGURAÇÃO DE CORS
origins = [
    "http://localhost:63342",  # seu frontend atual (JetBrains)
    "http://localhost:3000",   # caso use outro server
    "http://127.0.0.1:63342",
    "null",                    # quando o frontend e aberto via file://
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # pode usar ["*"] em desenvolvimento
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],    # permite OPTIONS, POST, GET etc
    allow_headers=["*"],
)


# adiciona os roteadores para as rotas de busca e RAG
app.include_router(search_router.router)
app.include_router(rag_router.router)
app.include_router(document_router.router)


@app.get("/")
async def root():
    return {"message": "API de busca"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
