import uvicorn
from fastapi import FastAPI
from routers import search, rag

app = FastAPI(title="API de busca das publicações do IPEA")

# adiciona os roteadores para as rotas de busca e RAG
app.include_router(search.router)
app.include_router(rag.router)

@app.get("/")
async def root():
    return {"message": "API de busca"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
