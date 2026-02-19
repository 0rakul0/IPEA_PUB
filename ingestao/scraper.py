import requests
from typing import Any, Dict, List
from datetime import datetime, timezone
from utils.clean_itens import clean_item
from db.banco_metadados import MetadataDB

BASE = "https://repositorio.ipea.gov.br/server/api/discover/browses/dateissued/items?sort=dateissued,DESC"


class Scraper:
    """
    Scraper para o repositório IPEA.
    Agrupa operações de listagem (API), extração/limpeza de metadados e download de PDF.
    """

    def __init__(
        self,
        base_api: str = BASE,
    ) -> None:
        self.base_api = base_api
        self.db = MetadataDB()

    def _buscar_pagina(self, page_number: int) -> List[Dict[str, Any]]:
        """Busca os itens brutos da API para a página fornecida."""
        sep = "&" if "?" in self.base_api else "?"
        url = f"{self.base_api}{sep}page={page_number}"

        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data["_embedded"]["items"]

    def _extrair_campos(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extrai campos brutos da API (não limpa)."""
        metadata = item.get("metadata", {})

        def get(field: str):
            arr = metadata.get(field, [])
            if not arr:
                return None
            if len(arr) == 1:
                return arr[0]["value"]
            return [v["value"] for v in arr]

        titulo = get("dc.title.scholar") or item.get("name")

        return {
            "id": item.get("id"),
            "handle": get("dc.identifier.uri"),
            "titulo": titulo or "",
            "autores": get("dc.contributor.author") or [],
            "ano": get("dc.date.issued"),
            "resumo": get("dc.description.abstract") or [],
            "palavras_chave": get("dc.subject.keyword") or [],
            "tipo": get("dc.type") or [],
            "last_modified": item.get("lastModified"),
        }

    def processar_pagina(self, pagina: int) -> int:
        """
        Extrai, limpa e salva todos itens de uma página no banco de metadados.
        Retorna a quantidade de itens processados.
        """
        itens_raw = self._buscar_pagina(pagina)
        count = 0

        for raw in itens_raw:
            try:
                bruto = self._extrair_campos(raw)
                item = clean_item(bruto)

                # prepara estrutura SQLite (Banco 1)
                doc = {
                    "id": item["id"],
                    "titulo": item["titulo"],
                    "autores": item["autores"],
                    "ano": item["ano"],
                    "tipo_conteudo": item.get("tipo") or item.get("tipo_conteudo") or "",
                    "resumo": item["resumo"],
                    "palavras_chave": item["palavras_chave"],
                    "link_pdf": item["handle"],
                    "link_download": None,
                    "status_ingestao": "pendente",
                    "data_ingestao": datetime.now(timezone.utc).isoformat(),
                }
                self.db.inserir_documento(doc)
                count += 1

            except Exception as e:
                print(f"[Scraper] Falha ao processar item {raw.get('id', 'desconhecido')}: {e}", flush=True)

        return count