"""
Todas as funções utilitárias para o ingestor

- modelos de visão para legendar imagens e tabelas
- cleaner de texto extraído
"""
import os
import hashlib
from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import Optional, Tuple
from datetime import datetime, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry



CRAWLER_URL = "https://repositorio.ipea.gov.br"
CRAWLER_HEADER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

CACHE_DIR = Path("./cache/pdfs")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def create_retry_session(
        total_retries=2,
        backoff_factor=1,
) -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=total_retries,
        read=total_retries,
        connect=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def _normalize_string(value):
    if value is None:
        return None
    if isinstance(value, list):
        value = "; ".join(str(v) for v in value)
    value = str(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or None

def _join_field(value, sep="; "):
    if value is None:
        return None
    if isinstance(value, list):
        flat = []
        for v in value:
            if isinstance(v, list):
                flat.extend(str(x) for x in v)
            else:
                flat.append(str(v))
        return _normalize_string(sep.join(flat))
    return _normalize_string(value)

def _parse_year(ano):
    if ano is None:
        return None
    match = re.search(r"(\d{4})", str(ano))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None

def _parse_last_modified(iso_str):
    if not iso_str:
        return None, None

    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.isoformat().replace("+00:00", "Z"), int(dt_utc.timestamp())
    except Exception:
        return None, None

def clean_item(raw):
    cleaned = {}

    cleaned["id"] = _normalize_string(raw.get("id"))
    cleaned["handle"] = _normalize_string(raw.get("handle"))

    cleaned["titulo"] = _normalize_string(raw.get("titulo")) or ""
    cleaned["resumo"] = _join_field(raw.get("resumo"), sep="\n\n") or ""

    cleaned["autores"] = _join_field(raw.get("autores"), sep="; ") or ""
    palavras = raw.get("palavras_chave") or raw.get("palavras-chave")
    cleaned["palavras_chave"] = _join_field(palavras, sep=", ") or ""

    cleaned["tipo"] = _join_field(raw.get("tipo"), sep="; ") or ""

    cleaned["ano"] = _parse_year(raw.get("ano"))

    iso_utc, epoch = _parse_last_modified(_normalize_string(raw.get("last_modified")))
    cleaned["last_modified"] = iso_utc
    cleaned["last_modified_ts"] = epoch

    return cleaned

def _url_to_cache_filename(url: str) -> Path:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{h}.pdf"

session = create_retry_session(total_retries=2)

def baixar_pdf_real(link_pagina: str) -> Optional[Path]:

    """
    Retorna:
        (caminho_pdf, hash_sha256)
    """

    print(f"[Crawler] Acessando página do documento:\n  {link_pagina}")

    try:
        resp = session.get(link_pagina, headers=CRAWLER_HEADER, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"[Crawler] ERRO ao acessar página: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    download_url = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "bitstreams" in href and "download" in href:
            download_url = urljoin(CRAWLER_URL, href)
            break

    if not download_url:
        print("[Crawler] Nenhum link de download encontrado.")
        return None

    print(f"[Crawler] Botão de download encontrado:\n  {download_url}")

    try:
        r = session.get(
            download_url,
            headers=CRAWLER_HEADER,
            timeout=60,
            allow_redirects=True,
        )
        r.raise_for_status()
    except Exception as e:
        print(f"[Crawler] ERRO ao baixar PDF: {e}")
        return None

    if not r.content.startswith(b"%PDF"):
        print("[Crawler] Conteúdo não parece ser um PDF válido.")
        return None

    # 🔐 Calcular hash
    sha256_hash = hashlib.sha256(r.content).hexdigest()

    filename = f"{sha256_hash}.pdf"
    cache_path = CACHE_DIR / filename

    # 📦 Cache hit
    if cache_path.exists():
        print("[Crawler] PDF recuperado do cache")
        return cache_path


    # 💾 Salvar
    with open(cache_path, "wb") as f:
        f.write(r.content)

    print(f"[Crawler] PDF salvo em cache:\n  {cache_path}")

    return cache_path
