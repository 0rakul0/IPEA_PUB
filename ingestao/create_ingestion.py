import os
import re
import uuid
import hashlib
from pathlib import Path
from typing import Optional
from collections import defaultdict, Counter

import torch

from docling_core.transforms.chunker import HybridChunker
from docling_core.types import DoclingDocument
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling_core.types.doc import PictureItem

from utils.semantic_chunker import SemanticChunker
from utils.clean_itens import baixar_pdf_real
from db.banco_metadados import MetadataDB

load_dotenv()

FILE_PATH = "data"

DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SPARSE_MODEL = "Qdrant/bm25"
COLBERT_MODEL = "colbert-ir/colbertv2.0"
COLLECTION_NAME = "publicacoes_ipea"

EMAIL = "jefferson.ti@hotmail.com"
MAX_TOKENS = 290

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
)

dense_model = TextEmbedding(DENSE_MODEL)
sparse_model = SparseTextEmbedding(SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)

db_metadata = MetadataDB()

def ler_pdf_com_docling(pdf_path: Path):

    accelerator = AcceleratorOptions(
        device=AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
    )

    pdf_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=False,
        ocr_options=EasyOcrOptions(lang=["pt", "en"]),
        accelerator_options=accelerator,
        generate_page_images=False,
        images_scale=1.5,
        generate_picture_images=True
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
        }
    )

    result = converter.convert(pdf_path)
    document = result.document

    img_dir = Path("img")/pdf_path.stem
    img_dir.mkdir(parents=True, exist_ok=True)

    page_images = {}
    picture_count = 0

    for element, _level in document.iterate_items():
        if isinstance(element, PictureItem):

            page_number = element.prov[0].page_no if element.prov else None
            if page_number is None:
                continue

            img = element.get_image(document)

            if img.width >= 100 and img.height >= 100:
                picture_count += 1
                image_path = img_dir / f"picture_{picture_count}.png"
                img.save(image_path, format="PNG")
                page_images.setdefault(page_number, []).append(str(image_path))

    return document, page_images

def limpar_texto(texto: str) -> str:
    texto = re.sub(r"glyph<[^>]+>", "", texto)
    texto = re.sub(r"[^\x00-\x7FÀ-ÿ0-9.,;:!?()%\-–—\n ]+", " ", texto)
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def remover_linhas_repetidas(texto: str, limite_repeticao: int = 5) -> str:
    linhas = [l.strip() for l in texto.split("\n") if l.strip()]
    contagem = Counter(linhas)

    linhas_filtradas = [
        linha for linha in linhas
        if contagem[linha] <= limite_repeticao
    ]

    return "\n".join(linhas_filtradas)


def remover_watermark_curto(texto: str, limite_repeticao=3, tamanho_max=40):
    linhas = [l.strip() for l in texto.split("\n") if l.strip()]
    contagem = Counter(linhas)

    linhas_filtradas = []
    for linha in linhas:
        if (
            contagem[linha] > limite_repeticao
            and len(linha) <= tamanho_max
        ):
            continue
        linhas_filtradas.append(linha)

    return "\n".join(linhas_filtradas)



def processar_documento() -> bool:
    metadata = db_metadata.buscar_pendente()
    if not metadata:
        return False

    db_metadata.atualizar_status(metadata["id"], "em processamento")

    pdf_path, link_download = baixar_pdf_real(metadata["link_pdf"])

    if not pdf_path:
        print("[Ingestão] Documento sem PDF válido. Pulando.")
        db_metadata.atualizar_status(metadata["id"], "sem_pdf")
        return True
    else:
        db_metadata.atualizar_link_donwload(metadata["id"], f"{link_download}")

    document, page_images = ler_pdf_com_docling(pdf_path)

    chunker = SemanticChunker(model_name=DENSE_MODEL, max_tokens=MAX_TOKENS)

    textos_por_pagina = defaultdict(str)

    for element, _level in document.iterate_items():
        if not hasattr(element, "text") or not element.text:
            continue
        page_number = element.prov[0].page_no if element.prov else None
        if page_number is None:
            continue

        texto_limpo = limpar_texto(element.text)
        texto_limpo = remover_linhas_repetidas(texto_limpo)
        texto_limpo = remover_watermark_curto(texto_limpo)

        # Se for título
        if element.label and "title" in element.label.lower():
            textos_por_pagina[page_number] += f"\n## {texto_limpo}\n\n"
        else:
            textos_por_pagina[page_number] += f"{texto_limpo}\n\n"

    points = []

    for page_number, bloco_texto in textos_por_pagina.items():

        chunks = chunker.create_chunks(bloco_texto)

        for text_chunk in chunks:

            dense_embedding = list(dense_model.passage_embed([text_chunk]))[0].tolist()
            sparse_embedding = list(sparse_model.passage_embed([text_chunk]))[0].as_object()
            colbert_embedding = list(colbert_model.passage_embed([text_chunk]))[0].tolist()

            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_embedding,
                    "sparse": sparse_embedding,
                    "colbert": colbert_embedding,
                },
                payload={
                    "text": text_chunk,
                    "metadata": {
                        "document_id": metadata["id"],
                        "titulo": metadata.get("titulo"),
                        "autores": metadata.get("autores"),
                        "ano": metadata.get("ano"),
                        "tipo_conteudo": metadata.get("tipo_conteudo"),
                        "palavras_chave": metadata.get("palavras_chave"),
                        "pagina": page_number,
                        "imagens_pagina": page_images.get(page_number, []),
                    },
                }
            )

            points.append(point)

    qdrant.upload_points(
        collection_name=COLLECTION_NAME,
        points=points,
        batch_size=8,
        wait=True,
    )

    db_metadata.atualizar_status(metadata["id"], "processado")

    print(f"[OK] Documento processado com {len(points)} chunks.\n\n")
    return True


while True:
    sucesso = processar_documento()
    if not sucesso:
        break

print("Pipeline concluído.")
