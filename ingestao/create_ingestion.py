import os
import re
import uuid
import hashlib
from pathlib import Path
from typing import Optional

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
MAX_TOKENS = 300

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
        do_ocr=False,
        do_table_structure=False,
        ocr_options=EasyOcrOptions(lang=["pt"]),
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

    # 📂 Diretório organizado por documento
    img_dir = Path("img")/pdf_path.stem
    img_dir.mkdir(parents=True, exist_ok=True)

    page_images = {}
    picture_count = 0

    for element, _level in document.iterate_items():
        if isinstance(element, PictureItem):

            page_number = element.prov[0].page_no if element.prov else None
            if page_number is None:
                continue

            picture_count += 1

            image_path = img_dir / f"picture_{picture_count}.png"
            img = element.get_image(document)
            img.save(image_path, format="PNG")

            page_images.setdefault(page_number, []).append(str(image_path))

    return document, page_images



def processar_documento() -> bool:

    metadata = db_metadata.buscar_pendente(randomize=True)
    if not metadata:
        return False

    db_metadata.atualizar_status(metadata["id"], "em processamento")

    resultado = baixar_pdf_real(metadata["link_pdf"])
    if not resultado:
        db_metadata.atualizar_status(metadata["id"], "erro_download")
        return False

    pdf_path = resultado

    document, page_images = ler_pdf_com_docling(pdf_path)

    chunker = SemanticChunker(model_name=DENSE_MODEL, max_tokens=MAX_TOKENS )

    blocos = document.export_to_markdown()

    chunks = chunker.create_chunks(blocos)

    points = []

    for text in chunks:

        dense_embedding = list(dense_model.passage_embed([text]))[0].tolist()
        sparse_embedding = list(sparse_model.passage_embed([text]))[0].as_object()
        colbert_embedding = list(colbert_model.passage_embed([text]))[0].tolist()

        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_embedding,
                "sparse": sparse_embedding,
                "colbert": colbert_embedding,
            },
            payload={
                "text": text,
                "metadata": metadata,
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

    print(f"[OK] Documento processado com {len(points)} chunks.")
    return True


while True:
    sucesso = processar_documento()
    if not sucesso:
        break

print("Pipeline concluído.")
