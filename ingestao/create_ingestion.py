import os
import uuid
from pathlib import Path

from docling_core.transforms.chunker import HybridChunker
from tqdm import tqdm
import shutil
import logging
from datetime import datetime, timezone

import torch
import pymupdf

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from ingestao.utils.clean_itens import baixar_pdf_real
from ingestao.db.banco_metadados import MetadataDB


# ======================================
# CONFIGURAÇÕES
# ======================================

load_dotenv()

DENSE_MODEL = "intfloat/multilingual-e5-large"
SPARSE_MODEL = "Qdrant/bm25"
COLBERT_MODEL = "colbert-ir/colbertv2.0"

COLLECTION_NAME = "publicacoes_ipea"

MAX_TOKENS = 600

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=120,
)

print(qdrant.get_collections())

dense_model = TextEmbedding(DENSE_MODEL)
sparse_model = SparseTextEmbedding(SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)

hf_tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL)

tokenizer_chunker = HuggingFaceTokenizer(
    tokenizer=hf_tokenizer,
    max_tokens=MAX_TOKENS
)
db_metadata = MetadataDB()

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ======================================
# LOGGER
# ======================================

def criar_logger_documento(doc_id: str) -> logging.Logger:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"erro_{doc_id}_{timestamp}.log"

    logger = logging.getLogger(f"doc_{doc_id}")
    logger.setLevel(logging.ERROR)

    handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger


# ======================================
# VERIFICAÇÃO NO QDRANT
# ======================================

def documento_ja_indexado(doc_id: str) -> bool:
    result = qdrant.count(
        collection_name=COLLECTION_NAME,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.document_id",
                    match=models.MatchValue(value=doc_id)
                )
            ]
        )
    )
    return result.count > 0


# ======================================
# PDF FRAGMENTAÇÃO (APENAS PARA MEMÓRIA)
# ======================================

def split_pdf_em_blocos(pdf_path: Path, temp_dir: Path, pages_per_chunk: int = 5):
    temp_dir.mkdir(parents=True, exist_ok=True)
    blocos = []

    with pymupdf.open(pdf_path) as doc:
        total_pages = len(doc)

        for start in range(0, total_pages, pages_per_chunk):
            end = min(start + pages_per_chunk - 1, total_pages - 1)

            new_doc = pymupdf.open()
            new_doc.insert_pdf(doc, from_page=start, to_page=end)

            block_number = (start // pages_per_chunk) + 1
            page_path = temp_dir / f"{pdf_path.stem}_bloco_{block_number}.pdf"

            new_doc.save(page_path)
            new_doc.close()

            blocos.append(page_path)

    return blocos


def ler_pdf_com_docling(pdf_path: Path):
    accelerator = AcceleratorOptions(
        device=AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
    )

    pdf_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        generate_page_images=False,
        generate_picture_images=False,
        images_scale=0.7,
        accelerator_options=accelerator,
        ocr_options=EasyOcrOptions(lang=["pt", "en"]),
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
        }
    )

    temp_dir = Path("temp_pages") / pdf_path.stem

    # fragmenta apenas para evitar estouro de memória
    with pymupdf.open(pdf_path) as doc:
        total_pages = len(doc)

    if total_pages > 15:
        paginas = split_pdf_em_blocos(pdf_path, temp_dir)
    else:
        paginas = [pdf_path]
        temp_dir = None

    documentos_parciais = []

    for pagina_pdf in paginas:
        try:
            result = converter.convert(pagina_pdf)
            documentos_parciais.append(result.document)
        except Exception as e:
            print(f"[WARN] Bloco falhou: {e}")
            continue

    if not documentos_parciais:
        return None, None

    return documentos_parciais, temp_dir


# ======================================
# PROCESSAMENTO
# ======================================

def processar_documento(metadata: dict) -> bool:

    if not metadata:
        return False

    doc_id = metadata["id"]
    logger = criar_logger_documento(doc_id)

    if documento_ja_indexado(doc_id):
        print(f"[SKIP] Documento {doc_id} já indexado.")
        db_metadata.atualizar_status(doc_id, "processado")
        return True

    try:
        db_metadata.atualizar_status(doc_id, "em processamento")

        resultado = baixar_pdf_real(metadata["link_pdf"])
        if not resultado or not resultado[0]:
            db_metadata.atualizar_status(doc_id, "sem_pdf")
            return True

        pdf_path, link_download = resultado
        db_metadata.atualizar_link_donwload(doc_id, link_download)

        documentos_parciais, temp_dir = ler_pdf_com_docling(pdf_path)

        if not documentos_parciais:
            db_metadata.atualizar_status(doc_id, "erro")
            return True

        chunker = HybridChunker(
            tokenizer=tokenizer_chunker,
            max_tokens=MAX_TOKENS,
            merge_peers=True
        )

        chunks = []
        for doc in documentos_parciais:
            chunks.extend(chunker.chunk(doc))

        points = []
        BATCH_SIZE = 8

        for idx, chunk in tqdm(enumerate(chunks), total=len(chunks)):

            text_chunk = chunk.text.strip()

            if not text_chunk:
                continue

            # 🔥 Controle real por tokens (E5 safety)
            tokens = hf_tokenizer(
                text_chunk,
                add_special_tokens=False,
                truncation=False
            )["input_ids"]

            if len(tokens) > MAX_TOKENS:
                tokens = tokens[:MAX_TOKENS]
                text_chunk = hf_tokenizer.decode(tokens)

            # 🔥 ColBERT safety (128 tokens ideal)
            colbert_tokens = hf_tokenizer(
                text_chunk,
                add_special_tokens=False,
                truncation=True,
                max_length=128
            )["input_ids"]

            colbert_text = hf_tokenizer.decode(colbert_tokens)

            # ==========================
            # Embeddings
            # ==========================

            dense_embedding = list(
                dense_model.passage_embed([text_chunk])
            )[0].tolist()

            sparse_embedding = list(
                sparse_model.passage_embed([text_chunk])
            )[0].as_object()

            colbert_embedding = list(
                colbert_model.passage_embed([colbert_text])
            )[0].tolist()

            # ==========================
            # Payload estruturado
            # ==========================

            payload = {
                "text": text_chunk,
                "metadata": {
                    "document_id": doc_id,
                    "titulo": metadata.get("titulo"),
                    "autores": metadata.get("autores"),
                    "ano": metadata.get("ano"),
                    "tipo_conteudo": metadata.get("tipo_conteudo"),
                    "palavras_chave": metadata.get("palavras_chave"),
                    "chunk_index": idx,
                    "link_pdf": metadata.get("link_pdf"),
                    "link_download": link_download,

                }
            }

            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_embedding,
                    "sparse": sparse_embedding,
                    "colbert": colbert_embedding,
                },
                payload=payload,
            )

            points.append(point)

            if len(points) >= BATCH_SIZE:
                qdrant.upload_points(
                    collection_name=COLLECTION_NAME,
                    points=points,
                    wait=True,
                )
                points.clear()

        # flush final
        if points:
            qdrant.upload_points(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )

        db_metadata.atualizar_status(doc_id, "processado")

        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

        print(f"[OK] Documento {doc_id} processado.\n")
        return True

    except Exception as e:
        logger.exception(f"Erro ao processar documento {doc_id}: {str(e)}")
        db_metadata.atualizar_status(doc_id, "erro")
        print(f"[ERRO] Documento {doc_id} falhou.")
        return True


# ======================================
# LOOP PRINCIPAL
# ======================================

AUTOR = "Danilo"
interesse = "inteligência"

documentos = db_metadata.buscar_interesse(interesse)

if not documentos:
    print("Nenhum documento encontrado para o autor.")
else:
    for metadata in documentos:
        processar_documento(metadata)

print("Pipeline concluído.")