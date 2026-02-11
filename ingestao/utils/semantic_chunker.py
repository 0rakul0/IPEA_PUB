import warnings
from collections import defaultdict

import hdbscan
import torch
from anyio.lowlevel import current_token
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

warnings.simplefilter(action="ignore", category=FutureWarning)


class SemanticChunker:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        min_cluster_size: int = 3,
        orphan_cluster_size: int = 2,
        max_tokens: int = 512,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(model_name, device=device)

        self.model.max_seq_length = 512

        self.min_cluster_size = min_cluster_size
        self.orphan_cluster_size = orphan_cluster_size
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _cluster_and_process(self, texts, min_size):
        if len(texts) <= 1:
            return texts, texts if len(texts) == 1 else texts[0]

        embeddings = self.model.encode(texts, show_progress_bar=False)

        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_size,
            metric="euclidean",
        ).fit_predict(embeddings)

        clusters = defaultdict(list)
        orphans = []

        for i, label in enumerate(labels):
            if label != -1:
                clusters[label].append(texts[i])
            else:
                orphans.append(texts[i])

        chunks = []

        for cluster_paras in clusters.values():
            current_chunk = []
            current_tokens = 0

            for para in cluster_paras:
                para_tokens = len(self.tokenizer.tokenize(para, add_special_tokens=False))
                if current_tokens + para_tokens > self.max_tokens and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [para]
                    current_tokens = para_tokens
                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens

            if current_chunk:
                chunks.append("\n\n".join(current_chunk))

        return chunks, orphans

    def _split_long_paragraph(self, paragraph: str):
        tokens = self.tokenizer.tokenize(paragraph, add_special_tokens=False)

        if len(tokens) <= self.max_tokens:
            return [paragraph]

        chunks = []
        for i in range(0, len(tokens), self.max_tokens):
            sub_tokens = tokens[i:i + self.max_tokens]
            sub_text = self.tokenizer.convert_tokens_to_string(sub_tokens)
            chunks.append(sub_text)

        return chunks

    def create_chunks(self, text_content: str):

        raw_paragraphs = [p.strip() for p in text_content.split("\n\n") if len(p.strip().split()) > 10]

        paragraphs = []
        for p in raw_paragraphs:
            paragraphs.extend(self._split_long_paragraph(p))

        if not paragraphs:
            return []

        final_chunks, orphans = self._cluster_and_process(paragraphs, self.min_cluster_size)

        if len(orphans) > 1:
            orphans_chunks, single_orphans = self._cluster_and_process(orphans, self.orphan_cluster_size)
            final_chunks.extend(orphans_chunks)
            final_chunks.extend(single_orphans)
        else:
            final_chunks.extend(orphans)

        return final_chunks