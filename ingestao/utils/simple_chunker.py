from transformers import AutoTokenizer
import re
class SimpleChunker:
    def __init__(self,
                 model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 max_tokens: int = 440):
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count_tokens(self, text: str):
        return len(
            self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=False
            )["input_ids"]
        )

    def _split_sentences(self, text):
        return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    def create_chunks(self, text_content: str):

        text_content = str(text_content).lower()

        paragraphs = []

        raw_paragraphs = [p.strip() for p in text_content.split("\n\n") if p.strip()]

        for p in raw_paragraphs:
            if self.count_tokens(p) > self.max_tokens:
                sentences = self._split_sentences(p)
                paragraphs.extend(sentences)
            else:
                paragraphs.append(p)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            if para_tokens > self.max_tokens:
                # quebra parágrafo grande
                words = para.split()
                temp = ""

                for w in words:
                    candidate = f"{temp} {w}".strip()
                    if self.count_tokens(candidate) > self.max_tokens:
                        chunks.append(temp)
                        temp = w
                    else:
                        temp = candidate

                if temp:
                    chunks.append(temp)
                continue

            if current_tokens + para_tokens > self.max_tokens:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks
