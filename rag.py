import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client


BASE_DIR = Path(__file__).resolve().parent
WEBSITE_TEXT = BASE_DIR / "extracted" / "NECA_Website_Knowledge_extracted.txt"


@lru_cache(maxsize=1)
def get_supabase():
    load_dotenv(BASE_DIR / ".env", override=True)

    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()

    if not url or not key:
        raise ValueError("Add SUPABASE_URL and SUPABASE_SERVICE_KEY to .env.")

    return create_client(url, key)


# Keep the original MiniLM model and pooling, using its quantized ONNX graph.
# No PyTorch or sentence-transformers package is loaded by this backend.
_MODEL_LOCK = __import__('threading').Lock()
_MODEL_INSTANCE = None
MODEL_REPOSITORY = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_GRAPH = "onnx/model_quint8_avx2.onnx"
MODEL_DIRECTORY = BASE_DIR / "model_cache"


def prepare_embedding_files():
    """Download only the tokenizer and quantized graph during the build."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for filename in ("tokenizer.json", MODEL_GRAPH):
        target = MODEL_DIRECTORY / filename
        if not target.is_file():
            hf_hub_download(
                repo_id=MODEL_REPOSITORY,
                revision=MODEL_REVISION,
                filename=filename,
                local_dir=str(MODEL_DIRECTORY),
            )
    return MODEL_DIRECTORY


class LightweightEmbeddingModel:
    """An encode-compatible MiniLM adapter with bounded inference memory."""

    def __init__(self):
        import threading
        import onnxruntime as ort
        from tokenizers import Tokenizer

        folder = prepare_embedding_files()
        self._lock = threading.Lock()
        self._tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=256)
        self._tokenizer.enable_padding(
            pad_id=self._tokenizer.token_to_id("[PAD]"), pad_token="[PAD]"
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self._session = ort.InferenceSession(
            str(folder / MODEL_GRAPH), sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._inputs = {entry.name for entry in self._session.get_inputs()}

    def encode(self, sentences, normalize_embeddings=True):
        import numpy as np

        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("Embedding inputs must be text.")
        vectors = []
        # Serialize tokenizer and inference: concurrent chats share one model.
        with self._lock:
            for offset in range(0, len(texts), 2):
                encoded = self._tokenizer.encode_batch(texts[offset:offset + 2])
                tensors = {
                    "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
                    "attention_mask": np.asarray([item.attention_mask for item in encoded], dtype=np.int64),
                    "token_type_ids": np.asarray([item.type_ids for item in encoded], dtype=np.int64),
                }
                outputs = self._session.run(
                    None, {name: tensors[name] for name in self._inputs}
                )
                hidden = outputs[0]
                if hidden.ndim != 3 or hidden.shape[-1] != 384:
                    raise ValueError("Unexpected MiniLM token embedding shape.")
                mask = tensors["attention_mask"][..., None].astype(np.float32)
                pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
                if normalize_embeddings:
                    pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
                if not np.isfinite(pooled).all():
                    raise ValueError("The embedding model returned invalid values.")
                vectors.extend(pooled)
        result = np.asarray(vectors, dtype=np.float32).reshape(-1, 384)
        return result[0] if single else result


def get_embedding_model():
    global _MODEL_INSTANCE
    with _MODEL_LOCK:
        if _MODEL_INSTANCE is None:
            print("Loading lightweight MiniLM embedding model...", flush=True)
            _MODEL_INSTANCE = LightweightEmbeddingModel()
    return _MODEL_INSTANCE

def chunk_text(text, chunk_size=500, overlap=50):
    if not 0 <= overlap < chunk_size:
        raise ValueError("Overlap must be smaller than the chunk size.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def read_website_sections():
    if not WEBSITE_TEXT.is_file():
        raise FileNotFoundError("Run document_processor.py first.")

    text = WEBSITE_TEXT.read_text(encoding="utf-8-sig")
    blocks = re.split(r"(?m)^SECTION \d+:\s*", text)[1:]
    sections = []

    for block in blocks:
        source = re.search(r"(?m)^Source URL:\s*(https?://\S+)", block)
        date = re.search(r"(?m)^Retrieved:\s*(\S+)", block)
        parts = re.split(r"(?m)^Retrieved:[^\n]*\n", block, maxsplit=1)

        if not source or not date or len(parts) != 2:
            raise ValueError("A website section is missing its source details.")

        body = parts[1].split("\n===")[0].strip()

        if not body:
            raise ValueError("A website section has no text.")

        sections.append({
            "title": block.splitlines()[0].strip(),
            "source": source.group(1),
            "retrieved": date.group(1),
            "text": body,
        })

    if not sections:
        raise ValueError("No website sections were found in the text file.")

    return sections


def ingest_website_knowledge():
    sections = read_website_sections()
    supabase = get_supabase()
    pending = []
    skipped = 0

    for section in sections:
        # Find previously stored chunks from this webpage.
        existing = set()
        offset = 0

        while True:
            result = (
                supabase.table("documents")
                .select("content")
                .eq("source", section["source"])
                .order("id")
                .range(offset, offset + 499)
                .execute()
            )

            rows = result.data or []
            existing.update(row["content"] for row in rows)

            if len(rows) < 500:
                break

            offset += 500

        for chunk in chunk_text(section["text"]):
            content = (
                f"Website snapshot retrieved: {section['retrieved']}.\n\n"
                f"{chunk}"
            )

            if content in existing:
                skipped += 1
                continue

            pending.append({
                "title": section["title"],
                "source": section["source"],
                "page_number": 0,
                "content": content,
            })
            existing.add(content)

    if pending:
        print(f"Creating embeddings for {len(pending)} chunks...", flush=True)

        embeddings = get_embedding_model().encode(
            [row["content"] for row in pending],
            normalize_embeddings=True,
        ).tolist()

        for row, embedding in zip(pending, embeddings):
            if len(embedding) != 384:
                raise ValueError("Expected 384-dimensional embeddings.")
            row["embedding"] = embedding

        print("Saving chunks to Supabase...", flush=True)
        supabase.table("documents").insert(pending).execute()

    total = (
        supabase.table("documents")
        .select("id", count="exact", head=True)
        .execute()
    )

    return {
        "sources": len(sections),
        "added": len(pending),
        "skipped": skipped,
        "total": total.count,
    }


def rag_search(question, top_k=5, match_threshold=0.3):
    question = question.strip()

    if not question:
        raise ValueError("Enter a question.")

    embedding = get_embedding_model().encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    result = get_supabase().rpc(
        "match_documents",
        {
            "query_embedding": embedding,
            "match_threshold": match_threshold,
            "match_count": top_k,
        },
    ).execute()

    return result.data or []


if __name__ == "__main__":
    try:
        print("Loading NECA website knowledge into Supabase...", flush=True)
        result = ingest_website_knowledge()

        print("\nSUCCESS: Website knowledge ready in Supabase")
        print(f"Website sources: {result['sources']}")
        print(f"New chunks added: {result['added']}")
        print(f"Unchanged chunks skipped: {result['skipped']}")
        print(f"Total chunks in database: {result['total']}")

    except Exception as error:
        message = str(error)
        secret = os.getenv("SUPABASE_SERVICE_KEY")

        if secret:
            message = message.replace(secret, "[hidden]")

        print(f"\nERROR: {message}")
        raise SystemExit(1)
