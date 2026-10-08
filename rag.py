import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
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


@lru_cache(maxsize=1)
def get_embedding_model():
    print("Loading embedding model...", flush=True)
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


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
    model = get_embedding_model()
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

        embeddings = model.encode(
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