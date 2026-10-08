import hashlib
import zipfile
from pathlib import Path

from pypdf import PdfReader

from document_processor import BASE_DIR, extract_pages, save_extracted_text
from rag import chunk_text, get_embedding_model, get_supabase


MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 200
MAX_TEXT_CHARACTERS = 200_000
MAX_CHUNKS = 400
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def validate_document(path):
    if not path.is_file():
        raise ValueError("The uploaded document could not be found.")
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Supported formats are .pdf, .docx and .txt.")
    if path.stat().st_size == 0:
        raise ValueError("The uploaded document is empty.")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Please use a document no larger than 20 MB.")

    if path.suffix.lower() == ".pdf":
        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError("Please use a PDF without a password.")
            if len(reader.pages) > MAX_PDF_PAGES:
                raise ValueError("Please split PDFs longer than 200 pages.")
        except ValueError:
            raise
        except Exception as error:
            raise ValueError("The file could not be read as a PDF.") from error

    if path.suffix.lower() == ".docx":
        try:
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()
                if "[Content_Types].xml" not in archive.namelist() or "word/document.xml" not in archive.namelist():
                    raise ValueError("Please upload a valid Word .docx document.")
                if len(entries) > 2048 or sum(e.file_size for e in entries) > 32 * 1024 * 1024:
                    raise ValueError("The Word document is too large after decompression.")
        except zipfile.BadZipFile as error:
            raise ValueError("Please upload a valid Word .docx document.") from error


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_document(file_path, original_name=None):
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    path = path.resolve()
    validate_document(path)

    try:
        pages = extract_pages(path)
    except UnicodeDecodeError as error:
        raise ValueError("Please save text files with UTF-8 encoding.") from error
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("The document could not be extracted. Please check the file.") from error

    if sum(len(text) for _, text in pages) > MAX_TEXT_CHARACTERS:
        raise ValueError("This document contains too much text. Please split it into smaller files.")

    # The hash identifies identical file contents, including files renamed later.
    source = f"upload://{file_digest(path)}"
    title = original_name or path.name
    candidates = []
    for page_number, text in pages:
        for chunk in chunk_text(text):
            heading = f"PDF page {page_number}" if page_number else "Document text"
            candidates.append({
                "title": title,
                "source": source,
                "page_number": page_number,
                "content": f"{heading}\n\n{chunk}",
            })
    if len(candidates) > MAX_CHUNKS:
        raise ValueError("This document produces too many chunks. Please split it into smaller files.")

    text_path = save_extracted_text(path, pages)
    supabase = get_supabase()
    existing = set()
    offset = 0
    while True:
        response = (
            supabase.table("documents")
            .select("content,page_number")
            .eq("source", source)
            .order("id")
            .range(offset, offset + 499)
            .execute()
        )
        rows = response.data or []
        existing.update((row["page_number"], row["content"]) for row in rows)
        if len(rows) < 500:
            break
        offset += 500

    pending = []
    skipped = 0
    for row in candidates:
        identity = (row["page_number"], row["content"])
        if identity in existing:
            skipped += 1
        else:
            pending.append(row)
            existing.add(identity)

    if pending:
        embeddings = get_embedding_model().encode(
            [row["content"] for row in pending],
            normalize_embeddings=True,
        ).tolist()
        if len(embeddings) != len(pending) or any(len(vector) != 384 for vector in embeddings):
            raise ValueError("Expected one 384-dimensional embedding per chunk.")
        for row, embedding in zip(pending, embeddings):
            row["embedding"] = embedding
        # One insert request: a failed request does not commit only a subset.
        supabase.table("documents").insert(pending).execute()

    return {
        "status": "ingested" if pending else "unchanged",
        "filename": title,
        "source": source,
        "words_extracted": sum(len(text.split()) for _, text in pages),
        "pages_with_text": len(pages),
        "chunks_added": len(pending),
        "chunks_skipped": skipped,
        "text_file": f"extracted/{text_path.name}",
    }

