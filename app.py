import hashlib
import logging
import os
import re
import secrets
import threading
import time
from collections import deque
from pathlib import Path, PurePosixPath
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from agents import AGENTS, handle_question
from guardrails import GuardrailError, safe_error
from ingestion import ALLOWED_EXTENSIONS, MAX_FILE_BYTES, ingest_document
from rag import ingest_website_knowledge, list_stored_sources


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
logger = logging.getLogger("neca.api")
chat_slots = threading.BoundedSemaphore(3)
upload_lock = threading.Lock()
rate_lock = threading.Lock()
chat_history = {}
upload_security = HTTPBearer(auto_error=False, scheme_name="UploadAdminKey")


class RequestSizeLimit:
    """Limit request bodies before multipart files are fully parsed."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        limit = MAX_FILE_BYTES + 64 * 1024 if scope.get("path") == "/upload" else 8192
        headers = dict(scope.get("headers", []))
        length = headers.get(b"content-length")
        if length is not None:
            try:
                declared = int(length)
                if declared < 0:
                    raise ValueError
            except ValueError:
                await JSONResponse({"detail": "Invalid request size."}, status_code=400)(scope, receive, send)
                return
            if declared > limit:
                await JSONResponse({"detail": "The request is too large."}, status_code=413)(scope, receive, send)
                return
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(status_code=413, detail="The request is too large.")
            return message

        await self.app(scope, limited_receive, send)


app = FastAPI(
    title="NECA Multi-Agent API",
    description="NECA chat with specialist agents and administrator document ingestion.",
    version="1.0.0",
)
app.add_middleware(RequestSizeLimit)
origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=3, max_length=2000, strict=True,
        examples=["What are the NECA membership requirements?"],
    )


class SourceReference(BaseModel):
    number: int
    title: str
    source: str
    url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    agent: str
    role: str
    status: str
    sources: list[SourceReference]


def log_error(operation, error):
    message = safe_error(error)
    key = os.getenv("UPLOAD_API_KEY", "").strip()
    if key:
        message = message.replace(key, "[hidden]")
    logger.error("%s failed: %s", operation, message)


def check_chat_rate(request):
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with rate_lock:
        if client not in chat_history and len(chat_history) >= 2048:
            chat_history.pop(next(iter(chat_history)))
        history = chat_history.setdefault(client, deque())
        while history and history[0] <= now - 60:
            history.popleft()
        if len(history) >= 12:
            raise HTTPException(
                status_code=429,
                detail="Please wait a minute before asking more questions.",
                headers={"Retry-After": "60"},
            )
        history.append(now)


def require_upload_key(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(upload_security)],
):
    expected = os.getenv("UPLOAD_API_KEY", "").strip()
    if len(expected) < 32:
        raise HTTPException(
            status_code=503,
            detail="Document ingestion is disabled until the server has an UPLOAD_API_KEY of at least 32 characters.",
        )
    supplied = credentials.credentials if credentials else ""
    if not secrets.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=401,
            detail="A valid administrator upload key is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/health")
def health():
    # Process health only: this does not test Groq or Supabase connections.
    return {
        "status": "ok",
        "service": "NECA Multi-Agent API",
        "upload_enabled": len(os.getenv("UPLOAD_API_KEY", "").strip()) >= 32,
    }


@app.get("/agents")
def list_agents():
    return [{"name": agent.name, "role": agent.role} for agent in AGENTS.values()]



class StoredSource(BaseModel):
    title: str
    source: str
    type: str
    url: str | None = None
    chunks: int


class StoredSourcesResponse(BaseModel):
    sources: list[StoredSource]
    total_sources: int
    total_chunks: int


@app.get("/sources", response_model=StoredSourcesResponse)
def stored_sources():
    try:
        return list_stored_sources()
    except Exception as error:
        log_error("Source listing", error)
        raise HTTPException(
            status_code=502,
            detail="Stored sources could not be listed. Please try again.",
        ) from error


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    check_chat_rate(request)
    key = os.getenv("UPLOAD_API_KEY", "").strip()
    if key and key in body.question:
        return {
            "answer": "Remove API keys or secrets from your question.",
            "agent": "Input Guard", "role": "Request safety",
            "status": "blocked", "sources": [],
        }
    if not chat_slots.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="The service is busy. Please try again shortly.")
    try:
        result = await run_in_threadpool(handle_question, body.question)
        if key and key in str(result):
            raise GuardrailError("The answer was blocked because it contained a secret.")
        for source in result["sources"]:
            stored = source["source"]
            source["url"] = stored if stored.startswith(("https://", "http://")) else None
        return result
    except GuardrailError as error:
        return {
            "answer": safe_error(error),
            "agent": "Input / Output Guard", "role": "Request and response safety",
            "status": "blocked", "sources": [],
        }
    except Exception as error:
        log_error("Chat", error)
        raise HTTPException(
            status_code=502,
            detail="The assistant could not finish the request. Please try again.",
        ) from error
    finally:
        chat_slots.release()


def clean_filename(raw):
    name = PurePosixPath((raw or "").replace("\\", "/")).name
    name = "".join(char for char in name if ord(char) >= 32).strip()[:120]
    if not name or Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Supported formats are .pdf, .docx and .txt.")
    return name


@app.post("/upload", dependencies=[Depends(require_upload_key)])
async def upload_document(file: Annotated[UploadFile, File(description="PDF, Word DOCX or UTF-8 TXT; up to 20 MB.")]):
    if not upload_lock.acquire(blocking=False):
        await file.close()
        raise HTTPException(status_code=429, detail="Another document is being ingested. Please try again shortly.")
    temporary_path = None
    try:
        filename = clean_filename(file.filename)
        folder = BASE_DIR / "uploads"
        folder.mkdir(parents=True, exist_ok=True)
        temporary_path = folder / f"pending_{secrets.token_hex(8)}{Path(filename).suffix.lower()}"
        size = 0
        digest = hashlib.sha256()
        with temporary_path.open("xb") as stream:
            while block := await file.read(1024 * 1024):
                size += len(block)
                if size > MAX_FILE_BYTES:
                    raise HTTPException(status_code=413, detail="Please use a document no larger than 20 MB.")
                digest.update(block)
                stream.write(block)
        if size == 0:
            raise ValueError("The uploaded document is empty.")
        safe_stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(filename).stem)[:50] or "document"
        stored_path = folder / f"{digest.hexdigest()[:16]}_{safe_stem}{Path(filename).suffix.lower()}"
        # The per-process lock prevents concurrent ingestion of the same file.
        temporary_path.replace(stored_path)
        temporary_path = None
        try:
            return await run_in_threadpool(ingest_document, stored_path, filename)
        except ValueError:
            stored_path.unlink(missing_ok=True)
            raise
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        log_error("Document ingestion", error)
        raise HTTPException(
            status_code=502,
            detail="The document could not be saved to the knowledge base. Please try again.",
        ) from error
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
        await file.close()
        upload_lock.release()


@app.post("/ingest/website", dependencies=[Depends(require_upload_key)])
async def ingest_website():
    if not upload_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Another document is being ingested. Please try again shortly.")
    try:
        return await run_in_threadpool(ingest_website_knowledge)
    except FileNotFoundError as error:
        raise HTTPException(status_code=400, detail="Run document_processor.py on the website knowledge file first.") from error
    except Exception as error:
        log_error("Website ingestion", error)
        raise HTTPException(status_code=502, detail="Website ingestion could not finish. Please try again.") from error
    finally:
        upload_lock.release()

