# NECA Multi-Agent API

A FastAPI backend that answers NECA questions using specialist agents
and a Supabase knowledge base.

## Features

- Routes questions to knowledge, membership, training, or support agents.
- Uses retrieved evidence and source citations.
- Applies input and output guardrails.
- Ingests saved website knowledge and PDF, DOCX, or TXT uploads.
- Saves extracted text and skips unchanged content.

## Local setup

Install dependencies:

    python -m pip install -r requirements.txt

Create a private .env file containing:

    SUPABASE_URL=your_project_url
    SUPABASE_SERVICE_KEY=your_backend_secret
    GROQ_API_KEY=your_groq_key
    UPLOAD_API_KEY=your_private_upload_key

Use an upload key containing at least 32 characters.
The Supabase database requires a documents table with 384-dimensional
embeddings and the match_documents search function.

Start the API:

    python -m uvicorn app:app --reload

Open http://127.0.0.1:8000/docs.

## Website knowledge

Prepare the saved website text:

    python document_processor.py

Ingest it:

    python rag.py

## API endpoints

- GET /health — service status.
- GET /agents — available agents.
- POST /chat — answer a question.
- POST /upload — ingest a document; administrator key required.
- POST /ingest/website — ingest the saved website snapshot;
  administrator key required.

In Swagger, use Authorize and enter the UPLOAD_API_KEY value.

Uploads support files up to 20 MB. Scanned PDFs require OCR before upload.
The uploads/ and extracted/ folders are created automatically.

Keep .env and administrator keys private.