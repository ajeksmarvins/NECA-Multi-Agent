import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from supabase import create_client


load_dotenv(Path(__file__).with_name(".env"), override=True)

stage = "Reading .env"

try:
    required = [
        "GROQ_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_KEY",
    ]

    for name in required:
        value = os.getenv(name, "").strip()
        if not value or value.startswith("replace_with_"):
            raise ValueError(f"Add your actual {name} to .env")

    # 1. Check access to the database.
    stage = "Connecting to Supabase"
    print(stage + "...", flush=True)

    supabase = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"],
    )

    result = (
        supabase.table("documents")
        .select("id", count="exact", head=True)
        .execute()
    )

    print(f"PASS: Supabase connected. Stored chunks: {result.count}")

    # 2. Download/load the instructor's embedding model.
    stage = "Loading the embedding model"
    print(stage + "...", flush=True)

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embedding = embedding_model.encode(
        "NECA training information",
        normalize_embeddings=True,
    ).tolist()

    if len(embedding) != 384:
        raise ValueError("Expected an embedding with 384 dimensions")

    print("PASS: Embedding model working. Dimensions: 384")

    # 3. Check that the database search function works.
    stage = "Testing the RAG search function"

    supabase.rpc(
        "match_documents",
        {
            "query_embedding": embedding,
            "match_threshold": 0.0,
            "match_count": 1,
        },
    ).execute()

    print("PASS: RAG search function working")

    # 4. Make a small request to Groq.
    stage = "Testing Groq"
    print(stage + "...", flush=True)

    groq_client = Groq(
        api_key=os.environ["GROQ_API_KEY"],
        timeout=30.0,
        max_retries=1,
    )

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": "Reply with only: Connection successful",
            }
        ],
        reasoning_effort="low",
        temperature=0,
        max_completion_tokens=256,
    )

    reply = response.choices[0].message.content

    if not reply or not reply.strip():
        raise ValueError("Groq returned an empty response")

    print("PASS: Groq connected")
    print("Groq response:", reply.strip())
    print("\nAll connection checks passed!")

except Exception as error:
    message = str(error)

    # Hide credentials if an error happens to contain them.
    for name in ("GROQ_API_KEY", "SUPABASE_SERVICE_KEY"):
        secret = os.getenv(name)
        if secret:
            message = message.replace(secret, "[hidden]")

    print(f"\nFAILED during: {stage}")
    print(f"{type(error).__name__}: {message}")
    raise SystemExit(1)