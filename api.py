import os
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI
from fastapi import FastAPI, Query

# =======================
# ENV
# =======================
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY or not OPENAI_API_KEY:
    raise RuntimeError("❌ Проверь .env: SUPABASE_URL, SUPABASE_ANON_KEY, OPENAI_API_KEY")

client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
oa = OpenAI(api_key=OPENAI_API_KEY)

# =======================
# FASTAPI INIT
# =======================
app = FastAPI(title="Reflexio Search API", version="1.1.0")


# =======================
# EMBEDDING
# =======================
def embed(text: str) -> list[float]:
    resp = oa.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding


# =======================
# SEARCH FUNCTION
# =======================
def hybrid_search(query: str, match_count: int = 5):
    embedding = embed(query)

    # Векторный поиск по messages
    messages = client.rpc("match_vectors", {
        "query_embedding": embedding,
        "match_count": match_count
    }).execute().data or []

    # Векторный поиск по summaries
    summaries = client.rpc("match_vectors_summaries", {
        "query_embedding": embedding,
        "match_count": match_count
    }).execute().data or []

    combined = []

    for r in messages:
        combined.append({
            "id": r.get("id"),
            "text": r.get("content"),
            "similarity": r.get("similarity", 0.0),
            "source": r.get("metadata", {}).get("source") if isinstance(r.get("metadata"), dict) else r.get("metadata"),
            "type": "message"
        })

    for r in summaries:
        combined.append({
            "id": r.get("id"),
            "text": r.get("summary"),
            "similarity": r.get("similarity", 0.0),
            "source": r.get("source"),
            "type": "summary"
        })

    # Если есть векторные совпадения — возвращаем их
    if combined:
        sorted_results = sorted(combined, key=lambda x: x["similarity"], reverse=True)
        return {
            "mode": "embedding",
            "results": sorted_results[:match_count]
        }

    # Fallback: поиск по ILIKE
    text_matches = client.table("reflexio_messages") \
        .select("*") \
        .ilike("content", f"%{query}%") \
        .limit(match_count) \
        .execute().data or []

    fallback = []
    for r in text_matches:
        fallback.append({
            "id": r.get("id"),
            "text": r.get("content"),
            "similarity": None,
            "source": r.get("metadata", {}).get("source") if isinstance(r.get("metadata"), dict) else r.get("metadata"),
            "type": "message"
        })

    return {
        "mode": "text-fallback",
        "results": fallback
    }


# =======================
# ROUTES
# =======================
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Reflexio Search API работает 🚀"
    }


@app.get("/search")
def api_search(
    q: str = Query(..., description="Поисковый запрос"),
    top_k: int = Query(5, description="Сколько результатов вернуть")
):
    return hybrid_search(q, top_k)
