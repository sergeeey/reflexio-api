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
app = FastAPI(title="Reflexio Search API", version="1.0.0")


# =======================
# EMBEDDINGS
# =======================
def embed(text):
    resp = oa.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding


# =======================
# SEARCH
# =======================
def search(query: str, match_count: int = 5):
    emb = embed(query)

    msgs = client.rpc("match_vectors", {
        "query_embedding": emb,
        "match_count": match_count
    }).execute().data or []

    sums = client.rpc("match_vectors_summaries", {
        "query_embedding": emb,
        "match_count": match_count
    }).execute().data or []

    results = []

    for r in msgs:
        results.append({
            "id": r.get("id"),
            "text": r.get("content"),
            "similarity": r.get("similarity") if r.get("similarity") is not None else 0.0,
            "source": r.get("metadata", {}).get("source") if isinstance(r.get("metadata"), dict) else r.get("metadata"),
            "type": "message"
        })

    for r in sums:
        results.append({
            "id": r.get("id"),
            "text": r.get("summary"),
            "similarity": r.get("similarity") if r.get("similarity") is not None else 0.0,
            "source": r.get("source"),
            "type": "summary"
        })

    # сортируем по similarity
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return results


# =======================
# API ROUTES
# =======================
@app.get("/search")
def api_search(q: str = Query(..., description="Вопрос пользователя"), top_k: int = 5):
    """
    Поиск по базе (messages + summaries).
    Возвращает JSON с топ-N результатами.
    """
    return search(q, top_k)


@app.get("/")
def root():
    return {"status": "ok", "message": "Reflexio Search API работает 🚀"}
