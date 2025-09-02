import os
import json
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

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
# EMBEDDINGS
# =======================
def embed(text):
    resp = oa.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding


# =======================
# SEARCH
# =======================
def search(query, match_count=5):
    emb = embed(query)

    msgs = client.rpc("match_vectors", {
        "query_embedding": emb,
        "match_count": match_count
    }).execute().data or []

    sums = client.rpc("match_vectors_summaries", {
        "query_embedding": emb,
        "match_count": match_count
    }).execute().data or []

    norm = []
    for r in msgs:
        norm.append({
            "id": r.get("id"),
            "text": r.get("content"),
            "source": r.get("metadata", {}).get("source") if isinstance(r.get("metadata"), dict) else r.get("metadata"),
            "similarity": r.get("similarity"),
            "type": "message"
        })
    for r in sums:
        norm.append({
            "id": r.get("id"),
            "text": r.get("summary"),
            "source": r.get("source"),
            "similarity": r.get("similarity"),
            "type": "summary"
        })

    norm = sorted(norm, key=lambda x: (x["similarity"] or 0), reverse=True)
    return norm


# =======================
# MAIN LOOP
# =======================
def main():
    print("🔎 Тест поиска в Supabase (единый топ, без GPT)\n")

    while True:
        q = input("❓ Вопрос (или 'exit' для выхода): ").strip()
        if q.lower() in {"exit", "quit", "выход"}:
            print("👋 Выход.")
            break

        results = search(q, match_count=5)
        if not results:
            print("❌ Ничего не найдено.")
            continue

        for r in results:
            sim = r["similarity"] if r["similarity"] is not None else 0.0
            print(f"[{r['type']}] {r['source']} (id={r['id']}, sim={sim:.3f})")
            if r["text"]:
                print("   " + r["text"][:200].replace("\n", " ") + "...")
            print()

        with open("last_search.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("💾 Результаты сохранены в last_search.json\n")


if __name__ == "__main__":
    main()
