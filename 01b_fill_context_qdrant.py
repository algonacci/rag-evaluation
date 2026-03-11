import os
import json
import argparse
import re
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from urllib import request


load_dotenv()


def _http_post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: int = 60) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def embed_query(text: str, base_url: str, model: str) -> List[float]:
    url = f"{base_url.rstrip('/')}/v1/embeddings"
    payload = {"model": model, "input": text}
    headers = {"Content-Type": "application/json"}
    res = _http_post_json(url, payload, headers)
    data = res.get("data") or []
    if not data or "embedding" not in data[0]:
        raise ValueError(f"Unexpected embedding response keys: {list(res.keys())}")
    return data[0]["embedding"]


def qdrant_search(
    qdrant_url: str,
    collection: str,
    vector: List[float],
    top_k: int,
    api_key: str | None = None,
) -> List[Dict[str, Any]]:
    url = f"{qdrant_url.rstrip('/')}/collections/{collection}/points/search"
    payload = {
        "vector": vector,
        "limit": top_k,
        "with_payload": True,
        "with_vectors": False,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    res = _http_post_json(url, payload, headers)
    return res.get("result", [])


def extract_text_from_payload(payload: Dict[str, Any], text_keys: List[str]) -> str:
    for key in text_keys:
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def is_empty(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    if isinstance(val, str) and not val.strip():
        return True
    return False


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(t) >= 2]


def keyword_overlap_score(query: str, text: str) -> int:
    q = set(tokenize(query))
    if not q:
        return 0
    t = set(tokenize(text))
    return len(q & t)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill empty Context column via Qdrant top-k retrieval.")
    parser.add_argument("--input", default="data_awal.xlsx", help="Input Excel/CSV file path")
    parser.add_argument("--output", default="data_awal_with_context.csv", help="Output CSV/Excel file path")
    parser.add_argument("--topk", type=int, default=int(os.getenv("TOP_K", "4")), help="Top-K documents")
    parser.add_argument("--prefilter-k", type=int, default=int(os.getenv("PREFILTER_K", "20")), help="Initial fetch size before rerank/filter")
    parser.add_argument("--min-chars", type=int, default=int(os.getenv("MIN_CHARS", "80")), help="Minimum characters for context")
    parser.add_argument("--test-query", default=None, help="Run a single retrieval test and print results")
    args = parser.parse_args()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "Skripsi-RAG")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
    text_keys_env = os.getenv("QDRANT_TEXT_KEYS", "pageContent,text,content,document,chunk")
    text_keys = [k.strip() for k in text_keys_env.split(",") if k.strip()]

    if not qdrant_url:
        raise ValueError("QDRANT_URL is required in environment variables.")

    if args.input.lower().endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        df = pd.read_excel(args.input, skiprows=1)

    question_col = "Question" if "Question" in df.columns else ("Pertanyaan" if "Pertanyaan" in df.columns else None)
    if not question_col:
        raise ValueError("No Question/Pertanyaan column found in input.")

    if "Context" not in df.columns:
        df["Context"] = None

    if args.test_query:
        question = args.test_query
        vector = embed_query(str(question), ollama_base_url, embedding_model)
        hits = qdrant_search(qdrant_url, qdrant_collection, vector, args.prefilter_k, qdrant_api_key)
        contexts = []
        for hit in hits:
            payload = hit.get("payload") or {}
            text = extract_text_from_payload(payload, text_keys).strip()
            if len(text) < args.min_chars:
                continue
            contexts.append(text)
        contexts.sort(key=lambda t: keyword_overlap_score(question, t), reverse=True)
        for i, ctx in enumerate(contexts[: args.topk], start=1):
            print(f"\n-- {i}")
            print(ctx[:500])
        return

    filled = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Retrieving contexts"):
        if not is_empty(row.get("Context")):
            continue
        question = row.get(question_col, "")
        if is_empty(question):
            continue

        vector = embed_query(str(question), ollama_base_url, embedding_model)
        hits = qdrant_search(qdrant_url, qdrant_collection, vector, args.prefilter_k, qdrant_api_key)

        contexts = []
        for hit in hits:
            payload = hit.get("payload") or {}
            text = extract_text_from_payload(payload, text_keys).strip()
            if len(text) < args.min_chars:
                continue
            contexts.append(text)

        contexts.sort(key=lambda t: keyword_overlap_score(str(question), t), reverse=True)

        df.at[idx, "Context"] = "\n\n---\n\n".join(contexts[: args.topk])
        filled += 1

    if args.output.lower().endswith(".csv"):
        df.to_csv(args.output, index=False)
    else:
        df.to_excel(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Filled contexts: {filled}")


if __name__ == "__main__":
    main()
