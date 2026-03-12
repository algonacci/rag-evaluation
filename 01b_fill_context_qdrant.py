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


def extract_title_hint(query: str) -> str:
    patterns = [
        r"(?:article|document)\s+(.+?)(?:[!?]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" \"'*!?.")
    return ""


def title_match_score(query: str, text: str) -> int:
    title = extract_title_hint(query)
    if not title:
        return 0

    normalized_text = re.sub(r"\s+", " ", text.lower())
    normalized_title = re.sub(r"\s+", " ", title.lower())
    if normalized_title in normalized_text:
        return 10_000

    title_tokens = set(tokenize(title))
    text_tokens = set(tokenize(text))
    overlap = len(title_tokens & text_tokens)
    if not title_tokens:
        return 0

    coverage = overlap / len(title_tokens)
    return int(coverage * 1000) + overlap


def file_name_match_score(query: str, file_name: str) -> int:
    title = extract_title_hint(query)
    if not title or not file_name:
        return 0

    title_tokens = set(tokenize(title))
    file_tokens = set(tokenize(file_name))
    overlap = len(title_tokens & file_tokens)
    if not title_tokens:
        return 0

    coverage = overlap / len(title_tokens)
    return int(coverage * 10000) + overlap


def candidate_score(query: str, text: str, file_name: str = "") -> tuple[int, int]:
    combined = f"{file_name} {text}".strip()
    return (
        file_name_match_score(query, file_name),
        title_match_score(query, combined),
        keyword_overlap_score(query, combined),
    )


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
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "my_documents")
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

    question_col = None
    for candidate in ("Question", "Pertanyaan", "question"):
        if candidate in df.columns:
            question_col = candidate
            break
    if not question_col:
        raise ValueError("No Question/Pertanyaan column found in input.")

    context_col = "Context" if "Context" in df.columns else ("context" if "context" in df.columns else "Context")
    if context_col not in df.columns:
        df[context_col] = None
    df[context_col] = df[context_col].astype(object)

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
            contexts.append((text, str(payload.get("file_name", ""))))
        contexts.sort(key=lambda item: candidate_score(question, item[0], item[1]), reverse=True)
        for i, (ctx, _) in enumerate(contexts[: args.topk], start=1):
            print(f"\n-- {i}")
            print(ctx[:500])
        return

    filled = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Retrieving contexts"):
        if not is_empty(row.get(context_col)):
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
            contexts.append((text, str(payload.get("file_name", ""))))

        contexts.sort(key=lambda item: candidate_score(str(question), item[0], item[1]), reverse=True)

        df.at[idx, context_col] = "\n\n---\n\n".join(text for text, _ in contexts[: args.topk])
        filled += 1

    if args.output.lower().endswith(".csv"):
        df.to_csv(args.output, index=False)
    else:
        df.to_excel(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Filled contexts: {filled}")


if __name__ == "__main__":
    main()
