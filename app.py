import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import asyncio
from flask import Flask, request, jsonify
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy

app = Flask(__name__)

# --- Inisialisasi Ragas & OpenAI ---
# Pastikan OPENAI_API_KEY ada di environment variable
client = AsyncOpenAI(
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"]
)
evaluator_llm = llm_factory("gpt-4o-mini", client=client)

# Kita buat instance metriknya di luar biar ga inisialisasi terus tiap request
metrics = [
    ContextPrecision(llm=evaluator_llm),
    ContextRecall(llm=evaluator_llm),
    Faithfulness(llm=evaluator_llm),
    AnswerRelevancy(llm=evaluator_llm)
]

RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "rag_evaluation_results.txt"


def normalize_retrieved_context(raw_context):
    if raw_context is None:
        return []

    if isinstance(raw_context, str):
        return [raw_context]

    if isinstance(raw_context, dict):
        return [json.dumps(raw_context, ensure_ascii=False)]

    if isinstance(raw_context, list):
        normalized = []
        for item in raw_context:
            if isinstance(item, str):
                normalized.append(item)
                continue
            if isinstance(item, dict):
                if "pageContent" in item and isinstance(item["pageContent"], str):
                    normalized.append(item["pageContent"])
                elif "text" in item and isinstance(item["text"], str):
                    normalized.append(item["text"])
                else:
                    normalized.append(json.dumps(item, ensure_ascii=False))
                continue
            normalized.append(str(item))
        return normalized

    return [str(raw_context)]


def save_evaluation_to_txt(question, answer, retrieved_contexts, reference, scores):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()

    lines = [
        "=" * 80,
        f"timestamp: {timestamp}",
        f"question: {question}",
        f"answer: {answer}",
        f"reference: {reference if reference is not None else ''}",
        "retrieved_contexts:",
    ]

    if retrieved_contexts:
        for idx, ctx in enumerate(retrieved_contexts, start=1):
            lines.append(f"  [{idx}] {ctx}")
    else:
        lines.append("  []")

    lines.append("scores:")
    for name, value in scores.items():
        lines.append(f"  {name}: {value}")

    lines.append("")

    with RESULTS_FILE.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


async def run_evaluation(question, answer, retrieved_context, reference=None):
    # Ragas butuh context dalam bentuk list of strings
    if isinstance(retrieved_context, str):
        retrieved_context = [retrieved_context]
    
    # Ground truth (reference) opsional tapi penting buat ContextRecall
    ref = reference if reference else answer 

    results = {}
    
    # Jalanin semua metrik secara async
    tasks = []
    for metric in metrics:
        # Kita panggil ascore untuk masing-masing metrik
        tasks.append(metric.ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=retrieved_context,
            reference=ref
        ))
    
    scores = await asyncio.gather(*tasks)
    
    # Map hasilnya ke dictionary
    for metric, score in zip(metrics, scores):
        results[metric.name] = score
        
    return results

@app.route("/rag_evaluation", methods=["POST"])
def rag_evaluation():
    data = request.get_json(silent=True) or {}
    
    # Validasi input
    required = ["question", "answer"]
    if not all(k in data for k in required):
        return jsonify({"success": False, "message": "Missing required fields"}), 400

    question = data["question"]
    answer = data["answer"]
    raw_context = data.get("retrieved_context")
    if raw_context is None:
        raw_context = data.get("retrieved_contexts")
    retrieved_context = normalize_retrieved_context(raw_context)
    reference = data.get("reference") # Optional ground truth

    try:
        # Karena Flask itu sync, kita panggil loop async di sini
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            evaluation_results = loop.run_until_complete(
                run_evaluation(question, answer, retrieved_context, reference)
            )
        finally:
            loop.close()
        save_evaluation_to_txt(
            question=question,
            answer=answer,
            retrieved_contexts=retrieved_context,
            reference=reference,
            scores=evaluation_results
        )

        return jsonify({
            "success": True,
            "message": "Success evaluated the RAG",
            "data": {
                "input": {
                    "question": question,
                    "answer": answer
                },
                "scores": evaluation_results
            }
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

if __name__ == "__main__":
    # Pakai port 5001 biar aman dari socket error tadi
    app.run(port=5001, debug=True)
