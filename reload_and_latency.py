"""Повторная загрузка дообученного ruBERT в новом процессе и замер времени прямого прохода."""
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("artifacts/rubert-tiny2-sentiment")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MAX_LENGTH = 128
BATCH_SIZES = (1, 2, 4, 8, 16)
REPEATS = 3
TOLERANCE = 1e-5
DEVICE = torch.device("cpu")


def synchronize() -> None:
    """На GPU ждём окончания вычислений, на CPU вызов синхронный."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


def load_texts() -> dict[str, str]:
    with (DATA_DIR / "validation.jsonl").open(encoding="utf-8") as file:
        return {row["id"]: row["text"] for row in map(json.loads, file)}


def reload_check(model, tokenizer, texts: dict[str, str], labels: list[str]) -> dict:
    reference = json.loads((RESULTS_DIR / "reload_reference.json").read_text(encoding="utf-8"))
    inputs = tokenizer(texts[reference["id"]], truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].tolist()
    reloaded = dict(zip(labels, probs))
    diff = {label: abs(reloaded[label] - reference["probs"][label]) for label in labels}
    label_after = labels[max(range(len(labels)), key=probs.__getitem__)]
    return {
        "id": reference["id"],
        "label_before": reference["label"],
        "label_after": label_after,
        "probs_before": reference["probs"],
        "probs_after": reloaded,
        "abs_diff": diff,
        "max_abs_diff": max(diff.values()),
        "tolerance": TOLERANCE,
        "passed": label_after == reference["label"] and max(diff.values()) <= TOLERANCE,
        "device": str(DEVICE),
    }


def measure_latency(model, tokenizer, texts: list[str]) -> list[dict]:
    rows = []
    with torch.inference_mode():
        for batch_size in BATCH_SIZES:
            batches = [
                tokenizer(texts[start:start + batch_size], truncation=True, max_length=MAX_LENGTH,
                          padding=True, return_tensors="pt").to(DEVICE)
                for start in range(0, len(texts), batch_size)
            ]
            for batch in batches:  # прогрев: полный проход по набору
                model(**batch)
            times = []
            for _ in range(REPEATS):
                synchronize()
                start = time.perf_counter()
                for batch in batches:
                    model(**batch)
                synchronize()
                times.append(time.perf_counter() - start)
            median = statistics.median(times)
            real_tokens = sum(int(batch["attention_mask"].sum()) for batch in batches)
            all_tokens = sum(batch["attention_mask"].numel() for batch in batches)
            rows.append({
                "batch_size": batch_size,
                "batches": len(batches),
                "padding_share": round(1 - real_tokens / all_tokens, 3),
                "runs_ms": [round(t * 1000, 1) for t in times],
                "median_ms": round(median * 1000, 1),
                "ms_per_review": round(median * 1000 / len(texts), 3),
                "reviews_per_s": round(len(texts) / median, 1),
            })
    return rows


def main() -> None:
    protocol = json.loads((DATA_DIR / "protocol.json").read_text(encoding="utf-8"))
    texts = load_texts()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    check = reload_check(model, tokenizer, texts, labels)
    (RESULTS_DIR / "reload_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    latency_texts = [texts[i] for i in protocol["latency_validation_ids"]]
    latency = {"device": str(DEVICE), "torch_threads": torch.get_num_threads(), "torch": torch.__version__,
               "reviews": len(latency_texts), "rows": measure_latency(model, tokenizer, latency_texts)}
    (RESULTS_DIR / "latency.json").write_text(json.dumps(latency, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"reload: {check['label_before']} → {check['label_after']}, max |Δp| = {check['max_abs_diff']:.2e}, "
          f"passed = {check['passed']}")
    print(f"latency: {latency['device']}, {latency['torch_threads']} потоков, {latency['reviews']} отзывов")


if __name__ == "__main__":
    main()
