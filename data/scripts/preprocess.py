"""
Phase 1 — Dataset Preprocessor
Merges all raw datasets → cleans → converts to Qwen chat format →
splits train/val/test → saves as HuggingFace Dataset.

Usage:
    python data/scripts/preprocess.py
"""

import json
import random
import re
from pathlib import Path
from datasets import Dataset, DatasetDict
from rich.console import Console
from rich.progress import track

console = Console()
RAW_DIR   = Path(__file__).parent.parent / "raw"
PROC_DIR  = Path(__file__).parent.parent / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are MediMind, a knowledgeable and empathetic medical AI assistant. "
    "Provide accurate, evidence-based health information based on established medical "
    "knowledge. Always recommend consulting a qualified healthcare professional for "
    "personal medical decisions, diagnoses, or treatment plans. "
    "Cite sources when available."
)


# ── Cleaning helpers ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove HTML tags, extra whitespace, and fix common artifacts."""
    text = re.sub(r"<[^>]+>", " ", text)              # strip HTML
    text = re.sub(r"\s+", " ", text)                   # collapse whitespace
    text = re.sub(r"http\S+", "[URL]", text)           # anonymize URLs
    return text.strip()


def is_valid(question: str, answer: str) -> bool:
    """Quality filter: reject very short or suspiciously long records."""
    if len(question.strip()) < 10:
        return False
    if len(answer.strip()) < 20:
        return False
    if len(answer.strip()) > 4000:   # truncate guard
        return False
    return True


# ── Format converters ─────────────────────────────────────────────────────────

def to_chat_format(record: dict) -> dict:
    """Convert a cleaned record to Qwen multi-turn chat format."""
    user_content = clean_text(record["question"])
    if record.get("context"):
        user_content += f"\n\nContext:\n{clean_text(record['context'])}"

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": clean_text(record["answer"])},
        ],
        "source":   record.get("source", "unknown"),
        "category": record.get("category", "general"),
    }


def apply_chat_template_text(record: dict) -> dict:
    """
    Flatten messages into a single 'text' field using Qwen's format.
    TRL SFTTrainer can consume this directly.
    """
    parts = []
    for msg in record["messages"]:
        role    = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")   # generation prompt
    return {**record, "text": "\n".join(parts)}


# ── Main pipeline ─────────────────────────────────────────────────────────────

def load_raw(filename: str) -> list[dict]:
    path = RAW_DIR / filename
    if not path.exists():
        console.print(f"[yellow]⚠ {filename} not found — skipping")
        return []
    with open(path) as f:
        return json.load(f)


def preprocess():
    console.rule("[bold magenta]MediMind AI — Phase 1: Preprocessing")

    all_records = []

    # Load each raw file
    for fname in ["medquad.json", "pubmedqa.json", "medmcqa.json"]:
        raw = load_raw(fname)
        console.print(f"  Loaded {fname}: {len(raw):,} records")

        for rec in track(raw, description=f"Cleaning {fname}..."):
            q = rec.get("question", "")
            a = rec.get("answer", "")
            if not is_valid(q, a):
                continue
            all_records.append(to_chat_format(rec))

    console.print(f"\n[green]Total after cleaning: {len(all_records):,}")

    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(all_records)

    # Apply chat template
    console.print("[cyan]Applying Qwen chat template...")
    all_records = [apply_chat_template_text(r) for r in all_records]

    # Train / validation / test split  (90 / 5 / 5)
    n = len(all_records)
    n_val  = int(n * 0.05)
    n_test = int(n * 0.05)

    test_data  = all_records[:n_test]
    val_data   = all_records[n_test: n_test + n_val]
    train_data = all_records[n_test + n_val:]

    # Save as HuggingFace DatasetDict
    ds = DatasetDict({
        "train":      Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test":       Dataset.from_list(test_data),
    })

    out_path = PROC_DIR / "medimind_train"
    ds.save_to_disk(str(out_path))

    # Also save test set as JSON for later RAGAS evaluation
    with open(PROC_DIR / "test_set.json", "w") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    # Print summary
    console.rule("[bold magenta]Preprocessing Summary")
    console.print(f"  Train   : {len(train_data):,} samples")
    console.print(f"  Val     : {len(val_data):,} samples")
    console.print(f"  Test    : {len(test_data):,} samples")
    console.print(f"  Saved to: {out_path}")

    # Show one example
    console.rule("[cyan]Example training sample")
    sample = train_data[0]
    console.print(f"Source  : {sample['source']}")
    console.print(f"Category: {sample['category']}")
    console.print(f"\n[white]{sample['text'][:600]}...[/white]")

    console.print(f"\n[bold green]✓ Done. Next step → Phase 2: python training/train.py[/bold green]")


if __name__ == "__main__":
    preprocess()