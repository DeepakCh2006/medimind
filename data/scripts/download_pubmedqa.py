"""
Phase 1 — PubMedQA Dataset Downloader
Source  : HuggingFace Hub (qiaojin/PubMedQA)
Size    : 1,000 labeled + 61,200 unlabeled + 211,269 artificially generated
We use  : pqa_labeled (1k gold) + pqa_artificial (211k silver)
License : MIT
"""

import json
from pathlib import Path
from datasets import load_dataset
from rich.console import Console
from rich.progress import track

console = Console()
RAW_DIR = Path(__file__).parent.parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_pubmedqa():
    console.rule("[bold blue]Downloading PubMedQA")

    records = []

    # --- Part A: Gold labeled (1,000 high-quality expert answers) ---
    console.print("[cyan]Loading pqa_labeled (gold standard)...")
    ds_labeled = load_dataset("qiaojin/PubMedQA", "pqa_labeled", trust_remote_code=True)
    split = ds_labeled["train"]

    for row in track(split, description="Processing labeled..."):
        # Combine up to 3 context abstracts as background
        contexts = row["context"]["contexts"][:3]
        context_text = "\n\n".join(contexts)

        long_answer = row.get("long_answer", "").strip()
        if not long_answer or len(long_answer) < 30:
            continue

        records.append({
            "question": row["question"].strip(),
            "answer":   long_answer,
            "context":  context_text,
            "label":    row.get("final_decision", ""),   # yes/no/maybe
            "source":   "PubMedQA-labeled",
            "category": "biomedical_research",
        })

    console.print(f"  [green]Labeled records: {len(records):,}")

    # --- Part B: Artificially generated (silver, larger volume) ---
    console.print("[cyan]Loading pqa_artificial (silver, 211k)...")
    ds_artificial = load_dataset("qiaojin/PubMedQA", "pqa_artificial", trust_remote_code=True)
    split_art = ds_artificial["train"]

    art_records = []
    for row in track(split_art, description="Processing artificial..."):
        contexts = row["context"]["contexts"][:2]
        context_text = "\n\n".join(contexts)

        long_answer = row.get("long_answer", "").strip()
        if not long_answer or len(long_answer) < 50:
            continue

        art_records.append({
            "question": row["question"].strip(),
            "answer":   long_answer,
            "context":  context_text,
            "label":    row.get("final_decision", ""),
            "source":   "PubMedQA-artificial",
            "category": "biomedical_research",
        })

    # Sample 15,000 from artificial to keep dataset balanced
    import random
    random.seed(42)
    art_sample = random.sample(art_records, min(15_000, len(art_records)))
    records.extend(art_sample)

    out_path = RAW_DIR / "pubmedqa.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    console.print(f"[bold green]✓ PubMedQA saved → {out_path}")
    console.print(f"  Total records : {len(records):,}")
    return records


if __name__ == "__main__":
    data = download_pubmedqa()
    console.print(f"\n[cyan]Sample record:[/cyan]")
    console.print(data[0])
    