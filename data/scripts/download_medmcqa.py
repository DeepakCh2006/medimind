"""
Phase 1 — MedMCQA Dataset Downloader
Source  : HuggingFace Hub (openlifescienceai/MedMCQA)
Size    : 182,822 MCQs
License : MIT
"""

import json
import random
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.progress import track

console = Console()

RAW_DIR = Path(__file__).parent.parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_MAP = {
    0: "Anaesthesia",
    1: "Anatomy",
    2: "Biochemistry",
    3: "Dental",
    4: "ENT",
    5: "Forensic Medicine",
    6: "Gynaecology & Obstetrics",
    7: "Medicine",
    8: "Microbiology",
    9: "Ophthalmology",
    10: "Orthopaedics",
    11: "Pathology",
    12: "Pediatrics",
    13: "Pharmacology",
    14: "Physiology",
    15: "Psychiatry",
    16: "Radiology",
    17: "Skin",
    18: "Social & Preventive Medicine",
    19: "Surgery",
}


def format_question(row):
    return (
        f"{str(row.get('question', '')).strip()}\n\n"
        f"Options:\n"
        f"A) {row.get('opa', '')}\n"
        f"B) {row.get('opb', '')}\n"
        f"C) {row.get('opc', '')}\n"
        f"D) {row.get('opd', '')}"
    )


def format_answer(row):
    options = [
        row.get("opa", ""),
        row.get("opb", ""),
        row.get("opc", ""),
        row.get("opd", ""),
    ]

    cop = row.get("cop")

    try:
        cop = int(cop)
    except Exception:
        cop = 0

    # MedMCQA stores cop as 1-4
    cop = max(1, min(cop, 4))

    letters = ["A", "B", "C", "D"]

    correct_letter = letters[cop - 1]
    correct_text = options[cop - 1]

    explanation = str(row.get("exp") or "").strip()

    answer = f"The correct answer is {correct_letter}) {correct_text}."

    if explanation:
        answer += f"\n\nExplanation:\n{explanation}"

    return answer


def download_medmcqa():
    console.rule("[bold blue]Downloading MedMCQA")

    ds = load_dataset("openlifescienceai/MedMCQA")

    console.print("[green]✓ MedMCQA loaded successfully[/green]")

    records = []
    skipped = 0

    for split_name in ["train", "validation"]:

        if split_name not in ds:
            continue

        for row in track(
            ds[split_name],
            description=f"Processing {split_name}..."
        ):

            exp = str(row.get("exp") or "").strip()

            if len(exp) < 15:
                skipped += 1
                continue

            subject = row.get("subject_name")

            if isinstance(subject, int):
                subject = SUBJECT_MAP.get(subject, "Medicine")
            elif subject is None:
                subject = "Medicine"
            else:
                subject = str(subject)

            records.append(
                {
                    "question": format_question(row),
                    "answer": format_answer(row),
                    "context": "",
                    "source": "MedMCQA",
                    "category": subject,
                }
            )

    random.seed(42)

    if len(records) > 50000:
        records = random.sample(records, 50000)

    output_file = RAW_DIR / "medmcqa.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            records,
            f,
            indent=2,
            ensure_ascii=False
        )

    console.print(
        f"[bold green]✓ MedMCQA saved → {output_file}"
    )
    console.print(
        f"Total records kept: {len(records):,}"
    )
    console.print(
        f"Skipped records: {skipped:,}"
    )

    return records


if __name__ == "__main__":

    data = download_medmcqa()

    if data:
        console.print("\n[cyan]Sample record:[/cyan]")
        console.print(data[0])