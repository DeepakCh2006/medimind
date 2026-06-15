"""
Phase 1 — MedQuAD Dataset Downloader
Source  : Hugging Face Hub
Dataset : keivalya/MedQuad-MedicalQnADataset
License : CC BY 4.0
"""

import json
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.progress import track

console = Console()

# Create raw data directory
RAW_DIR = Path(__file__).parent.parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_medquad():
    """Download and save MedQuAD dataset."""

    console.rule("[bold blue]Downloading MedQuAD")

    try:
        # Load dataset from Hugging Face
        ds = load_dataset("keivalya/MedQuad-MedicalQnADataset")
        console.print("[green]✓ MedQuAD loaded successfully[/green]")

    except Exception as e:
        console.print(f"[bold red]Failed to load dataset:[/bold red] {e}")
        raise

    # Select train split
    if "train" in ds:
        split = ds["train"]
    else:
        split = ds[list(ds.keys())[0]]

    records = []

    for row in track(split, description="Processing MedQuAD..."):

        question = (
            row.get("question")
            or row.get("Question")
            or row.get("query")
            or ""
        )

        answer = (
            row.get("answer")
            or row.get("Answer")
            or row.get("response")
            or ""
        )

        question = str(question).strip()
        answer = str(answer).strip()

        # Skip empty entries
        if not question or not answer:
            continue

        # Skip very short answers
        if len(answer) < 20:
            continue

        records.append(
            {
                "question": question,
                "answer": answer,
                "source": "MedQuAD",
                "category": row.get("focus_area", "general"),
            }
        )

    # Save dataset
    output_file = RAW_DIR / "medquad.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            records,
            f,
            indent=2,
            ensure_ascii=False
        )

    console.print(
        f"[bold green]✓ MedQuAD saved → {output_file}[/bold green]"
    )
    console.print(
        f"[bold cyan]Total records: {len(records):,}[/bold cyan]"
    )

    return records


if __name__ == "__main__":

    data = download_medquad()

    if len(data) > 0:
        console.print("\n[cyan]Sample record:[/cyan]")
        console.print(data[0])
    else:
        console.print("[red]No records found.[/red]")