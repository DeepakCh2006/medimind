"""
Phase 1 — Master Dataset Downloader
Runs all three downloaders and prints a final summary report.

Usage:
    python data/scripts/download_all.py
"""

import json
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()
RAW_DIR = Path(__file__).parent.parent / "raw"


def run_all():
    console.rule("[bold magenta]MediMind AI — Phase 1: Dataset Download")
    start = time.time()

    results = {}

    # --- MedQuAD ---
    try:
        from download_medquad import download_medquad
        data = download_medquad()
        results["MedQuAD"] = {"count": len(data), "status": "✓"}
    except Exception as e:
        console.print(f"[red]MedQuAD failed: {e}")
        results["MedQuAD"] = {"count": 0, "status": f"✗ {e}"}

    # --- PubMedQA ---
    try:
        from download_pubmedqa import download_pubmedqa
        data = download_pubmedqa()
        results["PubMedQA"] = {"count": len(data), "status": "✓"}
    except Exception as e:
        console.print(f"[red]PubMedQA failed: {e}")
        results["PubMedQA"] = {"count": 0, "status": f"✗ {e}"}

    # --- MedMCQA ---
    try:
        from download_medmcqa import download_medmcqa
        data = download_medmcqa()
        results["MedMCQA"] = {"count": len(data), "status": "✓"}
    except Exception as e:
        console.print(f"[red]MedMCQA failed: {e}")
        results["MedMCQA"] = {"count": 0, "status": f"✗ {e}"}

    # --- Summary table ---
    elapsed = time.time() - start
    console.rule("[bold magenta]Download Summary")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Dataset",  style="bold")
    table.add_column("Records",  justify="right")
    table.add_column("Status",   justify="center")
    table.add_column("File")

    total = 0
    for name, info in results.items():
        fname = name.lower().replace(" ", "_") + ".json"
        fpath = RAW_DIR / fname
        size_mb = fpath.stat().st_size / 1_048_576 if fpath.exists() else 0
        table.add_row(
            name,
            f"{info['count']:,}",
            f"[green]{info['status']}[/green]" if "✓" in info["status"] else f"[red]{info['status']}[/red]",
            f"{fname} ({size_mb:.1f} MB)",
        )
        total += info["count"]

    console.print(table)
    console.print(f"\n[bold]Total records : [cyan]{total:,}[/cyan]")
    console.print(f"[bold]Time elapsed  : [cyan]{elapsed:.1f}s[/cyan]")
    console.print(f"\n[bold green]Next step → run:  python data/scripts/preprocess.py[/bold green]")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent))
    run_all()