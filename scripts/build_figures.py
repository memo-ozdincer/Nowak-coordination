#!/usr/bin/env python3
"""Build figures exclusively from analyzer-produced tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aggregates = _rows(args.tables_dir / "aggregates.csv")
    models = sorted({row["model"] for row in aggregates})
    p_cc = [
        sum(float(row["mean_p_cc"]) for row in aggregates if row["model"] == model)
        / sum(row["model"] == model for row in aggregates)
        for model in models
    ]
    fig, axis = plt.subplots(figsize=(6, 3.5))
    axis.bar(models, p_cc, color="#4472c4")
    axis.set(ylabel="Mean P(CC)", ylim=(0, 1), title="Synthetic outcome check")
    fig.tight_layout()
    fig.savefig(args.output_dir / "outcomes.png", dpi=120, metadata={"Software": "Gate3"})
    plt.close(fig)

    forecasts = _rows(args.tables_dir / "forecast_skill.csv")
    fig, axis = plt.subplots(figsize=(6, 3.5))
    axis.bar(
        [row["model"] for row in forecasts],
        [float(row["brier_skill_score"]) for row in forecasts],
        color="#70ad47",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(ylabel="Brier Skill Score", title="Forecast skill vs frozen EMA")
    fig.tight_layout()
    fig.savefig(args.output_dir / "forecast_skill.png", dpi=120, metadata={"Software": "Gate3"})
    plt.close(fig)


if __name__ == "__main__":
    main()
