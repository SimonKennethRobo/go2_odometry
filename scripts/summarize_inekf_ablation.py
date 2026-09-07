#!/usr/bin/env python3
"""Create a compact comparison from multiple evaluate_inekf_bag metrics files."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="LABEL=METRICS_JSON",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for item in args.result:
        label, path = item.split("=", 1)
        metrics = json.loads(Path(path).read_text())
        rows.append(
            {
                "variant": label,
                "position_rmse_m": metrics["position_m"]["vector_rmse"],
                "orientation_rmse_rad": metrics["orientation_angle_rad"]["rmse"],
                "linear_velocity_rmse_mps": metrics["linear_velocity_world_mps"]["vector_rmse"],
            }
        )

    baseline = rows[0]
    improvements = {
        "position_rmse_m": "position_improvement_percent",
        "orientation_rmse_rad": "orientation_improvement_percent",
        "linear_velocity_rmse_mps": "linear_velocity_improvement_percent",
    }
    for row in rows:
        for key, output_key in improvements.items():
            row[output_key] = 100.0 * (
                baseline[key] - row[key]
            ) / baseline[key]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["variant"] for row in rows]
    fields = (
        ("position_rmse_m", "Position vector RMSE", "m"),
        ("orientation_rmse_rad", "Orientation angle RMSE", "rad"),
        ("linear_velocity_rmse_mps", "Linear velocity vector RMSE", "m/s"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.Blues(np.linspace(0.45, 0.9, len(rows)))
    for axis, (field, title, unit) in zip(axes, fields):
        values = [row[field] for row in rows]
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(True, axis="y", alpha=0.3)
        axis.tick_params(axis="x", rotation=18)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("Mocap fusion ablation on rosbag2_2026_09_02-15_04_25")
    fig.tight_layout()
    fig.savefig(args.output_dir / "ablation_rmse.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
