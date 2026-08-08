import argparse
import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).parent
DATASET_CSV = ROOT / "dataset.csv"
DEBUG_DIR = ROOT / "debug"

PER_GRID = 12


def load_rows(region: str | None, source: str | None) -> list[dict]:
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Не найден {DATASET_CSV}, сначала запустите build.py")

    with open(DATASET_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    if region:
        rows = [r for r in rows if r["region"] == region]
    if source:
        rows = [r for r in rows if r["source"] == source]

    return rows


def save_grid(rows: list[dict], grid_index: int):
    n = len(rows)
    cols = 4
    rows_count = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 3, rows_count * 3))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, row in zip(axes, rows):
        img_path = row["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"ERROR\n{e}", ha="center", va="center", wrap=True)

        caption = f"{row['text']}\n[{row['region']}/{row['source']}]"
        ax.set_title(caption, fontsize=9)
        ax.axis("off")

    for ax in axes[len(rows):]:
        ax.axis("off")

    fig.tight_layout()
    out_path = DEBUG_DIR / f"debug_grid_{grid_index:02d}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Сохранено: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="сколько записей проверить")
    parser.add_argument("--region", type=str, default=None, choices=["eu", "us"])
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    rows = load_rows(args.region, args.source)
    if not rows:
        print("Нет записей под заданные фильтры.")
        return

    sample_size = min(args.n, len(rows))
    sample = random.sample(rows, sample_size)

    DEBUG_DIR.mkdir(exist_ok=True)

    for grid_index, start in enumerate(range(0, len(sample), PER_GRID)):
        chunk = sample[start:start + PER_GRID]
        save_grid(chunk, grid_index)

    print(f"\nВсего проверено записей: {sample_size}")
    print(f"Результаты в: {DEBUG_DIR}")


if __name__ == "__main__":

    main()
