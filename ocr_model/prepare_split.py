import csv
import random
from pathlib import Path

ROOT = Path(__file__).parent
INPUT_CSV = ROOT / "dataset.csv"
TRAIN_CSV = ROOT / "train.csv"
VAL_CSV = ROOT / "val.csv"

VAL_RATIO = 0.1
SEED = 42
MAX_PLATE_SLOTS = 10


def main():
    random.seed(SEED)

    with open(INPUT_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    before = len(rows)
    rows = [r for r in rows if len(r["text"]) <= MAX_PLATE_SLOTS]
    dropped = before - len(rows)
    if dropped:
        print(f"Отфильтровано записей длиннее {MAX_PLATE_SLOTS} символов: {dropped}")

    by_region: dict[str, list[dict]] = {}
    for row in rows:
        by_region.setdefault(row["region"], []).append(row)

    train_rows = []
    val_rows = []

    for region, region_rows in by_region.items():
        random.shuffle(region_rows)
        n_val = max(1, int(len(region_rows) * VAL_RATIO))
        val_rows.extend(region_rows[:n_val])
        train_rows.extend(region_rows[n_val:])
        print(f"[{region}] всего {len(region_rows)}: train={len(region_rows) - n_val}, val={n_val}")

    def write(path: Path, data: list[dict]):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "plate_text", "plate_region"])
            for row in data:
                rel_path = Path(row["image_path"]).relative_to(ROOT)
                plate_region = "United States" if row["region"] == "us" else "Unknown"
                writer.writerow([rel_path.as_posix(), row["text"], plate_region])

    write(TRAIN_CSV, train_rows)
    write(VAL_CSV, val_rows)

    print()
    print(f"train.csv: {len(train_rows)} записей -> {TRAIN_CSV}")
    print(f"val.csv:   {len(val_rows)} записей -> {VAL_CSV}")


if __name__ == "__main__":
    main()
