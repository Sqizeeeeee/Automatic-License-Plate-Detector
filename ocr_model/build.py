import csv
import re
from pathlib import Path

ROOT = Path(__file__).parent
OUTPUT_CSV = ROOT / "dataset.csv"

VALID_CHARS = re.compile(r"^[A-Z0-9]+$")

BLACKLIST = {"SAMPLE", "SAMPLES", "DEMO", "TEST", "EXAMPLE"}


def clean_label(raw: str) -> str | None:
    """Приводит сырой текст метки к верхнему регистру, проверяет на валидность."""
    text = raw.strip().upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    if not text or not VALID_CHARS.match(text):
        return None
    if text in BLACKLIST:
        return None
    return text


def parse_eu1(rows: list):
    """train/val/test подпапки, текст = имя файла (с возможным суффиксом _N)."""
    base = ROOT / "eu1"
    if not base.exists():
        print(f"[eu1] пропущено, папка не найдена: {base}")
        return

    count = 0
    for split in ("train", "val", "test"):
        split_dir = base / split
        if not split_dir.exists():
            continue
        for img_path in split_dir.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".png", ".jpeg"):
                continue
            stem = img_path.stem
            stem = re.sub(r"_\d+$", "", stem)
            label = clean_label(stem)
            if label is None:
                print(f"[eu1] пропущен файл с некорректной меткой: {img_path.name}")
                continue
            rows.append({
                "image_path": str(img_path),
                "text": label,
                "region": "eu",
                "source": "eu1",
                "split": split,
            })
            count += 1
    print(f"[eu1] добавлено записей: {count}")


def parse_endtoend_format(rows: list, folder_name: str, region: str, pad: int = 5):

    from PIL import Image

    base = ROOT / folder_name
    if not base.exists():
        print(f"[{folder_name}] пропущено, папка не найдена: {base}")
        return

    crops_dir = base / "crops"
    crops_dir.mkdir(exist_ok=True)

    count = 0
    skipped = 0
    for txt_path in base.glob("*.txt"):
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = txt_path.with_suffix(ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print(f"[{folder_name}] нет картинки для {txt_path.name}")
            skipped += 1
            continue

        content = txt_path.read_text().strip()
        parts = content.split()
        if len(parts) < 6:
            print(f"[{folder_name}] неожиданный формат в {txt_path.name}: {content!r}")
            skipped += 1
            continue

        try:
            x, y, w, h = map(int, parts[1:5])
        except ValueError:
            print(f"[{folder_name}] не удалось распарсить bbox в {txt_path.name}: {content!r}")
            skipped += 1
            continue

        raw_text = parts[-1]
        label = clean_label(raw_text)
        if label is None:
            print(
                f"[{folder_name}] пропущен файл с некорректной меткой: "
                f"{txt_path.name} -> {raw_text!r}"
            )
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[{folder_name}] не удалось открыть {img_path.name}: {e}")
            skipped += 1
            continue

        img_w, img_h = img.size
        left = max(0, x - pad)
        top = max(0, y - pad)
        right = min(img_w, x + w + pad)
        bottom = min(img_h, y + h + pad)

        if right <= left or bottom <= top:
            print(f"[{folder_name}] некорректный bbox в {txt_path.name}: {content!r}")
            skipped += 1
            continue

        crop = img.crop((left, top, right, bottom))
        crop_path = crops_dir / f"{txt_path.stem}.jpg"
        crop.save(crop_path, quality=95)

        rows.append({
            "image_path": str(crop_path),
            "text": label,
            "region": region,
            "source": folder_name,
            "split": "",
        })
        count += 1

    print(f"[{folder_name}] добавлено записей: {count}, пропущено: {skipped}")


def parse_us3(rows: list):
    """us3/usimages/*.png + us3/groundtruth.csv = 'filename,state,text'."""
    base = ROOT / "us3"
    csv_path = base / "groundtruth.csv"
    img_dir = base / "usimages"

    if not csv_path.exists():
        print(f"[us3] пропущено, не найден CSV: {csv_path}")
        return
    if not img_dir.exists():
        print(f"[us3] пропущено, не найдена папка картинок: {img_dir}")
        return

    count = 0
    skipped = 0
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                skipped += 1
                continue
            filename, state, raw_text = row[0].strip(), row[1].strip(), row[2].strip()
            img_path = img_dir / filename
            if not img_path.exists():
                print(f"[us3] файл из CSV не найден на диске: {filename}")
                skipped += 1
                continue

            label = clean_label(raw_text)
            if label is None:
                print(f"[us3] пропущен файл с некорректной меткой: {filename} -> {raw_text!r}")
                skipped += 1
                continue

            rows.append({
                "image_path": str(img_path),
                "text": label,
                "region": "us",
                "source": f"us3_{state}",
                "split": "",
            })
            count += 1

    print(f"[us3] добавлено записей: {count}, пропущено: {skipped}")


def main():
    rows: list[dict] = []

    parse_eu1(rows)
    parse_endtoend_format(rows, "eu2", region="eu")
    parse_endtoend_format(rows, "us2", region="us")
    parse_us3(rows)

    if not rows:
        print("Не собрано ни одной записи — проверьте пути и структуру папок.")
        return

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "text", "region", "source", "split"])
        writer.writeheader()
        writer.writerows(rows)

    eu_count = sum(1 for r in rows if r["region"] == "eu")
    us_count = sum(1 for r in rows if r["region"] == "us")

    print()
    print("=" * 50)
    print(f"Итого записей: {len(rows)}")
    print(f"  EU: {eu_count}")
    print(f"  US: {us_count}")
    print(f"Манифест сохранён: {OUTPUT_CSV}")
    print("=" * 50)


if __name__ == "__main__":

    main()
