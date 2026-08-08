import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "cct_s_v2"

MODEL_CONFIG = MODEL_DIR / "cct_s_v2_global_model_config.yaml"
PLATE_CONFIG = MODEL_DIR / "cct_s_v2_global_plate_config.yaml"
WEIGHTS = MODEL_DIR / "cct_s_v2_global.keras"

TRAIN_CSV = ROOT / "train.csv"
VAL_CSV = ROOT / "val.csv"
OUTPUT_DIR = ROOT / "trained_models"

EPOCHS = 70
BATCH_SIZE = 32
LR = 0.0005
WEIGHT_DECAY = 0.0005
EARLY_STOPPING_PATIENCE = 15

REGION_LOSS_WEIGHT = 0.0
PLATE_LOSS_WEIGHT = 1.0


def main():
    for path in (MODEL_CONFIG, PLATE_CONFIG, WEIGHTS, TRAIN_CSV, VAL_CSV):
        if not path.exists():
            raise FileNotFoundError(f"Не найден обязательный файл: {path}")

    cmd = [
        "fast-plate-ocr", "train",
        "--model-config-file", str(MODEL_CONFIG),
        "--plate-config-file", str(PLATE_CONFIG),
        "--annotations", str(TRAIN_CSV),
        "--val-annotations", str(VAL_CSV),
        "--weights-path", str(WEIGHTS),
        "--output-dir", str(OUTPUT_DIR),
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--weight-decay", str(WEIGHT_DECAY),
        "--early-stopping-patience", str(EARLY_STOPPING_PATIENCE),
        "--region-loss-weight", str(REGION_LOSS_WEIGHT),
        "--plate-loss-weight", str(PLATE_LOSS_WEIGHT),
    ]

    print("Запуск:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
