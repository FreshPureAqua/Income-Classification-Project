from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "adult_income.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "adult_income_clean.csv"
FEATURED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "adult_income_featured.csv"

TARGET_COL = "income"     
RANDOM_SEED = 42

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
METRICS_DIR = REPORT_DIR / "metrics"

for d in [MODEL_DIR, REPORT_DIR, FIG_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
