"""Run the full pipeline: clean data -> train logistic regression -> train random forest."""

from src.clean_data import main as clean_main
from src.train_logreg import main as logreg_main
from src.train_rf import main as rf_main


if __name__ == "__main__":
    print("=" * 50)
    print("Step 1: Cleaning data")
    print("=" * 50)
    clean_main()

    print("\n" + "=" * 50)
    print("Step 2: Training Logistic Regression")
    print("=" * 50)
    logreg_main()

    print("\n" + "=" * 50)
    print("Step 3: Training Random Forest")
    print("=" * 50)
    rf_main()

    print("\nDone! Check reports/ for metrics and figures.")
