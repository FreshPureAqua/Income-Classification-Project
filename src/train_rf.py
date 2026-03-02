import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from .data_loader import load_data
from .preprocess import split_xy, build_preprocessor
from .evaluate import evaluate_classifier
from .config import MODEL_DIR, RANDOM_SEED


def main():
    df = load_data()
    split = split_xy(df)

    preprocessor = build_preprocessor(split.X_train)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", rf),
    ])

    pipeline.fit(split.X_train, split.y_train)

    metrics = evaluate_classifier(pipeline, split.X_test, split.y_test, name="random_forest")
    print("Random Forest metrics:", metrics)

    joblib.dump(pipeline, MODEL_DIR / "rf_pipeline.joblib")
    print("Saved model:", MODEL_DIR / "rf_pipeline.joblib")


if __name__ == "__main__":
    main()
