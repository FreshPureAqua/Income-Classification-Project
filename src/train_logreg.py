import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from .data_loader import load_data
from .preprocess import split_xy, build_preprocessor
from .evaluate import evaluate_classifier
from .config import MODEL_DIR, RANDOM_SEED


def main():
    df = load_data()
    split = split_xy(df)

    preprocessor = build_preprocessor(split.X_train)

    logreg = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", logreg),
    ])

    pipeline.fit(split.X_train, split.y_train)

    metrics = evaluate_classifier(pipeline, split.X_test, split.y_test, name="logreg")
    print("Logistic Regression metrics:", metrics)

    joblib.dump(pipeline, MODEL_DIR / "logreg_pipeline.joblib")
    print("Saved model:", MODEL_DIR / "logreg_pipeline.joblib")


if __name__ == "__main__":
    main()
