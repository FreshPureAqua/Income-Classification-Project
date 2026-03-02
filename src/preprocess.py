from dataclasses import dataclass
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from .config import TARGET_COL, RANDOM_SEED
from .category_maps import apply_groupings, COLS_TO_DROP_FOR_MODELING


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply groupings and drop raw columns that were replaced."""
    df = apply_groupings(df)
    drop = [c for c in COLS_TO_DROP_FOR_MODELING if c in df.columns]
    df = df.drop(columns=drop)
    return df


def split_xy(df: pd.DataFrame) -> SplitData:
    df = prepare_features(df)

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y if y.nunique() == 2 else None,
    )

    return SplitData(X_train, X_test, y_train, y_test)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )

    return preprocessor
