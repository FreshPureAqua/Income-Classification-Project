import pandas as pd
from .config import DATA_PATH, CLEAN_DATA_PATH, FEATURED_DATA_PATH, TARGET_COL
from .category_maps import apply_groupings

def main():
    df = pd.read_csv(DATA_PATH)

    # 1) Convert the Adult-style " ?" missing values to NA
    df = df.replace(" ?", pd.NA)

    # 2) Strip whitespace from ALL text columns
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

    # 3) Fix target labels: remove trailing periods like "<=50K."
    if TARGET_COL in df.columns:
        df[TARGET_COL] = (
            df[TARGET_COL]
            .astype(str)
            .str.strip()
            .str.replace(r"\.$", "", regex=True)
        )

    # 4) Drop rows missing the target
    df = df.dropna(subset=[TARGET_COL])

    # 5) Drop duplicate rows
    df = df.drop_duplicates()

    # Save cleaned data (original columns only)
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print("Saved cleaned dataset to:", CLEAN_DATA_PATH)

    # Save featured dataset (with collapsed bins added)
    df_featured = apply_groupings(df)
    FEATURED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_featured.to_csv(FEATURED_DATA_PATH, index=False)
    print("Saved featured dataset to:", FEATURED_DATA_PATH)

    print("Target unique values:", df[TARGET_COL].unique())
    print("Target counts:\n", df[TARGET_COL].value_counts())

if __name__ == "__main__":
    main()
