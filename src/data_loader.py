import pandas as pd
from .config import DATA_PATH, CLEAN_DATA_PATH

def load_data(use_cleaned: bool = True) -> pd.DataFrame:
    if use_cleaned and CLEAN_DATA_PATH.exists():
        return pd.read_csv(CLEAN_DATA_PATH)
    return pd.read_csv(DATA_PATH)