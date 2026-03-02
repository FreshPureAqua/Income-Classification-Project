import pandas as pd

EDUCATION_MAP = {
    "Preschool": "Before HS",
    "1st-4th": "Before HS",
    "5th-6th": "Before HS",
    "7th-8th": "Before HS",
    "9th": "Before HS",
    "10th": "Before HS",
    "11th": "Before HS",

    "12th": "High School",
    "HS-grad": "High School",

    "Some-college": "Some College",
    "Assoc-acdm": "Some College",
    "Assoc-voc": "Some College",

    "Bachelors": "Bachelors",
    "Masters": "Masters",
    "Doctorate": "Doctorate",
    "Prof-school": "Professional",
}

MARITAL_MAP = {
    "Never-married": "Never Married",
    "Married-civ-spouse": "Married",
    "Married-AF-spouse": "Married",
    "Divorced": "Previously Married",
    "Separated": "Previously Married",
    "Widowed": "Previously Married",
    "Married-spouse-absent": "Previously Married",
}

AGE_BINS = [17, 25, 35, 50, 65, 100]
AGE_LABELS = ["Early Career", "Establishing", "Peak", "Late Career", "Retirement"]

HOURS_BINS = [0, 10, 25, 39, 40, 60, 100]
HOURS_LABELS = ["0-10", "11-25", "26-39", "40", "41-60", "60+"]

COUNTRY_MAP = {
    # North America & US Territories
    "United-States": "North America",
    "Canada": "North America",
    "Puerto-Rico": "North America",
    "Outlying-US(Guam-USVI-etc)": "North America",

    # Latin America & Caribbean
    "Mexico": "Latin America & Caribbean",
    "Cuba": "Latin America & Caribbean",
    "Jamaica": "Latin America & Caribbean",
    "Dominican-Republic": "Latin America & Caribbean",
    "El-Salvador": "Latin America & Caribbean",
    "Guatemala": "Latin America & Caribbean",
    "Honduras": "Latin America & Caribbean",
    "Nicaragua": "Latin America & Caribbean",
    "Haiti": "Latin America & Caribbean",
    "Trinadad&Tobago": "Latin America & Caribbean",
    "Columbia": "Latin America & Caribbean",
    "Peru": "Latin America & Caribbean",
    "Ecuador": "Latin America & Caribbean",
    "South": "Latin America & Caribbean",

    # Europe
    "England": "Europe",
    "Germany": "Europe",
    "Italy": "Europe",
    "Poland": "Europe",
    "Portugal": "Europe",
    "France": "Europe",
    "Ireland": "Europe",
    "Greece": "Europe",
    "Scotland": "Europe",
    "Hungary": "Europe",
    "Yugoslavia": "Europe",
    "Holand-Netherlands": "Europe",

    # Asia
    "China": "Asia",
    "Japan": "Asia",
    "Taiwan": "Asia",
    "Hong": "Asia",
    "Philippines": "Asia",
    "Vietnam": "Asia",
    "Thailand": "Asia",
    "Cambodia": "Asia",
    "Laos": "Asia",
    "India": "Asia",
    "Iran": "Asia",
}


def apply_groupings(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all categorical groupings used in the analysis."""
    df = df.copy()

    if "education" in df.columns:
        df["education_grouped"] = df["education"].map(EDUCATION_MAP)
        df["education_grouped"] = df["education_grouped"].fillna(df["education"])

    if "marital-status" in df.columns:
        df["marital_grouped"] = df["marital-status"].map(MARITAL_MAP)

    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False
        )

    if "hours-per-week" in df.columns:
        df["hours_group"] = pd.cut(
            df["hours-per-week"], bins=HOURS_BINS, labels=HOURS_LABELS,
            include_lowest=True,
        )

    if "native-country" in df.columns:
        df["region"] = df["native-country"].map(COUNTRY_MAP)
        df["region"] = df["region"].fillna("Unknown")

    return df


COLS_TO_DROP_FOR_MODELING = [
    "education",
    "marital-status",
    "native-country",
    "fnlwgt",
    "education-num",
]
