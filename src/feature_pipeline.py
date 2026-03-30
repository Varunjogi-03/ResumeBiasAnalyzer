import pandas as pd
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(path):
    df = pd.read_csv(path)
    print("Loaded:", df.shape)
    return df


# -----------------------------
# CONVERT SKILLS → TEXT
# -----------------------------
def skills_to_text(val):
    if pd.isnull(val):
        return ""

    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return " ".join([str(x) for x in parsed])
    except:
        pass

    return str(val)


# -----------------------------
# PREPARE FEATURES
# -----------------------------
def prepare_dataframe(df):

    # Convert skills
    df["skills_text"] = df["skills"].apply(skills_to_text)

    # Ensure categorical columns exist
    categorical_cols = []

    for col in ["gender", "college_tier", "gap_year", "english_level"]:
        if col in df.columns:
            categorical_cols.append(col)

    print("Using categorical columns:", categorical_cols)

    return df, categorical_cols


# -----------------------------
# BUILD PREPROCESSOR
# -----------------------------
def build_preprocessor(text_col, categorical_cols):

    transformers = [
        ("text", TfidfVectorizer(max_features=500), text_col)
    ]

    if len(categorical_cols) > 0:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        )

    preprocessor = ColumnTransformer(transformers)

    return preprocessor


# -----------------------------
# CREATE FINAL PIPELINE
# -----------------------------
def create_feature_pipeline(preprocessor):
    pipeline = Pipeline([
        ("preprocessing", preprocessor)
    ])
    return pipeline


# -----------------------------
# MAIN TEST
# -----------------------------
if __name__ == "__main__":

    df = load_data("data/processed/bias_dataset.csv")

    df, cat_cols = prepare_dataframe(df)

    text_col = "skills_text"

    preprocessor = build_preprocessor(text_col, cat_cols)

    pipeline = create_feature_pipeline(preprocessor)

    # Test transformation
    X = df[[text_col] + cat_cols]

    X_transformed = pipeline.fit_transform(X)

    print("\nFeature matrix shape:", X_transformed.shape)