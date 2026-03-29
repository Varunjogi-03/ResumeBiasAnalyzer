import pandas as pd
import numpy as np
import re
import ast

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(path):
    df = pd.read_csv(path)
    print("Loaded:", df.shape)
    print("Columns:\n", df.columns.tolist())   # IMPORTANT DEBUG
    return df


# -----------------------------
# SAFE SKILL TEXT CONVERSION
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
# FIND EXPERIENCE COLUMN AUTOMATICALLY
# -----------------------------
def get_experience_column(df):
    for col in df.columns:
        if "experience" in col.lower():
            return col
    return None


def extract_experience_number(text):
    if pd.isnull(text):
        return 0

    text = str(text).lower()
    match = re.search(r'(\d+)', text)

    if match:
        return int(match.group(1))
    return 0


# -----------------------------
# CREATE TARGET
# -----------------------------
def create_target(df):

    # Find experience column dynamically
    exp_col = get_experience_column(df)

    if exp_col:
        print("Using experience column:", exp_col)
        df["experience_num"] = df[exp_col].apply(extract_experience_number)
    else:
        print("No experience column found → using 0")
        df["experience_num"] = 0

    # Ensure skills_count exists
    if "skills_count" not in df.columns:
        print("skills_count missing → generating from skills")
        df["skills_text"] = df["skills"].apply(skills_to_text)
        df["skills_count"] = df["skills_text"].apply(lambda x: len(x.split()))
    else:
        df["skills_text"] = df["skills"].apply(skills_to_text)

    # Score
    df["score"] = (
        df["skills_count"].fillna(0) * 0.6 +
        df["experience_num"].fillna(0) * 0.4
    )

    threshold = df["score"].median()
    df["selected"] = (df["score"] >= threshold).astype(int)

    return df


# -----------------------------
# PREPARE FEATURES
# -----------------------------
def prepare_features(df):

    text_col = "skills_text"

    categorical_cols = []

    # only include if present
    for col in ["gender", "college_tier", "gap_year", "english_level"]:
        if col in df.columns:
            categorical_cols.append(col)

    X = df[[text_col] + categorical_cols]
    y = df["selected"]

    return X, y, text_col, categorical_cols


# -----------------------------
# BUILD MODEL
# -----------------------------
def build_model(text_col, categorical_cols):

    transformers = [
        ("text", TfidfVectorizer(max_features=500), text_col)
    ]

    if len(categorical_cols) > 0:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        )

    preprocessor = ColumnTransformer(transformers)

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    return model


# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model(model, X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))

    return model, X_test, y_test


# -----------------------------
# FAIRNESS
# -----------------------------
def demographic_parity(df, preds, attribute):
    if attribute not in df.columns:
        return

    df = df.copy()
    df["pred"] = preds

    rates = df.groupby(attribute)["pred"].mean()
    print(f"\nDemographic Parity ({attribute}):")
    print(rates)


def disparate_impact(df, preds, attribute):
    if attribute not in df.columns:
        return

    df = df.copy()
    df["pred"] = preds

    rates = df.groupby(attribute)["pred"].mean()

    if len(rates) >= 2:
        ratio = rates.min() / rates.max()
        print(f"\nDisparate Impact ({attribute}): {ratio:.3f}")


# -----------------------------
# COUNTERFACTUAL TEST
# -----------------------------
def counterfactual_test(model, df, text_col, categorical_cols):

    if "gender" not in df.columns:
        print("\nSkipping counterfactual test (no gender column)")
        return

    print("\nCounterfactual Testing (Gender Flip):")

    sample = df.sample(50, random_state=42).copy()

    sample_flipped = sample.copy()
    sample_flipped["gender"] = sample_flipped["gender"].apply(
        lambda x: "female" if x == "male" else "male"
    )

    X_orig = sample[[text_col] + categorical_cols]
    X_flip = sample_flipped[[text_col] + categorical_cols]

    pred_orig = model.predict(X_orig)
    pred_flip = model.predict(X_flip)

    diff = np.abs(pred_orig - pred_flip)

    print("Changed predictions:", diff.sum(), "out of", len(diff))


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    df = load_data("data/processed/bias_dataset.csv")

    df = create_target(df)

    X, y, text_col, cat_cols = prepare_features(df)

    model = build_model(text_col, cat_cols)

    model, X_test, y_test = train_model(model, X, y)

    preds = model.predict(X_test)

    df_test = df.loc[X_test.index]

    demographic_parity(df_test, preds, "gender")
    demographic_parity(df_test, preds, "college_tier")

    disparate_impact(df_test, preds, "gender")

    counterfactual_test(model, df, text_col, cat_cols)