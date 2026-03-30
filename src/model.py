import pandas as pd
import numpy as np
import re
import ast
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# -----------------------------
# CREATE PLOTS FOLDER
# -----------------------------
os.makedirs("plots", exist_ok=True)


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(path):
    df = pd.read_csv(path)
    print("Loaded:", df.shape)
    print("Columns:\n", df.columns.tolist())
    return df


# -----------------------------
# SKILLS → TEXT
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
# EXPERIENCE
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
    return int(match.group(1)) if match else 0


# -----------------------------
# CREATE TARGET
# -----------------------------
def create_target(df):

    exp_col = get_experience_column(df)

    if exp_col:
        print("Using experience column:", exp_col)
        df["experience_num"] = df[exp_col].apply(extract_experience_number)
    else:
        df["experience_num"] = 0

    df["skills_text"] = df["skills"].apply(skills_to_text)
    df["skills_count"] = df["skills_text"].apply(lambda x: len(x.split()))

    df["score"] = (
        df["skills_count"] * 0.6 +
        df["experience_num"] * 0.4
    )

    threshold = df["score"].median()
    df["selected"] = (df["score"] >= threshold).astype(int)

    return df


# -----------------------------
# FEATURES
# -----------------------------
def prepare_features(df):

    text_col = "skills_text"

    categorical_cols = []

    for col in ["gender", "college_tier", "gap_year", "english_level"]:
        if col in df.columns:
            categorical_cols.append(col)

    X = df[[text_col] + categorical_cols]
    y = df["selected"]

    return X, y, text_col, categorical_cols


# -----------------------------
# MODEL BUILDERS
# -----------------------------
def build_lr_model(text_col, categorical_cols):

    transformers = [
        ("text", TfidfVectorizer(max_features=500), text_col)
    ]

    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        )

    preprocessor = ColumnTransformer(transformers)

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    return model


def build_rf_model(text_col, categorical_cols):

    transformers = [
        ("text", TfidfVectorizer(max_features=500), text_col)
    ]

    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        )

    preprocessor = ColumnTransformer(transformers)

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])

    return model


# -----------------------------
# TRAIN
# -----------------------------
def train_model(model, X, y, model_name):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, preds))

    return model, X_test, y_test, preds


# -----------------------------
# PLOTS
# -----------------------------
def plot_confusion(y_test, preds, name):

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()

    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"plots/{name}_confusion.png")
    plt.close()


def plot_selection(df, preds, attr, name):

    if attr not in df.columns:
        return

    df = df.copy()
    df["pred"] = preds

    rates = df.groupby(attr)["pred"].mean()

    rates.plot(kind="bar")
    plt.title(f"{name}: Selection by {attr}")
    plt.savefig(f"plots/{name}_{attr}.png")
    plt.close()


def plot_counterfactual(changed, total, name):

    plt.bar(["Changed", "Unchanged"], [changed, total - changed])
    plt.title(f"{name} Counterfactual")

    plt.savefig(f"plots/{name}_counterfactual.png")
    plt.close()


# -----------------------------
# FAIRNESS
# -----------------------------
def demographic_parity(df, preds, attr):

    if attr not in df.columns:
        return

    df = df.copy()
    df["pred"] = preds

    rates = df.groupby(attr)["pred"].mean()

    print(f"\nDemographic Parity ({attr}):")
    print(rates)


def disparate_impact(df, preds, attr):

    if attr not in df.columns:
        return

    df = df.copy()
    df["pred"] = preds

    rates = df.groupby(attr)["pred"].mean()

    if len(rates) >= 2:
        print(f"Disparate Impact ({attr}): {rates.min()/rates.max():.3f}")


# -----------------------------
# COUNTERFACTUAL
# -----------------------------
def counterfactual(model, df, text_col, cat_cols, name):

    if "gender" not in df.columns:
        return

    sample = df.sample(50, random_state=42)

    flipped = sample.copy()
    flipped["gender"] = flipped["gender"].apply(
        lambda x: "female" if x == "male" else "male"
    )

    X1 = sample[[text_col] + cat_cols]
    X2 = flipped[[text_col] + cat_cols]

    p1 = model.predict(X1)
    p2 = model.predict(X2)

    diff = np.abs(p1 - p2)

    changed = diff.sum()
    total = len(diff)

    print(f"{name} Counterfactual: {changed}/{total}")

    plot_counterfactual(changed, total, name)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    df = load_data("data/processed/bias_dataset.csv")

    df = create_target(df)

    X, y, text_col, cat_cols = prepare_features(df)

    # ---------------- LR ----------------
    lr_model = build_lr_model(text_col, cat_cols)

    lr_model, X_test, y_test, lr_preds = train_model(
        lr_model, X, y, "Logistic Regression"
    )

    df_test = df.loc[X_test.index]

    plot_confusion(y_test, lr_preds, "LR")
    plot_selection(df_test, lr_preds, "gender", "LR")

    demographic_parity(df_test, lr_preds, "gender")
    disparate_impact(df_test, lr_preds, "gender")

    counterfactual(lr_model, df, text_col, cat_cols, "LR")

    # ---------------- RF ----------------
    rf_model = build_rf_model(text_col, cat_cols)

    rf_model, X_test, y_test, rf_preds = train_model(
        rf_model, X, y, "Random Forest"
    )

    df_test = df.loc[X_test.index]

    plot_confusion(y_test, rf_preds, "RF")
    plot_selection(df_test, rf_preds, "gender", "RF")

    demographic_parity(df_test, rf_preds, "gender")
    disparate_impact(df_test, rf_preds, "gender")

    counterfactual(rf_model, df, text_col, cat_cols, "RF")

    print("\nAll plots saved in /plots folder ✅")