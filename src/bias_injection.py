import pandas as pd
import random

# -----------------------------
# LOAD CLEANED DATA
# -----------------------------
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Loaded:", df.shape)
    return df


# -----------------------------
# ADD SYNTHETIC BIAS FEATURES
# -----------------------------
def add_bias_features(df):

    # Gender
    df["gender"] = [random.choice(["male", "female"]) for _ in range(len(df))]

    # College tier (based on institution name)
    def assign_college_tier(name):
        if pd.isnull(name):
            return "tier3"

        name = str(name).lower()

        if any(x in name for x in ["iit", "nit", "mit", "stanford"]):
            return "tier1"
        elif any(x in name for x in ["state", "university", "college"]):
            return "tier2"
        else:
            return "tier3"

    df["college_tier"] = df["educational_institution_name"].apply(assign_college_tier)

    # Gap year
    df["gap_year"] = [random.choice(["yes", "no"]) for _ in range(len(df))]

    # English level (based on resume text quality proxy)
    df["english_level"] = [
        "fluent" if random.random() > 0.3 else "basic"
        for _ in range(len(df))
    ]

    return df


# -----------------------------
# CREATE COUNTERFACTUAL DATA
# -----------------------------
def create_counterfactuals(df):

    counterfactual_data = []

    for _, row in df.iterrows():

        # Original
        counterfactual_data.append(row.copy())

        # Gender flipped
        new_row = row.copy()
        new_row["gender"] = "female" if row["gender"] == "male" else "male"
        counterfactual_data.append(new_row)

        # College tier changed
        new_row = row.copy()
        new_row["college_tier"] = "tier3" if row["college_tier"] == "tier1" else "tier1"
        counterfactual_data.append(new_row)

        # Gap year flipped
        new_row = row.copy()
        new_row["gap_year"] = "no" if row["gap_year"] == "yes" else "yes"
        counterfactual_data.append(new_row)

    df_new = pd.DataFrame(counterfactual_data).reset_index(drop=True)

    print("After counterfactual expansion:", df_new.shape)

    return df_new


# -----------------------------
# MAIN RUN
# -----------------------------
if __name__ == "__main__":

    input_file = "data/processed/cleaned_resume_data.csv"
    output_file = "data/processed/bias_dataset.csv"

    df = load_data(input_file)

    # Add bias features
    df = add_bias_features(df)

    print("\nSample with bias features:")
    print(df[["gender", "college_tier", "gap_year", "english_level"]].head())

    # Create counterfactuals
    df = create_counterfactuals(df)

    df.to_csv(output_file, index=False)

    print(f"\nSaved to: {output_file}")