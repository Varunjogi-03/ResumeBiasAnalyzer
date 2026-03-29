import pandas as pd
import re
import ast

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Original shape:", df.shape)
    return df


# -----------------------------
# CLEAN TEXT FUNCTION
# -----------------------------
def clean_text(text):
    if pd.isnull(text):
        return ""

    text = str(text)

    # remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # lowercase
    text = text.lower()

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# -----------------------------
# CONVERT STRING LIST TO LIST
# -----------------------------
def convert_to_list(val):
    if pd.isnull(val):
        return []

    try:
        return ast.literal_eval(val)
    except:
        return [val]


# -----------------------------
# MAIN CLEANING FUNCTION
# -----------------------------
def clean_dataset(df):

    # -------------------------
    # HANDLE MISSING VALUES
    # -------------------------
    df = df.fillna({
        "skills": "[]",
        "degree_names": "[]",
        "educational_institution_name": "",
        "locations": "",
        "job_position_name": "",
        "responsibilities": "",
        "company_names": "",
        "experience_requirement": ""
    })

    # -------------------------
    # CONVERT LIST COLUMNS
    # -------------------------
    list_columns = [
        "skills",
        "degree_names",
        "company_names",
        "positions",
        "locations"
    ]

    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(convert_to_list)

    # -------------------------
    # CLEAN TEXT COLUMNS
    # -------------------------
    text_columns = [
        "career_objective",
        "educational_institution_name",
        "job_position_name",
        "responsibilities"
    ]

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # -------------------------
    # NORMALIZE SKILLS
    # -------------------------
    if "skills" in df.columns:
        df["skills"] = df["skills"].apply(
            lambda x: list(set([clean_text(skill) for skill in x]))
        )

        df["skills_count"] = df["skills"].apply(len)

    # -------------------------
    # CLEAN NUMERIC FIELDS
    # -------------------------
    if "passing_years" in df.columns:
        df["passing_years"] = pd.to_numeric(df["passing_years"], errors='coerce')

    # -------------------------
    # DROP DUPLICATES
    # -------------------------
    # Convert list columns to string for duplicate removal
    list_columns = df.select_dtypes(object).columns

    df_temp = df.copy()

    for col in list_columns:
     df_temp[col] = df_temp[col].apply(lambda x: str(x))

# Drop duplicates safely
    df_temp = df_temp.drop_duplicates()

# Restore original df structure (keep cleaned version)
    df = df.loc[df_temp.index]

    print("Cleaned shape:", df.shape)

    return df


# -----------------------------
# SAVE DATA
# -----------------------------
def save_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to: {output_path}")


# -----------------------------
# MAIN RUN
# -----------------------------
if __name__ == "__main__":
    input_file = "data/resume/resume_data.csv"
    output_file = "data/processed/cleaned_resume_data.csv"

    df = load_data(input_file)

    df = clean_dataset(df)

    print("\nSample after cleaning:")
    print(df.head())

    save_data(df, output_file)