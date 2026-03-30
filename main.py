import os
import pandas as pd
import logging
import traceback
from src.data_cleaning import load_data as load_raw_data, clean_dataset, save_data as save_cleaned_data
from src.bias_injection import load_data as load_cleaned_data, add_bias_features, create_counterfactuals
from src.feature_pipeline import prepare_dataframe, build_preprocessor, create_feature_pipeline
from src.model import (
    create_target, prepare_features, build_lr_model, train_model,
    plot_confusion, plot_selection, demographic_parity, disparate_impact, counterfactual,
    build_rf_model
)

# Configure logging
logging.basicConfig(
    filename='pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

def run_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "data", "resume", "resume_data.csv")
    cleaned_file = os.path.join(base_dir, "data", "processed", "cleaned_resume_data_v2.csv")
    bias_dataset_file = os.path.join(base_dir, "data", "processed", "bias_dataset_v2.csv")
    processed_dir = os.path.join(base_dir, "data", "processed")
    
    os.makedirs(processed_dir, exist_ok=True)

    # -----------------------------
    # STEP 1: DATA CLEANING
    # -----------------------------
    msg = "STEP 1: DATA CLEANING"
    print("\n" + "="*len(msg))
    print(msg)
    print("="*len(msg))
    logger.info(msg)
    
    logger.info(f"Loading raw data from: {input_file}")
    df_raw = load_raw_data(input_file)
    logger.info("Cleaning dataset...")
    df_cleaned = clean_dataset(df_raw)
    logger.info(f"Saving cleaned data to: {cleaned_file}")
    save_cleaned_data(df_cleaned, cleaned_file)

    # -----------------------------
    # STEP 2: BIAS INJECTION
    # -----------------------------
    msg = "STEP 2: BIAS INJECTION"
    print("\n" + "="*len(msg))
    print(msg)
    print("="*len(msg))
    logger.info(msg)
    
    logger.info(f"Loading cleaned data from: {cleaned_file}")
    df_for_bias = load_cleaned_data(cleaned_file)
    logger.info("Adding bias features...")
    df_with_bias = add_bias_features(df_for_bias)
    logger.info("Creating counterfactuals...")
    df_final_bias = create_counterfactuals(df_with_bias)
    
    logger.info(f"Saving bias dataset to: {bias_dataset_file}")
    df_final_bias.to_csv(bias_dataset_file, index=False)

    # -----------------------------
    # STEP 3: FEATURE PIPELINE
    # -----------------------------
    msg = "STEP 3: FEATURE PIPELINE"
    print("\n" + "="*len(msg))
    print(msg)
    print("="*len(msg))
    logger.info(msg)
    
    df_features, cat_cols = prepare_dataframe(df_final_bias)
    text_col = "skills_text"
    preprocessor = build_preprocessor(text_col, cat_cols)
    pipeline = create_feature_pipeline(preprocessor)
    
    X_test_transform = df_features[[text_col] + cat_cols]
    X_transformed = pipeline.fit_transform(X_test_transform)
    print(f"Feature matrix shape: {X_transformed.shape}")
    logger.info(f"Feature matrix shape: {X_transformed.shape}")

    # -----------------------------
    # STEP 4: MODEL TRAINING & EVALUATION
    # -----------------------------
    msg = "STEP 4: MODEL TRAINING & EVALUATION"
    print("\n" + "="*len(msg))
    print(msg)
    print("="*len(msg))
    logger.info(msg)
    
    df_model = create_target(df_final_bias)
    X, y, text_col, cat_cols = prepare_features(df_model)

    # Logistic Regression
    print("\n--- Logistic Regression ---")
    lr_model = build_lr_model(text_col, cat_cols)
    lr_model, X_test, y_test, lr_preds = train_model(lr_model, X, y, "Logistic Regression")
    
    df_test_lr = df_model.loc[X_test.index]
    plot_confusion(y_test, lr_preds, "LR")
    plot_selection(df_test_lr, lr_preds, "gender", "LR")
    demographic_parity(df_test_lr, lr_preds, "gender")
    disparate_impact(df_test_lr, lr_preds, "gender")
    counterfactual(lr_model, df_model, text_col, cat_cols, "LR")

    # Random Forest
    print("\n--- Random Forest ---")
    rf_model = build_rf_model(text_col, cat_cols)
    rf_model, X_test, y_test, rf_preds = train_model(rf_model, X, y, "Random Forest")
    
    df_test_rf = df_model.loc[X_test.index]
    plot_confusion(y_test, rf_preds, "RF")
    plot_selection(df_test_rf, rf_preds, "gender", "RF")
    demographic_parity(df_test_rf, rf_preds, "gender")
    disparate_impact(df_test_rf, rf_preds, "gender")
    counterfactual(rf_model, df_model, text_col, cat_cols, "RF")

    print("\n" + "="*30)
    print("PIPELINE COMPLETED SUCCESSFULLY ✅")
    print("Plots saved in /plots folder.")
    print("="*30)

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        logger.error("PIPELINE FAILED", exc_info=True)
        exit(1)
