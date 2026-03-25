from pathlib import Path
import numpy as np
import pandas as pd

from pycaret.classification import (
    setup,
    compare_models,
    tune_model,
    blend_models,
    stack_models,
    finalize_model,
    predict_model,
    pull,
    save_model,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
POWERBI_OUT = PROCESSED / "powerbi"
MODELS_OUT = ROOT / "models"
ML_OUT = PROCESSED / "ml"


def _score_column(df: pd.DataFrame) -> str:
    for c in ["prediction_score", "Score", "score", "Label"]:
        if c in df.columns:
            return c
    score_like = [c for c in df.columns if "score" in c.lower()]
    return score_like[0] if score_like else "prediction_label"


def _label_column(df: pd.DataFrame) -> str:
    for c in ["prediction_label", "Label", "label", "prediction"]:
        if c in df.columns:
            return c
    return "prediction_label"


def run_training() -> None:
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    ML_OUT.mkdir(parents=True, exist_ok=True)
    POWERBI_OUT.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED / "train_features.parquet"
    test_path = PROCESSED / "test_features.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    train_df["TARGET"] = train_df["TARGET"].astype(int)

    sample_size = min(120000, len(train_df))
    train_sample = train_df.sample(n=sample_size, random_state=42, replace=False)

    print(f"Train full shape: {train_df.shape}")
    print(f"Train sample shape for PyCaret compare/tune: {train_sample.shape}")
    print(f"Test shape: {test_df.shape}")

    exp = setup(
        data=train_sample,
        target="TARGET",
        session_id=42,
        fold=3,
        train_size=0.8,
        preprocess=True,
        imputation_type="simple",
        numeric_imputation="median",
        categorical_imputation="most_frequent",
        fix_imbalance=True,
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        n_jobs=-1,
        verbose=False,
    )

    top_models = compare_models(
        sort="AUC",
        n_select=5,
        turbo=True,
        errors="ignore",
    )
    leaderboard = pull().copy()
    leaderboard.to_csv(ML_OUT / "model_leaderboard.csv", index=False)
    leaderboard.to_parquet(ML_OUT / "model_leaderboard.parquet", index=False)

    if not isinstance(top_models, list):
        top_models = [top_models]

    tuned_models = []
    for m in top_models[:3]:
        try:
            tuned = tune_model(m, optimize="AUC", choose_better=True, verbose=False)
            tuned_models.append(tuned)
        except Exception:
            tuned_models.append(m)

    candidate_models = tuned_models.copy()

    if len(tuned_models) >= 2:
        try:
            blended = blend_models(estimator_list=tuned_models, optimize="AUC", verbose=False)
            candidate_models.append(blended)
        except Exception:
            pass

    if len(tuned_models) >= 2:
        try:
            stacked = stack_models(estimator_list=tuned_models, optimize="AUC", verbose=False)
            candidate_models.append(stacked)
        except Exception:
            pass

    metrics_rows = []
    best_model = None
    best_auc = -1

    for i, model in enumerate(candidate_models, start=1):
        pred = predict_model(model)
        m = pull().copy()
        m["candidate_index"] = i
        metrics_rows.append(m)

        auc = float(m["AUC"].iloc[0]) if "AUC" in m.columns else np.nan
        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_model = model

    metrics_df = pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame()
    metrics_df.to_csv(ML_OUT / "model_holdout_metrics.csv", index=False)
    if not metrics_df.empty:
        metrics_df.to_parquet(ML_OUT / "model_holdout_metrics.parquet", index=False)

    if best_model is None:
        best_model = candidate_models[0]

    final_model = finalize_model(best_model)
    save_model(final_model, str(MODELS_OUT / "best_pycaret_model"))

    train_scored = predict_model(final_model, data=train_df.drop(columns=["TARGET"]))
    test_scored = predict_model(final_model, data=test_df)

    train_scored["TARGET"] = train_df["TARGET"].values

    score_col_train = _score_column(train_scored)
    label_col_train = _label_column(train_scored)
    score_col_test = _score_column(test_scored)
    label_col_test = _label_column(test_scored)

    train_pred_out = pd.DataFrame(
        {
            "SK_ID_CURR": train_df["SK_ID_CURR"].values,
            "DATASET": "train",
            "TARGET": train_scored["TARGET"].values,
            "PRED_LABEL": train_scored[label_col_train].values,
            "PRED_SCORE": train_scored[score_col_train].values,
        }
    )

    test_pred_out = pd.DataFrame(
        {
            "SK_ID_CURR": test_df["SK_ID_CURR"].values,
            "DATASET": "test",
            "TARGET": np.nan,
            "PRED_LABEL": test_scored[label_col_test].values,
            "PRED_SCORE": test_scored[score_col_test].values,
        }
    )

    model_scores = pd.concat([train_pred_out, test_pred_out], ignore_index=True)
    model_scores["PRED_RISK_BAND"] = pd.qcut(
        model_scores["PRED_SCORE"].fillna(model_scores["PRED_SCORE"].median()),
        q=5,
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
        duplicates="drop",
    )

    train_pred_out.to_parquet(ML_OUT / "train_predictions.parquet", index=False)
    train_pred_out.to_csv(ML_OUT / "train_predictions.csv", index=False)
    test_pred_out.to_parquet(ML_OUT / "test_predictions.parquet", index=False)
    test_pred_out.to_csv(ML_OUT / "test_predictions.csv", index=False)

    model_scores.to_parquet(POWERBI_OUT / "fact_model_scores.parquet", index=False)
    model_scores.to_csv(POWERBI_OUT / "fact_model_scores.csv", index=False)

    print("PyCaret training and scoring complete")
    print(f"Leaderboard rows: {len(leaderboard)}")
    print(f"Holdout metrics rows: {len(metrics_df)}")
    print(f"Model scores rows: {len(model_scores)}")
    print(f"Best holdout AUC: {best_auc:.6f}")


if __name__ == "__main__":
    run_training()
