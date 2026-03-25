from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
POWERBI_OUT = PROCESSED / "powerbi"
PARQUET_CACHE = PROCESSED / "parquet_cache"


def load_table(name: str) -> pd.DataFrame:
    pq = PARQUET_CACHE / f"{name}.parquet"
    csv = RAW / f"{name}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    df = pd.read_csv(csv)
    return df


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def build_datamart() -> None:
    POWERBI_OUT.mkdir(parents=True, exist_ok=True)

    app_train = load_table("application_train")
    app_test = load_table("application_test")
    bureau = load_table("bureau")
    prev = load_table("previous_application")
    inst = load_table("installments_payments")
    pos = load_table("POS_CASH_balance")
    cc = load_table("credit_card_balance")

    app_test = app_test.copy()
    app_test["TARGET"] = np.nan

    app_train["DATASET"] = "train"
    app_test["DATASET"] = "test"

    app = pd.concat([app_train, app_test], ignore_index=True)

    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    app["EXT_SOURCE_MEAN"] = app[ext_cols].mean(axis=1)
    app["RISK_SCORE_BASE"] = 1.0 - app["EXT_SOURCE_MEAN"]
    app["RISK_SCORE_BASE"] = app["RISK_SCORE_BASE"].fillna(app["RISK_SCORE_BASE"].median())
    app["RISK_SCORE_BASE"] = app["RISK_SCORE_BASE"].clip(0, 1)

    app["RISK_BAND"] = pd.qcut(
        app["RISK_SCORE_BASE"],
        q=5,
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
        duplicates="drop",
    )

    dim_customer_cols = [
        "SK_ID_CURR",
        "CODE_GENDER",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_INCOME_TYPE",
        "NAME_HOUSING_TYPE",
        "REGION_RATING_CLIENT",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "AMT_INCOME_TOTAL",
    ]
    dim_customer = app[dim_customer_cols].copy()
    dim_customer["AGE_YEARS"] = (-dim_customer["DAYS_BIRTH"] / 365).round(1)
    dim_customer["EMPLOYMENT_YEARS"] = np.where(
        dim_customer["DAYS_EMPLOYED"] < 0,
        (-dim_customer["DAYS_EMPLOYED"] / 365).round(1),
        np.nan,
    )

    fact_app_cols = [
        "SK_ID_CURR",
        "DATASET",
        "TARGET",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "AMT_INCOME_TOTAL",
        "NAME_CONTRACT_TYPE",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "RISK_SCORE_BASE",
        "RISK_BAND",
    ]
    fact_application = app[fact_app_cols].copy()

    bureau_fact = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_LOAN_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("CREDIT_ACTIVE", lambda s: (s == "Active").sum()),
        BUREAU_CLOSED_COUNT=("CREDIT_ACTIVE", lambda s: (s == "Closed").sum()),
        BUREAU_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
    ).reset_index()

    prev_fact = prev.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_COUNT=("NAME_CONTRACT_STATUS", lambda s: (s == "Approved").sum()),
        PREV_REFUSED_COUNT=("NAME_CONTRACT_STATUS", lambda s: (s == "Refused").sum()),
        PREV_AMT_APPLICATION_SUM=("AMT_APPLICATION", "sum"),
        PREV_AMT_CREDIT_SUM=("AMT_CREDIT", "sum"),
    ).reset_index()

    inst2 = inst.copy()
    inst2["DAYS_LATE"] = inst2["DAYS_ENTRY_PAYMENT"] - inst2["DAYS_INSTALMENT"]
    inst2["IS_LATE"] = (inst2["DAYS_LATE"] > 0).astype(int)
    installments_fact = inst2.groupby("SK_ID_CURR").agg(
        INSTAL_COUNT=("NUM_INSTALMENT_NUMBER", "count"),
        INST_DAYS_LATE_MEAN=("DAYS_LATE", "mean"),
        INST_DAYS_LATE_MAX=("DAYS_LATE", "max"),
        INST_LATE_RATE=("IS_LATE", "mean"),
        INST_PAYMENT_SUM=("AMT_PAYMENT", "sum"),
    ).reset_index()

    pos_fact = pos.groupby("SK_ID_CURR").agg(
        POS_RECORD_COUNT=("SK_ID_PREV", "count"),
        POS_DPD_MEAN=("SK_DPD", "mean"),
        POS_DPD_MAX=("SK_DPD", "max"),
        POS_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
        POS_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
    ).reset_index()

    cc2 = cc.copy()
    cc2["CC_UTILIZATION"] = safe_divide(cc2["AMT_BALANCE"], cc2["AMT_CREDIT_LIMIT_ACTUAL"])
    cc_fact = cc2.groupby("SK_ID_CURR").agg(
        CC_RECORD_COUNT=("SK_ID_PREV", "count"),
        CC_BALANCE_SUM=("AMT_BALANCE", "sum"),
        CC_DRAWINGS_SUM=("AMT_DRAWINGS_CURRENT", "sum"),
        CC_PAYMENT_SUM=("AMT_PAYMENT_TOTAL_CURRENT", "sum"),
        CC_UTILIZATION_MEAN=("CC_UTILIZATION", "mean"),
        CC_UTILIZATION_MAX=("CC_UTILIZATION", "max"),
    ).reset_index()

    risk_segment = fact_application.groupby("RISK_BAND", dropna=False).agg(
        CUSTOMER_COUNT=("SK_ID_CURR", "count"),
        DEFAULT_COUNT=("TARGET", lambda s: s.fillna(0).sum()),
        AVG_CREDIT=("AMT_CREDIT", "mean"),
        AVG_INCOME=("AMT_INCOME_TOTAL", "mean"),
    ).reset_index()
    risk_segment["DEFAULT_RATE"] = safe_divide(risk_segment["DEFAULT_COUNT"], risk_segment["CUSTOMER_COUNT"])

    kpi_snapshot = pd.DataFrame(
        {
            "TOTAL_CUSTOMERS": [fact_application["SK_ID_CURR"].nunique()],
            "TRAIN_CUSTOMERS": [(fact_application["DATASET"] == "train").sum()],
            "TEST_CUSTOMERS": [(fact_application["DATASET"] == "test").sum()],
            "DEFAULT_COUNT": [fact_application["TARGET"].fillna(0).sum()],
            "DEFAULT_RATE": [fact_application["TARGET"].mean()],
            "TOTAL_CREDIT_AMOUNT": [fact_application["AMT_CREDIT"].sum()],
            "AVG_CREDIT_AMOUNT": [fact_application["AMT_CREDIT"].mean()],
            "AVG_ANNUITY": [fact_application["AMT_ANNUITY"].mean()],
            "AVG_INCOME": [fact_application["AMT_INCOME_TOTAL"].mean()],
        }
    )

    outputs = {
        "dim_customer": dim_customer,
        "fact_application": fact_application,
        "fact_bureau": bureau_fact,
        "fact_previous_application": prev_fact,
        "fact_installments": installments_fact,
        "fact_pos_cash": pos_fact,
        "fact_credit_card": cc_fact,
        "dim_risk_segment": risk_segment,
        "kpi_snapshot": kpi_snapshot,
    }

    for name, df in outputs.items():
        df.to_parquet(POWERBI_OUT / f"{name}.parquet", index=False)
        df.to_csv(POWERBI_OUT / f"{name}.csv", index=False)

    print("Power BI datamart generated:")
    for name, df in outputs.items():
        print(f"- {name}: {df.shape}")


if __name__ == "__main__":
    build_datamart()
