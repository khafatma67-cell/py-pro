"""
Student Performance Predictor - training pipeline
--------------------------------------------------
1. Load data (UCI Student Performance, or a synthetic fallback for demo)
2. Exploratory data analysis (plots saved to ./outputs)
3. Train and compare 3 models with cross-validation
4. Evaluate the best model on a held-out test set
5. Save the model (model.joblib) + metrics (outputs/metrics.json)

Usage:
    python train.py
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
DATA_PATH = Path("data/student-mat.csv")  # download from UCI / Kaggle (sep=';')
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

NUMERIC = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
]
CATEGORICAL = [
    "sex", "address", "famsup", "schoolsup", "paid",
    "activities", "higher", "internet", "romantic",
]
FEATURES = NUMERIC + CATEGORICAL
PASS_MARK = 10  # final grade G3 (0-20) >= 10 means "pass"


# ----------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------
def make_synthetic(n: int = 1000) -> pd.DataFrame:
    """Synthetic data with the same columns, used only if the real CSV is missing."""
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame({
        "age": rng.integers(15, 23, n),
        "Medu": rng.integers(0, 5, n),
        "Fedu": rng.integers(0, 5, n),
        "traveltime": rng.integers(1, 5, n),
        "studytime": rng.integers(1, 5, n),
        "failures": rng.choice([0, 1, 2, 3], n, p=[0.75, 0.15, 0.06, 0.04]),
        "famrel": rng.integers(1, 6, n),
        "freetime": rng.integers(1, 6, n),
        "goout": rng.integers(1, 6, n),
        "Dalc": rng.integers(1, 6, n),
        "Walc": rng.integers(1, 6, n),
        "health": rng.integers(1, 6, n),
        "absences": rng.poisson(5, n),
        "sex": rng.choice(["F", "M"], n),
        "address": rng.choice(["U", "R"], n, p=[0.75, 0.25]),
        "famsup": rng.choice(["yes", "no"], n),
        "schoolsup": rng.choice(["yes", "no"], n, p=[0.15, 0.85]),
        "paid": rng.choice(["yes", "no"], n),
        "activities": rng.choice(["yes", "no"], n),
        "higher": rng.choice(["yes", "no"], n, p=[0.9, 0.1]),
        "internet": rng.choice(["yes", "no"], n, p=[0.85, 0.15]),
        "romantic": rng.choice(["yes", "no"], n, p=[0.35, 0.65]),
    })
    score = (
        9
        + 1.4 * df["studytime"]
        - 2.6 * df["failures"]
        - 0.18 * df["absences"]
        - 0.5 * df["goout"]
        - 0.3 * df["Walc"]
        + 0.4 * (df["Medu"] + df["Fedu"])
        + 2.0 * (df["higher"] == "yes")
        + rng.normal(0, 2.2, n)
    )
    df["G3"] = np.clip(np.round(score), 0, 20).astype(int)
    return df


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        print(f"[data] loading real dataset: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, sep=";")
    else:
        print("[data] real dataset not found -> using SYNTHETIC demo data.")
        print(f"       Put the UCI file at {DATA_PATH} for real results.")
        df = make_synthetic()
    df["passed"] = (df["G3"] >= PASS_MARK).astype(int)
    return df


# ----------------------------------------------------------------------
# 2. EDA
# ----------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(5, 4))
    df["passed"].value_counts().sort_index().plot.bar(ax=ax, color=["#d9534f", "#5cb85c"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Fail", "Pass"], rotation=0)
    ax.set_title("Target distribution")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_target.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[NUMERIC + ["passed"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_correlation.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["studytime", "failures", "absences"]):
        sns.boxplot(data=df, x="passed", y=col, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Fail", "Pass"])
        ax.set_title(f"{col} vs result")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_key_features.png", dpi=150)
    plt.close(fig)
    print("[eda] plots saved to outputs/")


# ----------------------------------------------------------------------
# 3. Models
# ----------------------------------------------------------------------
def build_pipeline(model) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def main() -> None:
    df = load_data()
    print(f"[data] {len(df)} rows | pass rate = {df['passed'].mean():.1%}")
    run_eda(df)

    X, y = df[FEATURES], df["passed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    for name, model in candidates.items():
        pipe = build_pipeline(model)
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "cv_f1_mean": round(float(cv_f1.mean()), 4),
            "test_accuracy": round(accuracy_score(y_test, pred), 4),
            "test_f1": round(f1_score(y_test, pred), 4),
            "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
        }
        fitted[name] = pipe
        print(f"[model] {name:20s} {results[name]}")

    best_name = max(results, key=lambda k: results[k]["cv_f1_mean"])
    best = fitted[best_name]
    print(f"\n[best] {best_name}")
    print(classification_report(y_test, best.predict(X_test), target_names=["Fail", "Pass"]))

    # Confusion matrix + ROC
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        best, X_test, y_test, display_labels=["Fail", "Pass"], cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC curves")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    # Permutation importance (works on the raw input columns)
    imp = permutation_importance(
        best, X_test, y_test, n_repeats=15, random_state=SEED, scoring="f1"
    )
    imp_df = (
        pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
        .sort_values("importance")
        .tail(12)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp_df["feature"], imp_df["importance"], color="#337ab7")
    ax.set_title("Top features (permutation importance)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    # Save artifacts
    joblib.dump(best, "model.joblib")
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)
    print("[save] model.joblib + outputs/metrics.json")


if __name__ == "__main__":
    main()
