"""Train and evaluate the ML suitability models (classification + regression) on fused land profiles."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBRegressor

from features import CLIMATE_COLS, LAND_COLS, NDVI_COLS, WATER_COLS

ROOT = Path(__file__).resolve().parent.parent
MODELS, OUTPUTS = ROOT / "models", ROOT / "outputs"
SEED = 42
LABELS = ["Unsuitable", "Moderately suitable", "Suitable"]


def design_matrix(df, cols):
    return pd.concat([df[cols], pd.get_dummies(df["crop"], prefix="crop").astype(float)], axis=1)


def scaled(est):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), est)


def cls_metrics(y, p):
    kw = dict(average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(y, p), "precision": precision_score(y, p, **kw),
            "recall": recall_score(y, p, **kw), "f1": f1_score(y, p, **kw)}


def reg_metrics(y, p):
    return {"r2": r2_score(y, p), "mae": mean_absolute_error(y, p), "rmse": mean_squared_error(y, p) ** 0.5}


def split(df):
    tr, te = next(GroupShuffleSplit(test_size=0.3, random_state=SEED).split(df, groups=df["year"]))
    return df.iloc[tr], df.iloc[te]   # hold out whole sowing years


def run(df):
    MODELS.mkdir(exist_ok=True); OUTPUTS.mkdir(exist_ok=True)
    all_cols = CLIMATE_COLS + LAND_COLS + WATER_COLS + NDVI_COLS
    tr, te = split(df)
    Xtr, Xte = design_matrix(tr, all_cols), design_matrix(te, all_cols)
    report = {"n_train": len(tr), "n_test": len(te), "test_years": sorted(te.year.unique().tolist()),
              "classification": {}, "regression": {}, "ablation_rf_regressor": {}}

    classifiers = {
        "random_forest": RandomForestClassifier(300, random_state=SEED, n_jobs=-1, class_weight="balanced"),
        "svm": scaled(SVC(C=3, class_weight="balanced", random_state=SEED)),
        "dnn_2_hidden": scaled(MLPClassifier((64, 32), max_iter=600, early_stopping=True, random_state=SEED)),
    }
    for name, m in classifiers.items():
        m.fit(Xtr, tr["label"].map(LABELS.index))   # integer codes (MLP early stopping needs numeric y)
        report["classification"][name] = cls_metrics(te["label"].map(LABELS.index), m.predict(Xte))
        joblib.dump({"model": m, "columns": list(Xtr.columns)}, MODELS / f"clf_{name}.joblib")

    regressors = {
        "lasso": scaled(LassoCV(cv=5, random_state=SEED, max_iter=20000)),
        "xgboost": XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9, random_state=SEED),
        "random_forest": RandomForestRegressor(300, random_state=SEED, n_jobs=-1),
    }
    for name, m in regressors.items():
        m.fit(Xtr, tr["score"])
        pred = np.clip(m.predict(Xte), 0, 100)
        report["regression"][name] = reg_metrics(te["score"], pred)
        joblib.dump({"model": m, "columns": list(Xtr.columns)}, MODELS / f"reg_{name}.joblib")

    # single-source vs multi-modal (climate only / + land / + water / + NDVI)
    groups = {"climate": CLIMATE_COLS, "climate+land": CLIMATE_COLS + LAND_COLS,
              "climate+land+water": CLIMATE_COLS + LAND_COLS + WATER_COLS, "all (multi-modal)": all_cols}
    for g, cols in groups.items():
        m = RandomForestRegressor(200, random_state=SEED, n_jobs=-1).fit(design_matrix(tr, cols), tr["score"])
        report["ablation_rf_regressor"][g] = reg_metrics(te["score"], np.clip(m.predict(design_matrix(te, cols)), 0, 100))

    rf = joblib.load(MODELS / "reg_random_forest.joblib")["model"]
    imp = pd.Series(rf.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
    report["rf_feature_importance_top10"] = imp.head(10).round(4).to_dict()
    lasso = joblib.load(MODELS / "reg_lasso.joblib")["model"]
    report["lasso_nonzero_features"] = int((lasso[-1].coef_ != 0).sum())
    (OUTPUTS / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    r = run(pd.read_csv(ROOT / "data" / "processed" / "land_profiles.csv"))
    print(json.dumps(r, indent=2))
