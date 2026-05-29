from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


BASE_DIR = Path(__file__).resolve().parent
MONTHLY_PATH = BASE_DIR / "data" / "processed" / "workout_big3_monthly.csv"
MODEL_DIR = BASE_DIR / "data" / "model"
MODEL_PATH = MODEL_DIR / "lift_growth_model.pkl"

LIFT_MAPPING = {
    "bench": 0,
    "deadlift": 1,
    "squat": 2,
}
FEATURE_COLUMNS = [
    "current_1rm",
    "prev_1rm",
    "growth_kg",
    "growth_pct",
    "rolling_3m_mean",
    "rolling_3m_std",
    "rolling_3m_growth",
    "rolling_6m_mean",
    "month_index",
    "lift_encoded",
]
HORIZONS = [3, 6, 12]


def load_monthly_data() -> pd.DataFrame:
    if not MONTHLY_PATH.exists():
        raise FileNotFoundError(
            f"월별 전처리 파일이 없습니다: {MONTHLY_PATH}\n"
            "먼저 python preprocessing.py 를 실행하세요."
        )

    df = pd.read_csv(MONTHLY_PATH)
    required = {"month", "lift", "lift_name", "monthly_top_weight", "monthly_best_1rm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"workout_big3_monthly.csv 필수 컬럼 누락: {sorted(missing)}")

    df = df.copy()
    if "month_date" not in df.columns:
        df["month_date"] = df["month"] + "-01"
    df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
    df = df.dropna(subset=["month_date", "monthly_best_1rm"])
    df = df[df["lift"].isin(LIFT_MAPPING)].sort_values(["lift", "month_date"])
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    grouped = featured.groupby("lift", group_keys=False)

    featured["current_1rm"] = featured["monthly_best_1rm"]
    featured["prev_1rm"] = grouped["monthly_best_1rm"].shift(1)
    featured["growth_kg"] = featured["current_1rm"] - featured["prev_1rm"]
    featured["growth_pct"] = featured["growth_kg"] / featured["prev_1rm"] * 100
    featured["rolling_3m_mean"] = grouped["monthly_best_1rm"].rolling(3).mean().reset_index(level=0, drop=True)
    featured["rolling_3m_std"] = grouped["monthly_best_1rm"].rolling(3).std().reset_index(level=0, drop=True)
    featured["rolling_3m_growth"] = featured["current_1rm"] - grouped["monthly_best_1rm"].shift(3)
    featured["rolling_6m_mean"] = grouped["monthly_best_1rm"].rolling(6).mean().reset_index(level=0, drop=True)
    featured["month_index"] = grouped.cumcount()
    featured["lift_encoded"] = featured["lift"].map(LIFT_MAPPING)

    for horizon in HORIZONS:
        featured[f"target_{horizon}m"] = grouped["monthly_best_1rm"].shift(-horizon)

    return featured


def make_models() -> dict[str, object]:
    return {
        "Ridge": Ridge(alpha=1.0),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=2,
            random_state=42,
        ),
    }


def evaluate_model(model, x: pd.DataFrame, y: pd.Series) -> float:
    n_splits = min(5, len(x) - 1)
    if n_splits < 2:
        return np.nan

    scores = []
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(x):
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(x.iloc[test_idx])
        scores.append(mean_absolute_error(y.iloc[test_idx], pred))
    return float(np.mean(scores))


def choose_model(metrics: dict[str, dict[str, float]], sample_count: int) -> str:
    ridge_mae = metrics["Ridge"]["mae"]
    rf_mae = metrics["RandomForestRegressor"]["mae"]
    if sample_count < 20 or np.isnan(rf_mae):
        return "Ridge"
    if np.isnan(ridge_mae):
        return "RandomForestRegressor"
    if rf_mae < ridge_mae * 0.9:
        return "RandomForestRegressor"
    return "Ridge"


def train_models(featured: pd.DataFrame) -> dict[str, object]:
    trained_models = {}
    selected_model_names = {}
    metrics = {}
    training_rows = {}

    for horizon in HORIZONS:
        target_col = f"target_{horizon}m"
        dataset = featured.dropna(subset=FEATURE_COLUMNS + [target_col]).copy()
        print(f"\n[horizon {horizon}개월] 학습 가능 행 수: {len(dataset)}")
        if len(dataset) < 8:
            print(f"경고: {horizon}개월 후 target 데이터가 너무 적어 학습을 건너뜁니다.")
            continue

        x = dataset[FEATURE_COLUMNS]
        y = dataset[target_col]
        metrics[horizon] = {}

        for model_name, model in make_models().items():
            mae = evaluate_model(model, x, y)
            metrics[horizon][model_name] = {"mae": mae}
            mae_text = "nan" if np.isnan(mae) else f"{mae:.3f}kg"
            print(f"{model_name} MAE: {mae_text}")

        selected_name = choose_model(metrics[horizon], len(dataset))
        final_model = make_models()[selected_name]
        final_model.fit(x, y)
        trained_models[horizon] = final_model
        selected_model_names[horizon] = selected_name
        training_rows[horizon] = len(dataset)
        print(f"선택 모델: {selected_name}")

    return {
        "models": trained_models,
        "selected_model_names": selected_model_names,
        "metrics": metrics,
        "feature_columns": FEATURE_COLUMNS,
        "lift_mapping": LIFT_MAPPING,
        "training_rows": training_rows,
    }


def main() -> None:
    monthly_df = load_monthly_data()
    featured = create_features(monthly_df)
    model_bundle = train_models(featured)

    if not model_bundle["models"]:
        raise SystemExit("학습된 모델이 없습니다. 월별 데이터가 더 필요합니다.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"\nSaved: {MODEL_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
