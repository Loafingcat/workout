from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
SETS_PATH = BASE_DIR / "data" / "processed" / "workout_big3_sets.csv"
DAILY_PATH = BASE_DIR / "data" / "processed" / "workout_big3_daily.csv"
MONTHLY_PATH = BASE_DIR / "data" / "processed" / "workout_big3_monthly.csv"
MODEL_PATH = BASE_DIR / "data" / "model" / "lift_growth_model.pkl"

LIFT_ORDER = ["벤치프레스", "데드리프트", "스쿼트"]
LIFT_KEY_BY_NAME = {
    "벤치프레스": "bench",
    "데드리프트": "deadlift",
    "스쿼트": "squat",
}
DEFAULT_CAPS = {
    "벤치프레스": 180.0,
    "데드리프트": 260.0,
    "스쿼트": 220.0,
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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    paths = [SETS_PATH, DAILY_PATH, MONTHLY_PATH]
    if not all(path.exists() for path in paths):
        return None

    sets_df = pd.read_csv(SETS_PATH)
    daily_df = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    monthly_df = pd.read_csv(MONTHLY_PATH)
    return sets_df, daily_df, monthly_df


def calculate_big_three_total(daily_df: pd.DataFrame) -> float | None:
    if daily_df.empty:
        return None

    best_by_lift = daily_df.groupby("lift")["best_estimated_1rm"].max()
    required_lifts = {"bench", "deadlift", "squat"}
    if not required_lifts.issubset(set(best_by_lift.index)):
        return None
    return float(best_by_lift.loc[list(required_lifts)].sum())


def build_big_three_trend(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame(columns=["date", "big_three_total"])

    pivot = (
        daily_df.pivot_table(
            index="date",
            columns="lift",
            values="best_estimated_1rm",
            aggfunc="max",
        )
        .sort_index()
        .cummax()
        .ffill()
    )

    required = ["bench", "deadlift", "squat"]
    if not all(lift in pivot.columns for lift in required):
        return pd.DataFrame(columns=["date", "big_three_total"])

    result = pivot[required].dropna().copy()
    result["big_three_total"] = result[required].sum(axis=1)
    return result.reset_index()[["date", "big_three_total"]]


def build_monthly_growth(monthly_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "month",
        "month_date",
        "lift",
        "lift_name",
        "monthly_best_1rm",
        "prev_month_1rm",
        "growth_kg",
        "growth_pct",
        "growth_direction",
    ]
    if monthly_df.empty:
        return pd.DataFrame(columns=columns)

    growth_df = monthly_df.copy()
    if "month_date" not in growth_df.columns:
        growth_df["month_date"] = growth_df["month"] + "-01"
    growth_df["month_date"] = pd.to_datetime(growth_df["month_date"], errors="coerce")
    growth_df = growth_df.dropna(subset=["month_date"]).sort_values(["lift", "month_date"])
    growth_df["month"] = growth_df["month_date"].dt.strftime("%Y-%m")
    growth_df["prev_month_1rm"] = growth_df.groupby("lift")["monthly_best_1rm"].shift(1)
    growth_df["growth_kg"] = growth_df["monthly_best_1rm"] - growth_df["prev_month_1rm"]
    growth_df["growth_pct"] = growth_df["growth_kg"] / growth_df["prev_month_1rm"] * 100
    growth_df["growth_direction"] = np.select(
        [growth_df["growth_kg"] > 0, growth_df["growth_kg"] < 0],
        ["양수", "음수"],
        default="변화 없음",
    )
    return growth_df[columns]


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "-"
    if value > 0:
        return f"+{value:.2f}%"
    if value < 0:
        return f"{value:.2f}%"
    return "0.00%"


def format_kg(value: float) -> str:
    if pd.isna(value):
        return "-"
    if value > 0:
        return f"+{value:.1f}kg"
    if value < 0:
        return f"{value:.1f}kg"
    return "0.0kg"


def color_growth(value: float) -> str:
    if pd.isna(value):
        return "color: #6b7280;"
    if value > 0:
        return "color: #16a34a; font-weight: 700;"
    if value < 0:
        return "color: #dc2626; font-weight: 700;"
    return "color: #6b7280;"


def growth_symbol(value: float) -> str:
    if pd.isna(value):
        return "데이터 부족"
    if value > 0:
        return "▲"
    if value < 0:
        return "▼"
    return "—"


def growth_color(value: float) -> str:
    if pd.isna(value):
        return "#6b7280"
    if value > 0:
        return "#16a34a"
    if value < 0:
        return "#dc2626"
    return "#6b7280"


@st.cache_resource
def load_model_bundle() -> dict[str, object] | None:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def build_latest_feature_row(monthly_df: pd.DataFrame, lift_name: str, model_bundle: dict[str, object]) -> pd.DataFrame:
    lift_key = LIFT_KEY_BY_NAME[lift_name]
    actual = monthly_df[monthly_df["lift"] == lift_key].copy()
    if actual.empty or len(actual) < 6:
        return pd.DataFrame()

    if "month_date" not in actual.columns:
        actual["month_date"] = actual["month"] + "-01"
    actual["month_date"] = pd.to_datetime(actual["month_date"], errors="coerce")
    actual = actual.dropna(subset=["month_date"]).sort_values("month_date")
    actual["current_1rm"] = actual["monthly_best_1rm"]
    actual["prev_1rm"] = actual["monthly_best_1rm"].shift(1)
    actual["growth_kg"] = actual["current_1rm"] - actual["prev_1rm"]
    actual["growth_pct"] = actual["growth_kg"] / actual["prev_1rm"] * 100
    actual["rolling_3m_mean"] = actual["monthly_best_1rm"].rolling(3).mean()
    actual["rolling_3m_std"] = actual["monthly_best_1rm"].rolling(3).std()
    actual["rolling_3m_growth"] = actual["current_1rm"] - actual["monthly_best_1rm"].shift(3)
    actual["rolling_6m_mean"] = actual["monthly_best_1rm"].rolling(6).mean()
    actual["month_index"] = range(len(actual))
    actual["lift_encoded"] = model_bundle["lift_mapping"][lift_key]
    return actual.dropna(subset=FEATURE_COLUMNS).tail(1)[FEATURE_COLUMNS]


def apply_prediction_bounds(raw_prediction: float, current_1rm: float, cap: float) -> float:
    lower_bound = current_1rm * 0.9
    return min(max(raw_prediction, lower_bound), cap)


def build_forecast(
    monthly_df: pd.DataFrame,
    lift_name: str,
    cap: float,
    model_bundle: dict[str, object],
) -> tuple[pd.DataFrame, dict[int, dict[str, float]]]:
    lift_key = LIFT_KEY_BY_NAME[lift_name]
    actual = monthly_df[monthly_df["lift"] == lift_key].copy()
    if actual.empty:
        return pd.DataFrame(columns=["month", "estimated_1rm", "type"]), {}

    if "month_date" not in actual.columns:
        actual["month_date"] = actual["month"] + "-01"
    actual["month_date"] = pd.to_datetime(actual["month_date"], errors="coerce")
    actual = actual.dropna(subset=["month_date"]).sort_values("month_date")
    actual["estimated_1rm"] = actual["monthly_best_1rm"]

    feature_row = build_latest_feature_row(monthly_df, lift_name, model_bundle)
    if feature_row.empty:
        return pd.DataFrame(columns=["month", "estimated_1rm", "type"]), {}

    actual_chart = actual[["month_date", "estimated_1rm"]].copy()
    actual_chart["type"] = "실제 월별 최고 예상 1RM"
    actual_chart["raw_prediction"] = np.nan
    actual_chart["adjusted_prediction"] = np.nan

    current_1rm = float(actual_chart["estimated_1rm"].iloc[-1])
    last_month = actual_chart["month_date"].iloc[-1]
    predictions = {}
    forecast_rows = []
    for months_ahead, model in model_bundle["models"].items():
        raw_prediction = float(model.predict(feature_row)[0])
        adjusted_prediction = apply_prediction_bounds(raw_prediction, current_1rm, cap)
        predictions[int(months_ahead)] = {
            "raw": raw_prediction,
            "adjusted": adjusted_prediction,
        }
        forecast_rows.append(
            {
                "month_date": last_month + pd.DateOffset(months=months_ahead),
                "estimated_1rm": adjusted_prediction,
                "type": f"{months_ahead}개월 후 예측",
                "raw_prediction": raw_prediction,
                "adjusted_prediction": adjusted_prediction,
            }
        )

    chart_df = pd.concat([actual_chart, pd.DataFrame(forecast_rows)], ignore_index=True)
    chart_df["month"] = chart_df["month_date"].dt.strftime("%Y-%m")
    return chart_df[["month", "estimated_1rm", "type", "raw_prediction", "adjusted_prediction"]], predictions


def show_metrics(daily_df: pd.DataFrame) -> None:
    total_days = daily_df["date"].nunique() if not daily_df.empty else 0
    total_sets = int(daily_df["total_sets"].sum()) if not daily_df.empty else 0
    total_volume = float(daily_df["total_volume"].sum()) if not daily_df.empty else 0
    big_three_total = calculate_big_three_total(daily_df)

    cols = st.columns(4)
    cols[0].metric("총 운동일 수", f"{total_days:,}")
    cols[1].metric("총 세트 수", f"{total_sets:,}")
    cols[2].metric("총 볼륨", f"{total_volume:,.0f}")
    cols[3].metric(
        "현재 예상 3대 합산",
        "-" if big_three_total is None else f"{big_three_total:,.1f}",
    )


def filter_by_lift(
    sets_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    selected_lifts: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        sets_df[sets_df["lift_name"].isin(selected_lifts)].copy(),
        daily_df[daily_df["lift_name"].isin(selected_lifts)].copy(),
        monthly_df[monthly_df["lift_name"].isin(selected_lifts)].copy(),
    )


def show_main_dashboard(sets_df: pd.DataFrame, daily_df: pd.DataFrame, monthly_df: pd.DataFrame) -> None:
    if sets_df.empty or daily_df.empty:
        st.warning("표시할 운동 데이터가 없습니다.")

    selected_lifts = st.sidebar.multiselect("운동 선택", LIFT_ORDER, default=LIFT_ORDER)
    sets_filtered, daily_filtered, monthly_filtered = filter_by_lift(
        sets_df, daily_df, monthly_df, selected_lifts
    )

    with st.expander("요약 지표 보기", expanded=False):
        show_metrics(daily_filtered)

    big_three_df = build_big_three_trend(daily_df)
    show_charts(daily_filtered, monthly_filtered, big_three_df)

    with st.expander("세트 단위 데이터 보기"):
        st.dataframe(sets_filtered, use_container_width=True)

    with st.expander("날짜별 요약 데이터 보기"):
        st.dataframe(daily_filtered, use_container_width=True)


def show_charts(daily_df: pd.DataFrame, monthly_df: pd.DataFrame, big_three_df: pd.DataFrame) -> None:
    if daily_df.empty:
        st.info("표시할 날짜별 요약 데이터가 없습니다.")
        return

    st.plotly_chart(
        px.line(daily_df, x="date", y="top_weight", color="lift_name", title="날짜별 실제 최고 중량 변화"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.line(
            daily_df,
            x="date",
            y="best_estimated_1rm",
            color="lift_name",
            title="날짜별 예상 1RM 변화",
        ),
        use_container_width=True,
    )

    if not monthly_df.empty:
        st.plotly_chart(
            px.line(
                monthly_df,
                x="month",
                y="monthly_best_1rm",
                color="lift_name",
                title="월별 최고 예상 1RM",
            ),
            use_container_width=True,
        )

    st.plotly_chart(
        px.line(daily_df, x="date", y="total_volume", color="lift_name", title="날짜별 운동 볼륨 변화"),
        use_container_width=True,
    )

    if big_three_df.empty:
        st.info("예상 3대 합산 변화는 세 운동 기록이 모두 있을 때 표시됩니다.")
    else:
        st.plotly_chart(
            px.line(big_three_df, x="date", y="big_three_total", title="예상 3대 합산 변화"),
            use_container_width=True,
        )


def show_monthly_growth_tab(monthly_df: pd.DataFrame) -> None:
    growth_df = build_monthly_growth(monthly_df)
    if growth_df.empty:
        st.info("표시할 월별 성장률 데이터가 없습니다.")
        return

    st.subheader("월별 성장률")
    st.caption("성장률은 월별 최고 예상 1RM 기준으로 계산됩니다.")

    latest_by_lift = (
        growth_df.dropna(subset=["growth_pct"])
        .sort_values("month_date")
        .groupby("lift", as_index=False)
        .tail(1)
    )

    card_cols = st.columns(3)
    for index, lift_name in enumerate(LIFT_ORDER):
        lift_key = LIFT_KEY_BY_NAME[lift_name]
        rows = latest_by_lift[latest_by_lift["lift"] == lift_key]
        if rows.empty:
            card_cols[index].markdown(
                f"""
                <div style="border:1px solid #e5e7eb; border-radius:8px; padding:14px;">
                    <div style="font-weight:700;">{lift_name}</div>
                    <div style="color:#6b7280; margin-top:10px;">데이터 부족</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            continue

        row = rows.iloc[0]
        pct = row["growth_pct"]
        kg = row["growth_kg"]
        color = growth_color(pct)
        symbol = growth_symbol(pct)
        previous_1rm = row["prev_month_1rm"]
        current_1rm = row["monthly_best_1rm"]
        card_cols[index].markdown(
            f"""
            <div style="border:1px solid #e5e7eb; border-radius:8px; padding:14px;">
                <div style="font-weight:700;">{lift_name}</div>
                <div style="color:{color}; font-size:26px; font-weight:800; margin-top:8px;">
                    {symbol} {format_pct(pct)}
                </div>
                <div style="color:{color}; font-weight:700;">{format_kg(kg)}</div>
                <div style="color:#374151; margin-top:8px;">
                    {previous_1rm:.1f}kg → {current_1rm:.1f}kg
                </div>
                <div style="color:#6b7280; font-size:12px; margin-top:4px;">{row["month"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    pct_pivot = growth_df.pivot(index="month", columns="lift_name", values="growth_pct")
    kg_pivot = growth_df.pivot(index="month", columns="lift_name", values="growth_kg")
    pct_pivot = pct_pivot.reindex(columns=LIFT_ORDER).sort_index()
    kg_pivot = kg_pivot.reindex(columns=LIFT_ORDER).sort_index()

    st.markdown("#### 월별 전월 대비 변화율 %")
    st.dataframe(
        pct_pivot.style.format(format_pct).map(color_growth),
        use_container_width=True,
    )

    st.markdown("#### 월별 전월 대비 변화량 kg")
    st.dataframe(
        kg_pivot.style.format(format_kg).map(color_growth),
        use_container_width=True,
    )

    with st.expander("보조 그래프 보기"):
        st.plotly_chart(
            px.bar(
                growth_df.dropna(subset=["growth_pct"]),
                x="month",
                y="growth_pct",
                color="growth_direction",
                facet_row="lift_name",
                title="보조 그래프: 월별 전월 대비 변화율",
            ),
            use_container_width=True,
        )


def show_forecast_tab(monthly_df: pd.DataFrame) -> None:
    if monthly_df.empty:
        st.info("예측에 사용할 월별 데이터가 없습니다.")
        return

    st.caption("예측 기준: 지도학습 회귀 모델")
    model_bundle = load_model_bundle()
    if model_bundle is None:
        st.warning("학습된 모델 파일이 없습니다. 먼저 python train_model.py 를 실행하세요.")
        return

    selected_lift = st.selectbox("종목 선택", LIFT_ORDER)

    cap_cols = st.columns(3)
    caps = {}
    for index, lift_name in enumerate(LIFT_ORDER):
        caps[lift_name] = cap_cols[index].number_input(
            f"{lift_name} 예상 1RM 현실 상한선",
            min_value=0.0,
            value=DEFAULT_CAPS[lift_name],
            step=2.5,
        )

    forecast_df, predictions = build_forecast(monthly_df, selected_lift, caps[selected_lift], model_bundle)
    if forecast_df.empty:
        st.info("선택한 종목의 예측 데이터가 없습니다.")
        return

    selected_names = model_bundle.get("selected_model_names", {})
    metrics = model_bundle.get("metrics", {})
    for months_ahead in [3, 6, 12]:
        model_name = selected_names.get(months_ahead, "-")
        mae = metrics.get(months_ahead, {}).get(model_name, {}).get("mae")
        mae_text = "-" if mae is None or pd.isna(mae) else f"{mae:.1f}kg"
        st.caption(f"{months_ahead}개월 후 모델: {model_name} / 검증 MAE: {mae_text}")

    metric_cols = st.columns(3)
    for index, months_ahead in enumerate([3, 6, 12]):
        prediction = predictions.get(months_ahead)
        value = None if prediction is None else prediction["adjusted"]
        metric_cols[index].metric(
            f"{months_ahead}개월 후 예상 1RM",
            "-" if value is None else f"{value:,.1f} kg",
        )

    st.plotly_chart(
        px.line(
            forecast_df,
            x="month",
            y="estimated_1rm",
            color="type",
            markers=True,
            title=f"{selected_lift} 실제 월별 최고 예상 1RM과 모델 예측",
        ),
        use_container_width=True,
    )
    prediction_rows = forecast_df[forecast_df["raw_prediction"].notna()].copy()
    if not prediction_rows.empty:
        prediction_rows = prediction_rows.rename(
            columns={
                "raw_prediction": "모델 원본 예측값",
                "adjusted_prediction": "상한선 적용 후 예측값",
            }
        )
        st.dataframe(
            prediction_rows[["month", "type", "모델 원본 예측값", "상한선 적용 후 예측값"]],
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(page_title="운동 퍼포먼스 대시보드", layout="wide")
    st.title("운동 퍼포먼스 대시보드")

    loaded = load_data()
    if loaded is None:
        st.error("전처리 파일이 없습니다. 먼저 python preprocessing.py 를 실행하세요.")
        return

    sets_df, daily_df, monthly_df = loaded

    main_tab, growth_tab, forecast_tab = st.tabs(["기본 대시보드", "월별 성장률", "미래 예측"])
    with main_tab:
        show_main_dashboard(sets_df, daily_df, monthly_df)
    with growth_tab:
        show_monthly_growth_tab(monthly_df)
    with forecast_tab:
        show_forecast_tab(monthly_df)


if __name__ == "__main__":
    main()
