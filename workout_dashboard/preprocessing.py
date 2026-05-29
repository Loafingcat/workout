from pathlib import Path
import re

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data" / "raw" / "workout_log.txt"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SETS_PATH = PROCESSED_DIR / "workout_big3_sets.csv"
DAILY_PATH = PROCESSED_DIR / "workout_big3_daily.csv"
MONTHLY_PATH = PROCESSED_DIR / "workout_big3_monthly.csv"
GROWTH_PATH = PROCESSED_DIR / "workout_big3_growth.csv"
AUDIT_PATH = PROCESSED_DIR / "workout_big3_audit.csv"

LIFT_NAMES = {
    "bench": "벤치프레스",
    "deadlift": "데드리프트",
    "squat": "스쿼트",
}

FAILED_KEYWORDS = [
    "실패",
    "실 패",
    "fail",
    "반성공",
    "하프",
    "깔짝",
    "깊이 부족",
]

OTHER_EXERCISE_KEYWORDS = [
    "풀업",
    "딥스",
    "ohp",
    "오버헤드프레스",
    "인클라인",
    "덤벨",
    "아놀드",
    "플라이",
    "바벨로우",
    "케이블",
    "이두",
    "삼두",
    "햄스트링",
    "레그",
    "컬",
    "머신",
    "팩덱",
    "푸쉬업",
    "루마니안",
    "루마니안데드",
    "루마니안데드리프트",
    "rdl",
]


def normalize_text(line: str) -> str:
    return re.sub(r"\s+", "", line.strip().lower())


def correct_year(raw_date: pd.Timestamp) -> tuple[pd.Timestamp, bool]:
    if raw_date.year == 2022 and raw_date.month in [1, 2, 3, 4]:
        return raw_date.replace(year=2023), True
    return raw_date, False


def detect_lift(line: str) -> str | None:
    normalized = normalize_text(line)

    if any(keyword in normalized for keyword in ["루마니안", "rdl"]):
        return None
    if normalized in {"벤치", "벤치프레스"}:
        return "bench"
    if normalized in {"데드", "데드리프트", "컨벤데드", "컨벤"}:
        return "deadlift"
    if normalized == "스쿼트":
        return "squat"
    return None


def is_other_exercise(line: str) -> bool:
    normalized = normalize_text(line)
    if not normalized:
        return False
    return any(normalize_text(keyword) in normalized for keyword in OTHER_EXERCISE_KEYWORDS)


def is_text_line(line: str) -> bool:
    return bool(re.search(r"[가-힣A-Za-z]", line))


def is_failed(line: str) -> bool:
    text = line.lower()
    return any(keyword.lower() in text for keyword in FAILED_KEYWORDS)


def parse_workout_log(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"원본 파일이 없습니다: {raw_path}\n"
            "data/raw/workout_log.txt 파일을 추가한 뒤 다시 실행하세요."
        )

    rows = []
    date_headers = []
    current_raw_date = None
    current_corrected_date = None
    current_year_corrected = False
    current_lift = None

    # Date header: only a date inside [] is accepted, allowing spaces and one-digit month/day.
    date_pattern = re.compile(
        r"^\[(?P<year>\d{4})\s*-\s*(?P<month>\d{1,2})\s*-\s*(?P<day>\d{1,2})(?P<rest>[^\]]*)\]"
    )
    # Set line: weight * reps / x / X / × reps. Standalone numbers are ignored.
    set_pattern = re.compile(r"(?P<weight>\d+(?:\.\d+)?)\s*(?:\*|x|X|×)\s*(?P<reps>\d+)")

    with raw_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            date_match = date_pattern.match(line)
            if date_match:
                raw_date = pd.Timestamp(
                    year=int(date_match.group("year")),
                    month=int(date_match.group("month")),
                    day=int(date_match.group("day")),
                )
                corrected_date, year_corrected = correct_year(raw_date)
                current_raw_date = raw_date.date().isoformat()
                current_corrected_date = corrected_date.date().isoformat()
                current_year_corrected = year_corrected
                current_lift = None

                if raw_date.year == 2022 and raw_date.month in [1, 2, 3, 4]:
                    date_headers.append(
                        {
                            "line_number": line_number,
                            "raw_date": current_raw_date,
                            "candidate_corrected_date": current_corrected_date,
                            "raw_header": line,
                        }
                    )
                continue

            lift = detect_lift(line)
            if lift:
                current_lift = lift
                continue

            set_match = set_pattern.search(line)

            if is_other_exercise(line):
                current_lift = None
                continue

            if set_match is None and is_text_line(line):
                current_lift = None
                continue

            if current_raw_date is None or current_lift is None or set_match is None:
                continue

            weight = float(set_match.group("weight"))
            reps = int(set_match.group("reps"))
            sets = 1
            failed = is_failed(line)
            estimated_1rm = weight * (1 + reps / 30)

            rows.append(
                {
                    "line_number": line_number,
                    "raw_date": current_raw_date,
                    "corrected_date": current_corrected_date,
                    "date": current_corrected_date,
                    "month": current_corrected_date[:7],
                    "lift": current_lift,
                    "lift_name": LIFT_NAMES[current_lift],
                    "weight": weight,
                    "reps": reps,
                    "sets": sets,
                    "failed": failed,
                    "estimated_1rm": estimated_1rm,
                    "raw": line,
                    "year_correction_applied": current_year_corrected,
                }
            )

    set_columns = [
        "line_number",
        "raw_date",
        "corrected_date",
        "date",
        "month",
        "lift",
        "lift_name",
        "weight",
        "reps",
        "sets",
        "failed",
        "estimated_1rm",
        "raw",
        "year_correction_applied",
    ]
    candidate_columns = ["line_number", "raw_date", "candidate_corrected_date", "raw_header"]
    return pd.DataFrame(rows, columns=set_columns), pd.DataFrame(date_headers, columns=candidate_columns)


def build_daily_summary(sets_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "lift",
        "lift_name",
        "top_weight",
        "best_estimated_1rm",
        "total_volume",
        "total_sets",
    ]
    if sets_df.empty:
        return pd.DataFrame(columns=columns)

    successful = sets_df[~sets_df["failed"]].copy()
    if successful.empty:
        return pd.DataFrame(columns=columns)

    successful["volume"] = successful["weight"] * successful["reps"] * successful["sets"]
    daily = (
        successful.groupby(["date", "lift", "lift_name"], as_index=False)
        .agg(
            top_weight=("weight", "max"),
            best_estimated_1rm=("estimated_1rm", "max"),
            total_volume=("volume", "sum"),
            total_sets=("sets", "sum"),
        )
        .sort_values(["date", "lift"])
    )
    return daily[columns]


def build_monthly_summary(sets_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "month",
        "month_date",
        "lift",
        "lift_name",
        "monthly_top_weight",
        "monthly_best_1rm",
    ]
    if sets_df.empty:
        return pd.DataFrame(columns=columns)

    successful = sets_df[~sets_df["failed"]].copy()
    if successful.empty:
        return pd.DataFrame(columns=columns)

    monthly = (
        successful.groupby(["month", "lift", "lift_name"], as_index=False)
        .agg(
            monthly_top_weight=("weight", "max"),
            monthly_best_1rm=("estimated_1rm", "max"),
        )
        .sort_values(["month", "lift"])
    )
    monthly["month_date"] = monthly["month"] + "-01"
    return monthly[columns]


def build_growth_summary(monthly_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "month",
        "month_date",
        "lift",
        "lift_name",
        "monthly_best_1rm",
        "previous_best_1rm",
        "change_kg",
        "growth_rate_pct",
        "growth_direction",
    ]
    if monthly_df.empty:
        return pd.DataFrame(columns=columns)

    growth = monthly_df.copy().sort_values(["lift", "month"])
    growth["previous_best_1rm"] = growth.groupby("lift")["monthly_best_1rm"].shift(1)
    growth["change_kg"] = growth["monthly_best_1rm"] - growth["previous_best_1rm"]
    growth["growth_rate_pct"] = growth["change_kg"] / growth["previous_best_1rm"] * 100
    growth["growth_direction"] = "변화 없음"
    growth.loc[growth["change_kg"] > 0, "growth_direction"] = "양수"
    growth.loc[growth["change_kg"] < 0, "growth_direction"] = "음수"
    return growth[columns]


def build_audit(sets_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "line_number",
        "raw_date",
        "corrected_date",
        "date",
        "year_correction_applied",
        "lift",
        "lift_name",
        "weight",
        "reps",
        "failed",
        "estimated_1rm",
        "raw",
    ]
    return sets_df[columns].copy()


def print_validation(sets_df: pd.DataFrame, date_candidates: pd.DataFrame) -> None:
    audit_columns = ["line_number", "date", "lift", "weight", "reps", "failed", "estimated_1rm", "raw"]
    corrected_columns = [
        "line_number",
        "raw_date",
        "corrected_date",
        "date",
        "lift",
        "weight",
        "reps",
        "failed",
        "estimated_1rm",
        "raw",
    ]

    print("\n[연도 보정 후보]")
    print("없음" if date_candidates.empty else date_candidates.to_string(index=False))

    corrected_rows = sets_df[sets_df["year_correction_applied"]].copy()
    print(f"\n[검증] 연도 보정 적용 행 수: {len(corrected_rows)}")
    print("\n[검증] 연도 보정 적용 날짜")
    if corrected_rows.empty:
        print("없음")
    else:
        pairs = corrected_rows[["raw_date", "corrected_date"]].drop_duplicates().sort_values(["raw_date"])
        for _, row in pairs.iterrows():
            print(f"{row['raw_date']} -> {row['corrected_date']}")

    target_2022 = sets_df[sets_df["date"] == "2022-01-19"]
    print(f"\n[검증] 최종 date 기준 2022-01-19 행 수: {len(target_2022)}")
    if not target_2022.empty:
        print(target_2022[corrected_columns].to_string(index=False))

    target_2023 = sets_df[sets_df["date"] == "2023-01-19"]
    print("\n[검증] 2023-01-19 추출 행")
    print("없음" if target_2023.empty else target_2023[corrected_columns].to_string(index=False))

    print("\n[검증] 2022-01-19 추출 행")
    print("없음" if target_2022.empty else target_2022[audit_columns].to_string(index=False))

    successful = sets_df[~sets_df["failed"]].copy()
    print("\n[검증] 실제 최고 중량")
    for lift in ["bench", "deadlift", "squat"]:
        value = successful.loc[successful["lift"] == lift, "weight"].max()
        print(f"{lift}: {'없음' if pd.isna(value) else f'{value:g}kg'}")

    print("\n[검증] 종목별 최고 중량 raw 추적")
    for lift in ["bench", "deadlift", "squat"]:
        lift_rows = successful[successful["lift"] == lift]
        print(f"{lift}:")
        if lift_rows.empty:
            print("  없음")
            continue
        row = lift_rows.sort_values(["weight", "date"], ascending=[False, True]).iloc[0]
        print(f"  line_number: {row['line_number']}")
        print(f"  date: {row['date']}")
        print(f"  weight: {row['weight']:g}")
        print(f"  reps: {row['reps']}")
        print(f"  raw: {row['raw']}")

    suspicious = sets_df[
        ((sets_df["lift"] == "bench") & (sets_df["weight"] > 150))
        | ((sets_df["lift"] == "deadlift") & (sets_df["weight"] > 230))
        | ((sets_df["lift"] == "squat") & (sets_df["weight"] > 180))
    ]
    print("\n[검증] 이상치 의심 행")
    print("없음" if suspicious.empty else suspicious[audit_columns].to_string(index=False))

    standalone_pollution = sets_df[sets_df["weight"] >= 300]
    print("\n[검증] 단독 숫자 오염 의심 weight >= 300")
    print("없음" if standalone_pollution.empty else standalone_pollution[audit_columns].to_string(index=False))


def save_outputs(
    sets_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    growth_df: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sets_df.to_csv(SETS_PATH, index=False, encoding="utf-8-sig")
    daily_df.to_csv(DAILY_PATH, index=False, encoding="utf-8-sig")
    monthly_df.to_csv(MONTHLY_PATH, index=False, encoding="utf-8-sig")
    growth_df.to_csv(GROWTH_PATH, index=False, encoding="utf-8-sig")
    audit_df.to_csv(AUDIT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved: {SETS_PATH.relative_to(BASE_DIR)}")
    print(f"Saved: {DAILY_PATH.relative_to(BASE_DIR)}")
    print(f"Saved: {MONTHLY_PATH.relative_to(BASE_DIR)}")
    print(f"Saved: {GROWTH_PATH.relative_to(BASE_DIR)}")
    print(f"Saved: {AUDIT_PATH.relative_to(BASE_DIR)}")


def main() -> None:
    print("Rebuilding processed files from raw txt...")
    try:
        sets_df, date_candidates = parse_workout_log(RAW_PATH)
    except FileNotFoundError as error:
        print(error)
        raise SystemExit(1)

    daily_df = build_daily_summary(sets_df)
    monthly_df = build_monthly_summary(sets_df)
    growth_df = build_growth_summary(monthly_df)
    audit_df = build_audit(sets_df)
    save_outputs(sets_df, daily_df, monthly_df, growth_df, audit_df)
    print_validation(sets_df, date_candidates)

    if sets_df.empty:
        print("주의: 추출된 Big3 세트 데이터가 없습니다.")


if __name__ == "__main__":
    main()
