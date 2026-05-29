# 운동 퍼포먼스 대시보드

자유 형식의 운동일지 txt 파일에서 벤치프레스, 데드리프트, 스쿼트 기록만 추출하고, Streamlit 대시보드에서 기록 추세, 월별 성장률, 미래 예상 1RM을 확인하는 프로젝트입니다.

## 주요 기능

- 자유 형식 txt 운동일지 파싱
- 벤치프레스 / 데드리프트 / 스쿼트 세트 기록 추출
- 실제 중량과 예상 1RM 분리 저장
- 날짜 오기입 보정 추적
- 월별 성장률 확인
- Ridge Regression / RandomForestRegressor 기반 미래 예상 1RM 예측
- Streamlit 대시보드 시각화

## 프로젝트 구조

```text
workout_dashboard/
├── app.py
├── preprocessing.py
├── train_model.py
├── requirements.txt
├── README.md
└── data/
    ├── raw/
    │   └── workout_log.txt
    ├── processed/
    │   ├── workout_big3_sets.csv
    │   ├── workout_big3_daily.csv
    │   ├── workout_big3_monthly.csv
    │   ├── workout_big3_growth.csv
    │   └── workout_big3_audit.csv
    └── model/
        └── lift_growth_model.pkl
```

## 설치

```bash
pip install -r requirements.txt
```

`streamlit` 실행 파일이 PATH에 잡히지 않는 환경에서는 아래처럼 실행할 수 있습니다.

```bash
python -m streamlit run app.py
```

## 실행 순서

1. 원본 txt 전처리

```bash
python preprocessing.py
```

2. 예측 모델 학습

```bash
python train_model.py
```

3. 대시보드 실행

```bash
streamlit run app.py
```

또는:

```bash
python -m streamlit run app.py
```

## 전처리 기준

원본 파일은 아래 위치에 둡니다.

```text
data/raw/workout_log.txt
```

전처리 대상 운동은 3대 운동만 포함합니다.

- 벤치프레스: `벤치`, `벤치프레스`
- 데드리프트: `데드`, `데드리프트`, `컨벤 데드`, `컨벤데드`, `컨벤`
- 스쿼트: `스쿼트`

루마니안 데드리프트, RDL 등은 일반 데드리프트 기록에 포함하지 않습니다.

세트 기록은 아래 형태만 인식합니다.

```text
60 * 5
100 x 5
140 × 1
```

단독 숫자, 시간, 메모, 합산 기록은 세트 기록으로 처리하지 않습니다.

## 날짜 보정

원본 운동일지 일부에 연도 오기입이 있어 전처리 단계에서 보정합니다.

현재 보정 규칙:

```text
raw_date.year == 2022 and raw_date.month in [1, 2, 3, 4]
→ corrected_date.year = 2023
```

원본 파일은 수정하지 않고, CSV에 아래 컬럼을 남겨 추적합니다.

- `raw_date`
- `corrected_date`
- `date`
- `year_correction_applied`
- `line_number`
- `raw`

## 생성 CSV

### `workout_big3_sets.csv`

세트 단위 데이터입니다.

주요 컬럼:

- `line_number`
- `raw_date`
- `corrected_date`
- `date`
- `month`
- `lift`
- `lift_name`
- `weight`
- `reps`
- `sets`
- `failed`
- `estimated_1rm`
- `raw`
- `year_correction_applied`

### `workout_big3_daily.csv`

날짜별 요약 데이터입니다.

- `top_weight`: 실제로 든 최고 중량
- `best_estimated_1rm`: 반복 횟수 기반 최고 예상 1RM
- `total_volume`
- `total_sets`

### `workout_big3_monthly.csv`

월별 요약 데이터입니다.

- `monthly_top_weight`: 해당 월 실제 최고 중량
- `monthly_best_1rm`: 해당 월 최고 예상 1RM

### `workout_big3_growth.csv`

월별 성장률 데이터입니다.

- `monthly_best_1rm`
- `previous_best_1rm`
- `change_kg`
- `growth_rate_pct`
- `growth_direction`

### `workout_big3_audit.csv`

전처리 결과를 원본 txt 라인으로 추적하기 위한 검증용 파일입니다.

## 예상 1RM 계산

Epley 공식을 사용합니다.

```text
estimated_1rm = weight * (1 + reps / 30)
```

예:

```text
150 * 3
```

이면:

```text
weight = 150
reps = 3
estimated_1rm = 165
```

즉, `estimated_1rm`은 실제로 든 중량이 아니라 추정값입니다.

## 예측 모델

`train_model.py`는 월별 최고 예상 1RM 데이터를 사용해 3개월, 6개월, 12개월 후 예상 1RM을 예측하는 회귀 모델을 학습합니다.

사용 모델:

- `Ridge(alpha=1.0)`
- `RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=2, random_state=42)`

검증 방식:

- `TimeSeriesSplit`
- `MAE`

모델은 horizon별로 평가한 뒤 MAE와 안정성을 기준으로 선택됩니다.

저장 위치:

```text
data/model/lift_growth_model.pkl
```

## 대시보드 구성

Streamlit 앱은 다음 탭으로 구성됩니다.

- 기본 대시보드
- 월별 성장률
- 미래 예측

미래 예측 탭에서는 학습된 모델을 불러와 다음 값을 표시합니다.

- 3개월 후 예상 1RM
- 6개월 후 예상 1RM
- 12개월 후 예상 1RM
- horizon별 사용 모델
- horizon별 검증 MAE
- 모델 원본 예측값
- 현실 상한선 적용 후 예측값

## 주의사항

- 원본 txt 파일은 전처리 과정에서 수정하지 않습니다.
- 기존 processed CSV는 신뢰하지 않고 `python preprocessing.py` 실행 시 raw txt에서 다시 생성합니다.
- 실패 기록은 세트 CSV에는 남기지만 daily/monthly/growth 계산에서는 제외합니다.
- 예측값은 실제 기록이 아니라 모델 기반 추정값입니다.
