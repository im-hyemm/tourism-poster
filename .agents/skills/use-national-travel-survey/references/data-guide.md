# 2025년 국민여행조사 국내여행 데이터 가이드

## 기준 자료

- 원자료: `data/raw/national_travel_survey/y2025/2025년_국민여행조사_국내여행.SAV`
- 프로젝트 변수표:
  `data/raw/national_travel_survey/y2025/national_travel_survey_codebook_2025.csv`,
  `national_travel_survey_codebook_2025_full.csv`
- 공식 자료: `docs/report/2025년_국민여행조사_유저가이드.pdf`
- 공식 자료: `docs/report/2025년_국민여행조사_코드북.pdf`
- 공식 자료: `docs/report/2025년_국민여행조사_설문지.pdf`
- 보조 문서: `docs/note/tour_survey_2025_variable_code.md`
- 방문지역 보조 코드:
  `data/raw/national_travel_survey/national_travel_survey_region_code.csv`

충돌할 때는 SAV의 실제 메타데이터와 공식 2025년 유저가이드·코드북·설문지를
우선하고 프로젝트 메모는 보조 설명으로 사용한다.

## 파일과 조사 구조

- 현재 SAV는 52,185행, 2,419열이며 `ID`는 응답자 식별자다.
- 공식 유저가이드는 데이터를 응답자 Base의 가로형 자료로 설명하며, 한 행에
  응답자 1명의 결과가 있고 월별 자료가 누적되어 있다.
- 조사 대상은 만 15세 이상이며 1:1 가구방문 면접 방식이다.
- 공식 표본설계의 목표 표본은 월 4,300명, 연 51,600명이다. 실제 파일 행 수와
  목표 표본 수가 다르므로 둘을 같은 값으로 단정하지 않는다.
- 현재 파일의 국내여행 가중치는 `WT_DOM`이다. 공식 분석값은 가중치를 적용한다.
- 공식 유저가이드에 따르면 `WT_DOM`은 설계가중치, 무응답 조정, 2025년
  장래인구추계의 만 15세 이상 인구를 이용한 조정을 거쳐 산출되며 월별 모집단
  수를 반영한다.

## 가중치 적용

### 적용 원칙

- 가중치는 분석 변수에 무조건 한 번씩 곱해 새 변수를 만드는 값이 아니다.
  추정하려는 총계, 비율, 평균과 분석 단위에 맞춰 분자와 분모에 적용한다.

유효 분석대상 여부를 `I`, 분석값을 `y`, 가중치를 `w = WT_DOM`이라 할 때 다음
산식을 사용한다.

| 추정량       | 산식                                  | 해석                          |
| ------------ | ------------------------------------- | ----------------------------- |
| 모집단 총계  | `sum(w * I * y)`                      | 여행횟수, 여행일수, 지출액 등 |
| 가중 사례 수 | `sum(w * I)`                          | 해당 조건의 사람·여행·방문 수 |
| 가중 비율    | `sum(w * I * condition) / sum(w * I)` | 유효 분모 중 조건 충족 비율   |
| 가중 평균    | `sum(w * I * y) / sum(w * I)`         | 유효 관측치의 평균            |

- 변수별 결측을 분모에서 제외하고, 설문 분기로 인한 구조적 비해당은 분석대상
  정의에서 제외한다. 분자와 분모가 같은 유효 집단을 사용했는지 확인한다.
- 공식 모집단 총계를 낼 때 `WT_DOM`을 표본 수에 맞춰 정규화하거나 반올림하지
  않는다. 정규화한 가중치는 비율·평균에는 같은 결과를 줄 수 있지만 공식 총계는
  보존하지 못한다.
- 연간 누적 파일에서 월별 가중치의 단순 합을 연간 고유 인구로 해석하지 않는다.
  월별 여행행동을 합산한 연간 여행횟수·일수·지출 총계와, 사람-월 관측치의 가중
  비율·평균을 구분해 이름과 단위를 적는다.
- 월별 경험 여부의 연간 누적 가중 비율은 평균적인 월 경험률이지 연간 고유
  인구의 경험률이 아니다. 연간 경험률은 공식 연간 지표에 대응하는 변수가
  확인된 경우에만 산출한다.

### 분석 단위별 적용

- 응답자 수준의 경험률·인구통계 비율은 응답자 한 행당 `WT_DOM`을 한 번
  적용한다.
- 여행 슬롯을 long으로 변환한 뒤 `WT_DOM`을 각 여행에 반복하는 것은 가중
  여행횟수나 여행 기준 구성비를 계산할 때만 사용한다. `sum(WT_DOM)`의 결과는
  여행 건수이지 사람 수가 아니다.
- 여행자 기준 비율은 먼저 응답자별 조건 충족 여부를 하나로 축약한 뒤 응답자
  수준에서 가중한다. 여행별 비율과 여행자별 비율을 바꾸어 쓰지 않는다.
- 방문지 long 자료에서 가중치를 합산하면 방문 건수 기준 결과가 된다. 여행 수나
  방문자 수가 필요하면 각각 여행 키 또는 `ID`로 중복을 제거하거나 축약한 뒤
  가중한다.
- 응답자 행의 여행횟수 변수로 총량을 구한 `sum(WT_DOM * 여행횟수)`와 여행
  long 자료의 `sum(WT_DOM)`이 같은 대상·여행유형에서 일치하는지 대조한다.

### Python과 SAV 구현

- `src.path.NATIONAL_TRAVEL_SURVEY_RAW_DATA_DIR`에서 SAV 경로를 만들고
  `pyreadstat.read_sav()`로 읽는다.
  메모리를 줄이려면 `usecols`에 분석 변수, 키와 `WT_DOM`을 함께 지정한다.
- 코드북과 SAV 값 라벨을 대조할 수 있도록 원코드를 유지한다. 분석 중 라벨이
  필요하면 `meta.variable_value_labels`를 확인해 별도 표시 열을 만들고 원변수를
  덮어쓰지 않는다.
- 가중치만큼 행을 복제하지 않는다. `float64` 벡터 연산으로 가중 분자와 분모를
  따로 계산한다.

```python
import numpy as np
import pandas as pd
import pyreadstat

from src.path import NATIONAL_TRAVEL_SURVEY_RAW_DATA_DIR


SAV_PATH = (
    NATIONAL_TRAVEL_SURVEY_RAW_DATA_DIR
    / "y2025"
    / "2025년_국민여행조사_국내여행.SAV"
)
WEIGHT_COLUMN = "WT_DOM"

columns = ["ID", WEIGHT_COLUMN, "SA1_2", "SA1_3"]
data, metadata = pyreadstat.read_sav(
    SAV_PATH,
    usecols=columns,
    apply_value_formats=False,
)

weight = pd.to_numeric(data[WEIGHT_COLUMN], errors="coerce")
assert data["ID"].notna().all()
assert data["ID"].is_unique
assert weight.notna().all()
assert weight.gt(0).all()


def weighted_total(values: pd.Series, weights: pd.Series) -> float:
    """결측을 쌍별 제외하여 가중 총계를 계산한다.

    Args:
        values: 합산할 분석값.
        weights: 분석값과 같은 인덱스를 가진 가중치.

    Returns:
        유효 관측치의 가중 총계.
    """
    valid = values.notna() & weights.notna() & weights.gt(0)
    return float((values.loc[valid] * weights.loc[valid]).sum())


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """결측을 쌍별 제외하여 가중 평균 또는 비율을 계산한다.

    불리언 또는 0/1 값을 전달하면 가중 비율이 된다.

    Args:
        values: 평균을 계산할 수치, 불리언 또는 0/1 값.
        weights: 분석값과 같은 인덱스를 가진 가중치.

    Returns:
        유효 관측치의 가중 평균. 유효값이 없으면 ``np.nan``.
    """
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return float(np.nan)
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


trip_count = pd.to_numeric(data["SA1_2"], errors="coerce")
visitor = trip_count.gt(0).astype(float)
visitor.loc[trip_count.isna()] = np.nan

trip_total = weighted_total(trip_count, weight)
visitor_share = weighted_mean(visitor, weight)
```

- `.gt(0)`처럼 비교 연산으로 만든 불리언은 원변수가 결측이어도 `False`가 될 수
  있다. 결측이 있는 변수는 먼저 유효 마스크를 만들거나 결측을 유지하는 0/1 열을
  만든 뒤 가중 비율을 계산한다.
- 그룹별 결과도 각 그룹 안에서 같은 함수를 적용하고 비가중 N, 가중 분모
  `sum(WT_DOM)`, 결측 N을 함께 저장한다.
- 가중 중앙값·분위수는 `numpy.average()`로 계산할 수 없다. 값으로 정렬한 뒤
  누적 가중치가 목표 비율에 처음 도달하는 값을 사용하고, 사용한 정의를 기록한다.

### 검산

- 가중 전후 분포, `WT_DOM`의 결측·비양수 값, 극단값, 분자·분모와 단위를
  확인한다. 공식 보고서와 같은 정의의 지표가 있으면 허용 가능한 반올림 오차
  안에서 재현되는지 검산한다.

## 분석 단위

| 단위            | 기본 키                  | 관련 변수                                |
| --------------- | ------------------------ | ---------------------------------------- |
| 응답자·조사회차 | `ID`                     | `SA1_*`, `MON_EXP_*`, 인구통계, `WT_DOM` |
| 개별 여행       | `ID + TRIP_NO`           | `D_TRA1_*`부터 `D_TRA6_*`의 여행 슬롯    |
| 방문지          | `ID + TRIP_NO + SPOT_NO` | 각 여행 슬롯의 `*_SPOT`, 숙박지역·시설   |

- 시장 규모처럼 유형별 총 여행횟수를 추정할 때는 응답자 행의 `SA1_1`~`SA1_5`와
  `WT_DOM`을 사용한다.
- 여행 슬롯을 long 형태로 바꾸면 같은 응답자의 `WT_DOM`이 여행마다 반복된다.
  이를 응답자 모집단 총량처럼 단순 합산하지 않는다.
- 방문지 슬롯은 한 여행에 여러 개일 수 있다. 첫 방문지를 주목적지나 실제 이동
  순서라고 단정하지 않는다.

## 여행유형

`D_TRA*_CASE`의 값 라벨은 다음과 같다.

1. 국내 관광·휴양 여행
2. 관광·휴양 활동을 포함한 가족·친지·친구 방문
3. 관광·휴양 활동이 없는 단순 가족·친지·친구 방문
4. 관광·휴양 활동을 포함한 출장·업무 여행
5. 관광·휴양 활동이 없는 단순 출장·업무

프로젝트에서 연고 방문은 운영상 2와 3을 포함한다. 이는 출생지나 고향을 직접
측정한 값이 아니다.

## 설문 분기와 결측

- A-series의 상세 관광행태 문항은 여행유형에 따라 응답 가능 여부가 다르다.
- 허용된 통합 EDA에서 A-series 비교는 CASE 1과 2로 제한되었고, CASE 3의
  비결측 일부는 대표 표본으로 사용하지 않았다.
- CASE 2와 3을 비교할 때는 두 유형에 공통으로 관측되는 `D_TRA*` 변수만 사용한다.
- 숫자 `9`, 빈 문자열, 시스템 결측 등은 변수마다 의미가 다를 수 있다. 코드북과
  SAV의 `variable_value_labels`를 확인한 뒤 비해당·무응답·유효값을 구분한다.
- 다중응답은 항목별 선택률의 합이 100%를 넘을 수 있으므로 분모를 명시한다.

## 최소 품질 확인

- 실제 열 수, `ID` 유일성, `WT_DOM`의 결측·양수 여부를 확인한다.
- 필요한 변수가 프로젝트 코드북과 SAV 메타데이터에 모두 있는지 확인한다.
- long 변환 후 키 중복과 여행 슬롯별 관측 수를 확인한다.
- 표에는 비가중 N, 가중 분모, 결측 수와 가중 여부를 함께 기록한다.
- 조사설계의 층화·PSU 정보 없이 계산한 단순 표준오차를 공식 복합표본 오차로
  표현하지 않는다.
- `*_SPOT`을 지역명으로 변환할 때는 `region-code.md`의 복합키와 행정구역
  주의사항을 적용한다.
