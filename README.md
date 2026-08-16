# 프로젝트 구조 안내

이 프로젝트는 데이터 원본, 전처리 결과, 분석 노트북을 분리하고 공통 경로를 한 곳에서 관리하도록 구성했습니다.

```text
tourism_poster/
├─ run_all.py
├─ src/
│  ├─ __init__.py
│  └─ path.py
│
├─ data/
│  ├─ README.md
│  ├─ raw/
│  │  ├─ national_travel_survey/
│  │  │  ├─ y2023/
│  │  │  │  ├─ 2023년_국민여행조사_국내여행.SAV
│  │  │  │  ├─ 2023년_국민여행조사_코드북.csv
│  │  │  │  └─ region_code_2023.csv
│  │  │  ├─ y2024/
│  │  │  │  ├─ 2024년_국민여행조사_국내여행.sav
│  │  │  │  ├─ 2024년_국민여행조사_코드북.csv
│  │  │  │  └─ region_code_2024.csv
│  │  │  ├─ y2025/
│  │  │  │  ├─ 2025년_국민여행조사_국내여행.SAV
│  │  │  │  ├─ 2025년_국민여행조사_코드북.csv
│  │  │  │  ├─ national_travel_survey_codebook_2025.csv
│  │  │  │  ├─ national_travel_survey_codebook_2025_full.csv
│  │  │  │  └─ region_code_2025.csv
│  │  │  ├─ national_travel_survey_2023_2025.csv
│  │  │  ├─ national_travel_survey_2023_2025_codebook.csv
│  │  │  ├─ national_travel_survey_region_code.csv
│  │  │  └─ 2023_2025_국민여행조사_코드북_대응변수_및_처리방식.csv
│  │  ├─ pop_decline_danger/
│  │  │  └─ pop_decline_danger_region.csv
│  │  ├─ sigungu_map/
│  │  │  └─ 전국_시군구_2025_2Q.geojson
│  │  ├─ store/
│  │  │  ├─ 상권업종분류코드.csv
│  │  │  └─ 소상공인시장진흥공단_상가(상권)정보_{시도}_202512.csv (17개)
│  │  ├─ tour_site/
│  │  │  └─ 전국관광지정보표준데이터_관광지유형추가.geojson
│  │  └─ transportation/
│  │     ├─ 고속도로_출입시설.geojson
│  │     ├─ 고속터미널_5179.geojson
│  │     ├─ 국토교통부_전국 버스정류장 위치정보_20251031.csv
│  │     ├─ 국토교통부_전국버스정류장위치정보_20251031.geojson
│  │     ├─ 한국철도공사_역 위치 정보_20240401.csv
│  │     └─ 한국철도공사_역위치정보_20240401.geojson
│  │
│  └─ preprocess/
│
├─ notebooks/
│  └─ {분석 구분}
│     └─ yymmdd_분석주제.ipynb
│
├─ outputs/
│  └─ {출처 노트북 파일명}
│     ├─ figures/
│     └─ tables/
│
├─ docs/
│  ├─ apply/
│  ├─ note/
│  ├─ report/
│  └─ winner/
│
└─ tourism/
   └─ Python 가상환경
```

## 폴더별 역할

### `data/raw/`

외부에서 받은 원본 데이터와 원본에서 직접 파생한 보조 자료를 보관합니다.

- 원본 파일을 직접 수정하거나 덮어쓰지 않습니다.
- 원본에서 정제한 데이터(연도별 조합, 형 변환 등)는 반드시
  `data/preprocess/`에 별도로 저장합니다.
- 국민여행조사 자료는 `data/raw/national_travel_survey/`에 모아 두며,
  연도별 SAV·코드북·지역코드는 `y2023/`, `y2024/`, `y2025/` 하위 폴더에,
  3개년 통합 CSV·코드북·전용 지역코드는 상위 폴더에 둡니다.
- 인구감소지역 목록은 `data/raw/pop_decline_danger/pop_decline_danger_region.csv`에
  둡니다.
- 시군구 경계 geojson은 `data/raw/sigungu_map/`에 둡니다.
- 상권(상가) 정보는 `data/raw/store/`에 시도별 CSV로 둡니다.
- 관광지 표준데이터는 `data/raw/tour_site/`에 둡니다.
- 교통 시설(고속도로 출입시설, 버스정류장, 철도역, 고속버스터미널) 원자료와
  CSV에서 변환한 geojson은 모두 `data/raw/transportation/`에 둡니다.

### `data/preprocess/`

원본 데이터에서 추출하거나 분석에 맞게 정리한 데이터를 보관합니다.
SAV에서 직접 추출한 코드북은 원본과 함께 읽기 전용으로 관리하고, 이후 분석용
전처리 데이터만 이 폴더에 저장합니다.

### `notebooks/`

EDA, 후속 분석, 방문자 세분화 등 분석 노트북을 보관합니다.

노트북 이름에는 작성일과 분석 목적을 함께 표시합니다.

### `src/path.py`

프로젝트의 모든 공통 경로를 관리합니다.

노트북 내부에 `C:\Users\...` 같은 절대경로나 `../data` 같은 상대경로를 직접 작성하지 않습니다.

예시는 다음과 같습니다.

```python
sav_path = (
    NATIONAL_TRAVEL_SURVEY_RAW_DATA_DIR
    / "y2025"
    / "2025년_국민여행조사_국내여행.SAV"
)

codebook_path = (
    NATIONAL_TRAVEL_SURVEY_RAW_DATA_DIR
    / "national_travel_survey_2023_2025_codebook.csv"
)
```

이 구조 덕분에 프로젝트 폴더 전체를 다른 컴퓨터로 옮겨도 경로를 별도로 수정할 필요가 없습니다.

### `outputs/`

보고서와 발표자료에 사용할 최종 산출물을 저장합니다.

- `outputs/{출처 노트북 파일명}/figures/`: 그래프와 이미지
- `outputs/{출처 노트북 파일명}/tables/`: 핵심 지표와 결과표

```python
figure.savefig(
    FIGURES_DIR / "regional_comparison.png",
    dpi=900,
    bbox_inches="tight",
)

summary.to_csv(
    TABLES_DIR / "key_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)
```

### `docs/`

공모전과 분석에 필요한 문서를 보관합니다.

- `apply/`: 신청서, 모집요강, 제출 서류
- `note/`: 분석 계획과 연구 메모
- `report/`: 원자료 관련 공식 보고서
- `winner/`: 이전 수상작 참고 자료

### `tourism/`

프로젝트에서 사용하는 Python 가상환경입니다. 분석 결과나 소스코드를 넣는 폴더가 아닙니다.

## 협업 규칙

- `data/raw/`의 원본은 수정하지 않습니다.
- 새 전처리 데이터는 `data/preprocess/`에 저장합니다.
- 경로는 반드시 `src/path.py`에서 가져옵니다.
- 최종 그래프와 표는 `outputs/`에 저장합니다.
- 기존 노트북을 크게 수정할 때는 새 파일로 복사하기보다 변경 목적을 팀원과 먼저 공유합니다.
- 최종 수치나 그림을 변경했다면 관련 노트북을 처음부터 끝까지 다시 실행합니다.
