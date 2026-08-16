# CASE 1·2 Base/Full 분류모형 노트북 작성 계획

## 1. 산출물과 실행 환경

- 새 노트북: `notebooks/03_Classification/260726_CASE1_CASE2_분류모형.ipynb`
- 생성 스크립트: `script/260726_CASE1_CASE2_분류모형.py`
  - `nbformat`으로 노트북을 생성한다.
  - 노트북과 생성 스크립트의 셀·메타데이터를 일치시킨다.
  - 커널은 로컬에서 `tourism` Python 3 환경을 사용한다.
- 최상단 설정 인터페이스:
  - `COLAB_MODE = False`
  - `COLAB_DATA_DIR = Path("/content/drive/MyDrive/tourism_poster/data")`
  - `RESUME_RUN_DIR = None`
  - `RANDOM_STATE = 20260726`
  - `SEARCH_CANDIDATES = 20`
  - `CV_FOLDS = 3`
  - `SHAP_SAMPLE_SIZE = 2_000`
- `COLAB_MODE=True`일 때:
  - `google.colab.drive.mount("/content/drive")`를 실행한다.
  - `src.path`를 사용하지 않고 사용자가 입력한 `COLAB_DATA_DIR`에서 입력 경로를 구성한다.
  - `OUTPUTS_DIR = COLAB_DATA_DIR.parent / "outputs"`로 두어 Drive의 `data`, `notebooks`, `outputs` 구조를 그대로 사용한다.
  - 빠진 `xgboost`, `lightgbm`, `catboost`, `shap`, `pyarrow`, `holidays` 패키지만 설치하고 실제 버전을 실행 메타데이터에 기록한다.
  - 로컬 전용 `src.visualization`과 노트북 동기화 기능은 조건부로 건너뛰고, 동일 색상·Pretendard 설정을 노트북 내부 대체 코드로 적용한다.
- 로컬에서는 `src.path`와 `create_output_directories()`를 사용한다. 마지막 셀은 Colab이 아닐 때 다음 노트북을 생성 스크립트와 동기화한다.
  - `notebooks/03_Classification/260726_CASE1_CASE2_분류모형.ipynb`

## 2. 데이터 전처리와 피처 계약

- 고정 입력:
  - `data/preprocess/national_travel_survey/national_travel_survey_2023_2025_preprocessed.csv`
  - `data/preprocess/national_travel_survey/national_travel_survey_2023_2025_preprocessed_codebook.csv`
- 분석 단위는 응답자·조사회차 1행, 키는 `YEAR + ID`다.
- `D_TRA_CASE == 1`을 타깃 0, `D_TRA_CASE == 2`를 타깃 1로 둔다.
  - 확인된 분석 표본: 70,694행
  - CASE 1: 57,904행
  - CASE 2: 12,790행, 18.1%
- `WT_DOM`은 품질 확인용 메타데이터로 보존하지만 모델 학습·분할·평가에는 사용하지 않는다.
- 코드북을 이용해 자료형, 값 라벨, 구조적 비해당 코드를 복원하고, 다음을 누출 변수로 강제 제외한다.
  - `YEAR`, `ID`, CASE·CHECK·타깃과 개별 여행유형 횟수
  - 원 거주 시도 `BARA`
  - 세부 날짜, 자유응답, 전부 결측, 연도·CASE 간 비교 불가 변수
  - 비용 포함 인원 `P*`, 의미가 중복되는 총액
  - `A4_18`, 만족도·재방문·추천·여행효과
- Base 피처는 다음 의미 중심 변수로 고정한다.
  - 여행자: 연령대, 가구소득구간, 가구규모, 혼인, 아동 포함 가구
  - 일정: 계절, 당일·숙박, 여행일수 구간, 주말·공휴일 포함
  - 동반: 1인·2인·3~4인·5인 이상, 아동 동반, 동반유형 수, 가족·친지 및 친구 동반
  - 이동: 자가용·렌터카, 대중교통, 기타 교통, 교통수단 수
  - 방문지역: 관내·관외, 수도권·비수도권·혼합, 단일·다지역
  - 숙박: 가족·친지집만, 상업숙박만, 혼합, 기타·무박
  - 예약: 예약 여부·개수, 교통·숙박·활동 예약
  - 정보: 정보 없음, 온라인, 주변인, 정보원 수
  - 활동: 자연·미식·문화·체험·레저·웰니스·쇼핑·이벤트 및 활동군 수
  - 소비: 1인·1일 지출 원값, 학습자료에서 정한 99% 상한과 로그 변환, 항목별 지출 여부·구성비
- Full은 Base 전체에 다음 세부변수를 추가한다.
  - 이동수단 1~3순위와 순위 가중치
  - 방문 시도 여부, 숙박시설 코드별 이용 여부
  - 예약 세부항목, `A4_18`을 제외한 활동 세부항목
  - 정보원·인터넷 사이트, 방문지 선택 이유와 순위 가중치
  - 동반자 세부유형, 항목별 지출액·구성비
  - 코드북상 CASE 1·2에서 공통 관측되는 동일 도메인의 유효 여행행태 변수
- 행 단위 파생은 분할 전에 수행하되, 결측 대치·인코딩·스케일링·지출 상한은 Train에서만 학습한다.
- 전처리 결과는 원본을 건드리지 않고 다음에 원자적으로 저장한다.
  - `data/preprocess/classification/case1_case2_base.parquet`
  - `data/preprocess/classification/case1_case2_full.parquet`
  - `data/preprocess/classification/feature_dictionary.csv`
  - `data/preprocess/classification/exclusion_log.csv`
  - `data/preprocess/classification/split_assignment.csv`
- Parquet에는 키·타깃·`WT_DOM`을 함께 보존하되 피처 사전에 모델 입력 여부를 명시한다. 기존 파일이 완전하면 재로딩해 전처리를 생략한다.

## 3. 분할, 모델 학습과 즉시 저장

- `YEAR × target` 조합으로 공통 층화하여 Train 70%, Validation 15%, Test 15%로 나눈다.
- Base와 Full의 모든 모델은 `split_assignment.csv`의 동일한 행 분할을 사용한다.
- 비교 모델:
  1. Elastic Net 로지스틱 회귀
  2. 랜덤포레스트
  3. XGBoost
  4. LightGBM
  5. CatBoost
- Base에서만 모델별 20개 하이퍼파라미터 후보를 동일한 3-fold 층화 CV로 탐색한다.
  - Elastic Net: `C`, `l1_ratio`, class weight
  - 랜덤포레스트: 트리 수·깊이·최소 잎 크기·피처 비율·class weight
  - Boosting 3종: 트리 수·깊이/잎 수·학습률·행/열 샘플링·규제·불균형 보정
- Full은 계산시간 절감을 위해 각 알고리즘에서 Base가 확정한 최적 하이퍼파라미터를 그대로 적용하고 별도 탐색하지 않는다.
- 각 모델은 Train으로 학습한 뒤 Validation 예측에서 `0.05~0.95`, 간격 `0.005`로 F1 최적 임계값을 선택한다. 동률이면 0.5에 가까운 임계값을 사용한다.
- 선택 순서는 Validation F1 → PR-AUC → Base 우선이다. ROC-AUC는 참고 지표로만 저장한다.
- 최종 선택 구성만 Train+Validation 85%로 재학습하고, 고정 임계값으로 Test를 한 번 평가한다.
- Base 최고 모델과 Full 최고 모델의 Validation 예측을 이용해 2,000회 paired bootstrap으로 `Full−Base F1` 분포와 95% 구간을 항상 저장한다.
- 장시간 작업은 다음처럼 즉시 체크포인트한다.
  - CV fold 하나가 끝날 때마다 탐색 진행표와 상태 JSON을 임시 파일 후 교체 방식으로 저장한다.
  - 모델별 탐색 종료 즉시 최적 파라미터, Train 적합 모델, Validation 확률·임계값·지표를 저장한다.
  - Full 모델도 알고리즘 하나가 끝날 때마다 같은 구조로 저장한다.
  - `RESUME_RUN_DIR`가 지정되면 완료된 fold·후보·모델을 건너뛰고 이어서 실행한다.
  - CatBoost는 `allow_writing_files=False`로 두어 노트북 폴더에 `catboost_info`를 만들지 않는다.
- 실행별 저장 루트:
  - `outputs/260726_CASE1_CASE2_분류모형_{yymmdd_hhmm}/figures/`
  - `outputs/260726_CASE1_CASE2_분류모형_{yymmdd_hhmm}/tables/`
  - `outputs/260726_CASE1_CASE2_분류모형_{yymmdd_hhmm}/models/`
- `models/`에는 모델별 체크포인트, 전처리 파이프라인, 최종 모델, 고정 임계값, 예측값, 패키지 버전과 실행 상태를 저장한다.

## 4. SHAP, 민감도 분석과 검증

- Base 최고 모델과 Full 최고 모델을 동일한 층화 Validation 표본으로 설명한다.
  - 트리 모델은 `TreeExplainer`, Elastic Net은 `LinearExplainer`를 사용한다.
  - SHAP 배열·설명 대상 값·피처명·표본 키를 즉시 저장한다.
  - 개별 변수 mean absolute SHAP 상위 20개와 beeswarm을 생성한다.
  - 원핫 피처를 원변수로 다시 합산한 뒤 도메인 중요도, CASE 1·2 방향, 도메인별 상위 3~5개 변수를 저장한다.
- 관계 직접신호 민감도는 최종 선택 알고리즘·고정 하이퍼파라미터로만 수행한다.
  - 가족·친지집 숙박 코드 12와 이를 포함한 숙박유형 파생변수
  - `A6B_1`, `A6B_2` 가족·친지 동반
  - `A6B_3` 친구 동반
  - 제거 후 Validation 임계값은 다시 정하되 모델 경쟁과 Test 평가에는 포함하지 않는다.
  - 본모형 대비 Validation F1·PR-AUC 변화, SHAP 상위 20개 중첩과 도메인 순위 변화를 저장한다.
- 표와 그림:
  - 표본 흐름 및 연도별 CASE 구성
  - Base 탐색 결과와 Base/Full Validation leaderboard
  - paired bootstrap 결과
  - 최종 Test 지표·혼동행렬·PR 곡선
  - Base/Full SHAP bar·beeswarm, Full 도메인 SHAP
  - 직접신호 제거 민감도 비교
- 시각화는 프로젝트 색상과 Pretendard를 적용하고, 제목에 2023~2025년·비가중 분석·분모 N을 표시하며 PNG를 900 DPI 이상으로 저장한다.
- 완료 검증:
  - 입력 열과 코드북 행의 1:1 대응, `YEAR + ID` 유일성, 타깃 건수 확인
  - 제외목록·정규식 누출 검사와 `WT_DOM` 미사용 확인
  - Base/Full의 키·분할 완전 일치와 각 split의 `YEAR × target` 분포 확인
  - Validation/Test에 전처리기가 적합되지 않았는지 확인
  - 모든 확률·평가지표의 유효 범위와 혼동행렬 합계 확인
  - 저장한 Parquet·모델·SHAP 파일을 다시 열어 손상 여부 확인
  - 로컬/Colab 경로 분기와 중단 후 재개 동작 확인
  - 생성 스크립트 실행 후 유효한 `.ipynb`인지 확인하되 장시간 모델 학습은 자동 실행하지 않음

## 가정

- Full의 하이퍼파라미터를 별도로 탐색하지 않으므로 Base보다 불리할 수 있으며, 이를 계산예산 제약에 따른 한계로 명시한다.
- 모델 성능과 SHAP은 CASE 1·2의 연관 특성으로만 해석하며 인과효과로 표현하지 않는다.
- 공개 `src` API는 추가하지 않고 반복 사용 가능성이 확인되지 않은 로직은 노트북 내부에 둔다.
