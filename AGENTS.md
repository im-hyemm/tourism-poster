# 프로젝트 지침

## 기본 원칙

- 사용자에게 한국어로 답변한다.
- Python 코드는 PEP 8을 준수한다.
- 공개 함수와 클래스에는 한국어 Google Style docstring을 작성한다.
- YAGNI 원칙을 따르고 요청되지 않은 구조나 기능을 추가하지 않는다.
- tourism 가상환경을 사용한다.

## 프로젝트 목적

이 프로젝트는 2026년 관광데이터 분석 포스터 공모전 출품작을 준비한다.
주제는 `연고 방문을 활용한 지역관광 활성화 전략`이며, 국민여행조사를 핵심
데이터로 사용해 관광 현상을 설명하고 실행 가능한 정책 제언으로 연결한다.

## 파일과 데이터

- `data/raw/`의 원본 파일을 수정하거나 덮어쓰지 않는다.
- 국민여행조사 원본과 전용 보조 자료는
  `data/raw/national_travel_survey/`에 저장한다.
- 인구감소지역 목록은 `data/raw/pop_decline_danger/pop_decline_danger_region.csv`를
  사용한다.
- 전처리 결과는 `data/preprocess/`에 저장한다.
- 공통 경로는 `src/path.py`에서 가져오고 개인 절대경로나 임의의 `../data`
  경로를 코드에 작성하지 않는다.
- 사용자가 저장을 요청했거나 포스터·보고서에 사용할 최종 산출물인 경우에만 기본 경로를 다음과 같이 지정한다.
  이외의 탐색·검증용 출력은 노트북에만 남긴다.
    - 그림: outputs/{노트북 파일명}\_{yymmdd_hhmm}/figures/
    - 결과표: outputs/{노트북 파일명}\_{yymmdd_hhmm}/tables/
    - 모델 학습 결과: outputs/{노트북 파일명}\_{yymmdd_hhmm}/models/
- 핵심 수치나 그림을 바꾸면 관련 노트북을 처음부터 끝까지 다시 실행하고,
  보고서 수치를 저장된 표 및 노트북 출력과 대조한다.

## 프로젝트 스킬

작업에 필요한 스킬만 선택해 사용한다.

- 공모전 요건·연구 목적·수상작 패턴: `use-tourism-context`
- 2025년 국민여행조사와 전용 지역코드: `use-national-travel-survey`
- 인구감소지역 목록: `use-population-decline-data`
- 분석 노트북 생성·수정: `build-analysis-notebook`
- 그래프·포스터 시각화: `visualize-tourism-results`
- 시군구 경계 지도 결합·단계구분도: `use-sigungu-map`

공모전 문맥과 개별 데이터의 특성을 한 문서에 섞지 않는다. 데이터 결합 작업은
관련된 모든 데이터 스킬을 읽고 각 스킬이 소유한 키와 주의사항을 함께 적용한다.
`national_travel_survey_region_code.csv`는 국민여행조사 전용 보조 코드로 취급한다.
