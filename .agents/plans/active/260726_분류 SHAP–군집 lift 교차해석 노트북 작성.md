# 분류 SHAP–군집 lift 교차해석 노트북 작성

## 요약

- `Full-LightGBM`이 최적모형임을 저장된 leaderboard로 재확인하고 해당 SHAP만 사용한다.
- 기존 분류·군집 노트북은 수정하거나 재실행하지 않는다.
- 새 노트북은 `notebooks/07_Cross/260726_분류_SHAP_군집_lift_교차해석.ipynb`, 생성 스크립트는 `script/260726_분류_SHAP_군집_lift_교차해석.py`로 작성한다.
- 중앙값 기반 4분면을 주 결과로, 고정 기준 기반 분류를 민감도 결과로 함께 제시한다.
- `docs/note/260726_연고관광_최종_분석_흐름.md`는 수정하지 않고, 7절에 옮길 수 있는 해석 문안을 노트북에 제공한다.

## 구현 내용

- 입력은 분류 결과의 `validation_leaderboard.csv`, `full_lightgbm_shap_domains.csv`, `full_lightgbm_shap_original_features.csv`와 지정된 `cluster_lift_all_variables.csv`만 사용한다.
- SHAP은 변환 피처가 아닌 원 피처 단위 `mean_abs_shap`, 순위, 방향, 도메인을 사용한다. 군집 차이는 각 피처에서 품질조건을 통과한 수준들의 `max(abs(log2(lift)))`로 나타내고, 그 값을 만든 군집·수준·실제 값·층 기준값·N을 함께 기록한다.
- 군집 lift 대표 수준은 실질 수준, 양의 유효 lift, 군집 N≥30을 만족해야 한다. 범주형은 층 비중≥1%, 해석용 대표 수준은 수준 N≥30도 적용한다. 희소 수준은 감사표에는 남기되 결론에서는 제외한다.
- 변수 연결은 다음 순서로 관리한다.
  - 원변수와 정의가 같은 피처는 직접 연결한다.
  - 검증된 별칭인 `age_group→BAGE`, `household_income_band→BINC1`, `household_size→BFAM`, `marital_status→BMAR`, `party_size_band→party_group`, `with_friend→A6B_3`, `with_family_relative→A6B_1/A6B_2`, `share_a8f→share_transport`, `spent_a8g→activity_spend`를 명시적으로 연결한다.
  - `has_lodging_12→family_home_lodging`은 정의상 연결하되 군집 lift에서 제외된 사실을 표시한다.
  - 이름이 같아도 정의가 다른 활동 대분류·`activity_group_count`·`reservation_count` 등은 결합하지 않는다.
  - 숙박 직접신호, 사전층화 변수, 방문시도, 분모가 다른 상세 소비변수 등은 미매칭·비교불가 사유를 남긴다.
- 주 4분면은 비교 가능한 피처들의 SHAP과 최대 lift 편차 각각의 중앙값을 축으로 사용한다.
- 민감도 4분면은 `SHAP 상위 20개`와 `lift≥1.5 또는 ≤0.67`을 기준으로 사용한다. 두 방식의 유형 일치 여부와 변경 피처를 별도 표로 제시하며, 고정 기준은 통계적 유의성 기준이 아닌 실무적 표시 기준임을 명시한다.
- 도메인 표에는 Full-LightGBM 도메인 SHAP 순위, 연결 가능한 피처 수, 대표 군집 차이와 근거 피처를 기록한다. 도메인 합성점수나 새로운 모형은 만들지 않는다.
- 중앙값·고정 기준을 나란히 보여주는 2패널 산점도를 생성한다. `src.visualization`의 Pretendard·프로젝트 색상을 적용하고 900 DPI로 저장한다.
- 출력표는 `feature_mapping_audit.csv`, `cross_feature_evidence.csv`, `cross_domain_summary.csv`, `quadrant_rule_comparison.csv`로 제한한다. 공개 API나 `src/` 인터페이스 변경은 없다.

## 노트북 구성

1. `tl;dr`와 해석 범위
2. 입력 경로·최적모형 확인
3. SHAP·lift 스키마 및 분석 단위 검증
4. 변수 정의 매칭 감사
5. 개별 피처 교차근거표
6. 중앙값 4분면과 고정 기준 민감도
7. 도메인별 교차해석
8. 최종 문서 7절에 옮길 수 있는 공통 정책·군집별 정책·핵심 정책축 문안
9. 한계와 과대해석 방지사항
10. 생성 스크립트 동기화 셀

## 검증 및 완료 기준

- leaderboard 최상단과 최종 성능표가 모두 `Full-LightGBM`을 가리키는지 단언한다.
- lift 키의 중복, 수치 범위, 가중 분모, 실질 수준과 희소 수준 처리를 검사한다.
- 모든 SHAP 피처가 `정의 일치·검증 별칭·정의 불일치·군집표 제외·미매칭` 중 하나로 분류되도록 한다.
- 1:N 별칭이 의도치 않은 행 증폭을 만들지 않는지 검증한다.
- 중앙값 기준과 고정 기준의 축·유형 배정 및 대표 근거행을 재계산해 저장표와 대조한다.
- 새 노트북만 `tourism` 가상환경에서 처음부터 끝까지 실행하고, 분류·군집 원본 노트북의 해시가 전후 동일한지 확인한다.
- 저장된 4분면 그림을 직접 열어 한글 깨짐, 레이블 겹침, 잘림과 색상 의존성을 점검한다.
- 결론은 연관성과 내부 유형 차이로만 표현하며 인과효과, 출생지·법적 고향, 새로운 복합점수로 확대하지 않는다.
