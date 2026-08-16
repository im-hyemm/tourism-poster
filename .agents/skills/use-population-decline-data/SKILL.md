---
name: use-population-decline-data
description: data/raw/pop_decline_danger/pop_decline_danger_region.csv의 인구감소지역 목록을 해석한다. 인구감소지역 여부 생성, 지역별 필터링·집계, 지역코드와의 결합 또는 해당 결과 검토가 필요할 때 사용한다.
---

# 인구감소지역 데이터 사용

1. 데이터를 사용하거나 결합하기 전에 `references/schema.md`를 읽는다.
2. 이 파일을 89개 지정 지역의 명칭 목록으로만 사용한다.
3. 국민여행조사 방문지역과 결합하면 `use-national-travel-survey`의
   `references/region-code.md`를 함께 읽고 명칭을 명시적으로 정규화한다.
4. 결합 전후 행 수, 중복, 미매칭 지역을 확인하고 결과에 남긴다.
5. 지정 연도나 법적 기준을 설명할 때는 원본 출처를 별도로 확인한다.

## 금지 사항

- 이 파일만으로 인구감소의 정도, 위험 점수, 지정 사유를 추론하지 않는다.
- 시도 약칭을 지역코드 파일의 정식 명칭과 직접 결합하지 않는다.
- 미매칭 지역을 누락한 채 결합 성공으로 보고하지 않는다.
