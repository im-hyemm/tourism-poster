"""프로젝트에서 사용하는 파일 시스템 경로를 정의한다."""

from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PREPROCESS_DATA_DIR = DATA_DIR / "preprocess"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# 기존 노트북의 무인자 호출과 import를 위한 호환 경로다.
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# 국민여행조사 데이터 경로
NATIONAL_TRAVEL_SURVEY_RAW_DATA_DIR = RAW_DATA_DIR / "national_travel_survey"
NATIONAL_TRAVEL_SURVEY_PREPROCESSED_DATA_DIR = PREPROCESS_DATA_DIR / "national_travel_survey"

# 인구감소지역 데이터 경로
POP_DECLINE_DANGER_DIR = RAW_DATA_DIR / "pop_decline_danger"
POP_DECLINE_DANGER_REGION_PATH = POP_DECLINE_DANGER_DIR / "pop_decline_danger_region.csv"

# 시군구 단위 관광연관 피처 데이터 경로
FEATURES_BY_REGION_DIR = PREPROCESS_DATA_DIR / "features_by_region"

def create_output_directories(
    notebook_filename: str | Path | None = None,
) -> tuple[Path, Path]:
    """노트북별 분석 결과 디렉터리를 생성한다.

    같은 노트북을 다시 실행해도 이전 실행 결과를 덮어쓰지 않도록, 실행 시각
    (yymmdd_hhmm)을 노트북 파일명 뒤에 붙여 매 실행마다 새 디렉터리를 만든다.

    Args:
        notebook_filename: 출력 경로의 기준이 되는 노트북 파일명이다. 확장자를
            제외한 파일명을 하위 디렉터리로 사용한다. 전달하지 않으면 기존
            노트북과의 호환을 위해 공용 출력 경로를 사용한다.

    Returns:
        그림 디렉터리와 결과표 디렉터리의 튜플이다.
    """
    if notebook_filename is None:
        directories = (FIGURES_DIR, TABLES_DIR)
    else:
        notebook_name = Path(notebook_filename).stem
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        notebook_output_dir = OUTPUTS_DIR / f"{notebook_name}_{timestamp}"
        directories = (
            notebook_output_dir / "figures",
            notebook_output_dir / "tables",
        )

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    return directories
