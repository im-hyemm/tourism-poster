"""제출 결과에 연결된 핵심 분석 노트북을 순서대로 실행한다."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from src.path import NOTEBOOKS_DIR


DEFAULT_NOTEBOOKS = (
    "02_EDA/260713_연고관광_통합_EDA.ipynb",
    "03_Aanalysis/260714_연고관광_후속분석_codex.ipynb",
    "03_Aanalysis/260717_연고관광_방문자_세분화.ipynb",
)


def execute_notebook(notebook_path: Path) -> None:
    """노트북을 처음부터 끝까지 실행하고 결과를 저장한다.

    Args:
        notebook_path: 실행할 노트북 경로.
    """
    with notebook_path.open(encoding="utf-8") as file_handle:
        notebook = nbformat.read(file_handle, as_version=4)

    client = NotebookClient(
        notebook,
        timeout=1_800,
        kernel_name="python3",
    )
    client.execute(cwd=NOTEBOOKS_DIR)

    with notebook_path.open("w", encoding="utf-8") as file_handle:
        nbformat.write(notebook, file_handle)


def parse_args() -> argparse.Namespace:
    """명령행 인수를 해석한다.

    Returns:
        해석된 명령행 인수.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "notebooks",
        nargs="*",
        default=DEFAULT_NOTEBOOKS,
        help="실행할 노트북 상대경로. 생략하면 핵심 노트북을 실행합니다.",
    )
    return parser.parse_args()


def main() -> None:
    """선택한 분석 노트북을 순서대로 실행한다."""
    args = parse_args()

    for notebook_name in args.notebooks:
        notebook_path = NOTEBOOKS_DIR / notebook_name
        if not notebook_path.is_file():
            raise FileNotFoundError(f"노트북을 찾을 수 없습니다: {notebook_path}")

        print(f"실행 중: {notebook_path.name}", flush=True)
        execute_notebook(notebook_path)
        print(f"완료: {notebook_path.name}", flush=True)


if __name__ == "__main__":
    main()
