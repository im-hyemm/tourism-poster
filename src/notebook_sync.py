"""현재 노트북을 기준으로 안전한 생성 스크립트를 만든다.

생성 스크립트는 노트북의 파생물이다. 따라서 동기화할 때마다 현재 노트북의
셀 전체로 다시 만들며, 스크립트가 없어도 새로 생성한다. 생성된 스크립트는
`--write` 옵션을 명시해야만 노트북 파일을 쓸 수 있어 실수로 실행해도
노트북을 덮어쓰지 않는다.
"""

from pathlib import Path

import nbformat as nbf

from src.path import PROJECT_ROOT

SCRIPT_DIR = PROJECT_ROOT / "script"


def _indent(text: str, prefix: str = " " * 8) -> str:
    """빈 줄은 그대로 두고 나머지 줄에만 들여쓰기를 적용한다."""
    lines = text.splitlines()
    return "\n".join(prefix + line if line.strip() else "" for line in lines)


def _render_cell(cell: nbf.NotebookNode) -> str:
    """노트북 셀 하나를 생성 스크립트의 `markdown()`/`code()` 호출로 렌더링한다."""
    if cell.cell_type not in ("markdown", "code"):
        raise ValueError(f"지원하지 않는 셀 유형입니다: {cell.cell_type}")

    kind = cell.cell_type if cell.cell_type == "markdown" else "code"
    # 백슬래시와 셀 내부의 `"""`는 감싸는 삼중따옴표 문자열과 충돌하므로
    # 이스케이프한다. 백슬래시를 먼저 이스케이프해야 뒤이어 넣는 `\"\"\"`의
    # 백슬래시가 다시 이스케이프되지 않는다.
    escaped_source = cell.source.rstrip("\n").replace("\\", "\\\\")
    escaped_source = escaped_source.replace('"""', '\\"\\"\\"')
    source = _indent(escaped_source)
    return f'    {kind}(\n        """\n{source}\n        """\n    ),'


def _render_cells_block(cells: list[nbf.NotebookNode]) -> str:
    """노트북 셀 목록을 생성 스크립트의 `cells = [...]` 블록 문자열로 렌더링한다."""
    body = "\n".join(_render_cell(cell) for cell in cells)
    return f"\ncells = [\n{body}\n]\n"


def _render_script(notebook_path: Path, cells: list[nbf.NotebookNode]) -> str:
    """노트북 전체를 재현하는 안전한 생성 스크립트를 렌더링한다.

    Args:
        notebook_path: 생성 대상 노트북의 절대 경로다.
        cells: 생성 스크립트에 기록할 노트북 셀 목록이다.

    Returns:
        UTF-8 텍스트로 저장할 완성된 Python 스크립트다.
    """
    relative_notebook_path = notebook_path.relative_to(PROJECT_ROOT).as_posix()
    notebook_name = notebook_path.name
    cells_block = _render_cells_block(cells)

    return f'''"""`{relative_notebook_path}` 노트북을 생성한다.

이 파일은 노트북에서 자동 생성된 파생물이다. 노트북 파일을 덮어쓰려면
반드시 `--write` 옵션을 명시해야 한다.
"""

from argparse import ArgumentParser
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / {relative_notebook_path!r}


def markdown(text: str) -> nbf.NotebookNode:
    """마크다운 셀을 만든다.

    Args:
        text: 셀에 넣을 마크다운 문자열이다.

    Returns:
        생성된 마크다운 셀이다.
    """
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbf.NotebookNode:
    """코드 셀을 만든다.

    Args:
        text: 셀에 넣을 Python 코드다.

    Returns:
        생성된 코드 셀이다.
    """
    return nbf.v4.new_code_cell(dedent(text).strip())
{cells_block}


def create_notebook() -> nbf.NotebookNode:
    """현재 셀 목록으로 노트북 객체를 만든다.

    Returns:
        저장 가능한 노트북 객체다.
    """
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={{
            "kernelspec": {{
                "display_name": "tourism",
                "language": "python",
                "name": "python3",
            }},
            "language_info": {{"name": "python", "version": "3"}},
        }},
    )


def main() -> None:
    """명시적 요청이 있을 때만 노트북 파일을 생성한다."""
    parser = ArgumentParser(description={notebook_name!r} + " 생성")
    parser.add_argument(
        "--write",
        action="store_true",
        help="현재 노트북 파일을 생성하거나 덮어쓴다.",
    )
    arguments = parser.parse_args()

    if not arguments.write:
        print("노트북을 보호하기 위해 쓰지 않았습니다. 생성하려면 --write를 사용하세요.")
        return

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(create_notebook(), NOTEBOOK_PATH)
    print(f"생성 완료: {{NOTEBOOK_PATH}}")


if __name__ == "__main__":
    main()
'''


def sync_script_from_notebook(
    notebook_path: str | Path, script_path: str | Path | None = None
) -> Path:
    """저장된 노트북 내용으로 대응하는 생성 스크립트를 만든다.

    Args:
        notebook_path: 동기화할 노트북 경로. 프로젝트 루트 기준 상대 경로
            또는 절대 경로를 받는다.
        script_path: 생성할 스크립트 경로. 전달하지 않으면
            `script/{노트북 파일명}.py`를 사용한다.

    Returns:
        생성한 스크립트 경로다.
    """
    notebook_path = Path(notebook_path)
    if not notebook_path.is_absolute():
        notebook_path = PROJECT_ROOT / notebook_path

    if script_path is None:
        script_path = SCRIPT_DIR / f"{notebook_path.stem}.py"
    else:
        script_path = Path(script_path)
        if not script_path.is_absolute():
            script_path = PROJECT_ROOT / script_path

    notebook = nbf.read(notebook_path, as_version=4)
    script_text = _render_script(notebook_path, notebook.cells)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_text, encoding="utf-8")
    print(f"생성 스크립트를 노트북 기준으로 만들었습니다: {script_path}")
    return script_path
