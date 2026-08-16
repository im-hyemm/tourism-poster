"""프로젝트 시각화의 글꼴과 공통 스타일을 정의한다."""

from functools import lru_cache

import matplotlib as mpl
from matplotlib import font_manager

from src.path import PROJECT_ROOT


COLORS = {
    "primary": "#2F5D8C",
    "secondary": "#2A9D8F",
    "accent": "#E76F51",
    "highlight": "#E9C46A",
    "neutral": "#6B7280",
    "negative": "#C44E52",
    "background": "#FAFAF8",
    "ink": "#252A31",
    "grid": "#D9DDE3",
}

PALETTE = (
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["accent"],
    COLORS["highlight"],
    COLORS["neutral"],
)

FONT_DIR = (
    PROJECT_ROOT
    / "font"
    / "Pretendard-1.3.9"
    / "public"
    / "static"
)
FONT_FILES = (
    FONT_DIR / "Pretendard-Regular.otf",
    FONT_DIR / "Pretendard-Medium.otf",
    FONT_DIR / "Pretendard-SemiBold.otf",
    FONT_DIR / "Pretendard-Bold.otf",
)


@lru_cache(maxsize=1)
def register_pretendard_fonts() -> str:
    """프로젝트에 포함된 Pretendard 글꼴을 Matplotlib에 등록한다.

    Returns:
        Matplotlib이 인식한 Pretendard 글꼴 패밀리 이름.

    Raises:
        FileNotFoundError: 필요한 글꼴 파일이 없을 때 발생한다.
    """
    missing_files = [path for path in FONT_FILES if not path.is_file()]
    if missing_files:
        missing_text = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Pretendard 글꼴 파일이 없습니다: {missing_text}")

    for font_path in FONT_FILES:
        font_manager.fontManager.addfont(str(font_path))

    return font_manager.FontProperties(fname=str(FONT_FILES[0])).get_name()


def apply_plot_style() -> None:
    """프로젝트 공통 Matplotlib 스타일을 현재 세션에 적용한다."""
    font_family = register_pretendard_fonts()
    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.labelcolor": COLORS["ink"],
            "axes.edgecolor": COLORS["neutral"],
            "axes.facecolor": COLORS["background"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "figure.facecolor": COLORS["background"],
            "figure.titlesize": 16,
            "figure.titleweight": "bold",
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            "legend.fontsize": 10,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "savefig.facecolor": COLORS["background"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["neutral"],
            "ytick.color": COLORS["neutral"],
        }
    )
