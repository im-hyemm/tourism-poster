"""`notebooks/04_Clustering/260726_연고관광_도메인별_클러스터_lift.ipynb` 노트북을 생성한다.

이 파일은 노트북에서 자동 생성된 파생물이다. 노트북 파일을 덮어쓰려면
반드시 `--write` 옵션을 명시해야 한다.
"""

from argparse import ArgumentParser
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / 'notebooks/04_Clustering/260726_연고관광_도메인별_클러스터_lift.ipynb'


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

cells = [
    markdown(
        """
        # 연고관광 도메인별 클러스터 lift

        ## 분석 목적과 범위

        기존 전체변수 클러스터 lift 결과 중 해석 가능한 특징을 도메인별로 다시
        정리한다. 각 도메인에서 층별로 차이가 큰 특징을 선정하고, 같은 특징의
        모든 군집 값을 함께 제시해 군집 프로필을 비교한다.

        - 입력은 기존 노트북이 저장한 `cluster_lift_all_variables.csv` 한 파일이다.
          원자료·군집 라벨링 자료·모형은 다시 불러오거나 재계산하지 않는다.
        - 방문 동기와 여행지 선택 이유는 `여행 행태·구조` 도메인에 포함한다.
        - 색과 셀 숫자는 모두 `log2(lift)`다. 0은 같은 층 평균, 양수는 과대표,
          음수는 과소대표를 뜻한다.
        - lift는 군집 특성의 상대 비교이며 인과효과가 아니다.
        """
    ),
    code(
        """
        import sys
        from pathlib import Path

        PROJECT_ROOT = Path.cwd().resolve()
        while not (PROJECT_ROOT / "src").is_dir():
            if PROJECT_ROOT.parent == PROJECT_ROOT:
                raise FileNotFoundError("src 폴더가 있는 프로젝트 루트를 찾지 못했습니다.")
            PROJECT_ROOT = PROJECT_ROOT.parent
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
        from textwrap import fill

        from src.path import OUTPUTS_DIR, create_output_directories
        from src.visualization import COLORS, apply_plot_style

        apply_plot_style()

        NOTEBOOK_NAME = "260726_연고관광_도메인별_클러스터_lift.ipynb"
        SOURCE_OUTPUT_NAME = "260726_연고관광_전체변수_클러스터_lift_260726_0745"
        SOURCE_TABLE_PATH = (
            OUTPUTS_DIR
            / SOURCE_OUTPUT_NAME
            / "tables"
            / "cluster_lift_all_variables.csv"
        )
        FIGURES_DIR, TABLES_DIR = create_output_directories(NOTEBOOK_NAME)

        STRATUM_COLUMN = "층"
        CLUSTER_COLUMN = "군집"
        DOMAIN_COLUMN = "도메인"
        MIN_CLUSTER_N = 30
        MIN_BASE_SHARE = 0.01
        TOP_FEATURES_PER_DOMAIN = 12
        COLOR_LIMIT = 3.0
        FEATURE_LABEL_WRAP_WIDTH = 28
        DOMAIN_ORDER = [
            "여행 행태·구조",
            "활동·경험",
            "지출·소비구성",
            "정보·예약·상품",
            "인구사회학",
            "여행 평가·영향",
        ]

        VISUAL_RANKED_PREFIXES = ("A3", "A5", "A5A", "D_TRA_B7", "ZQ1")
        VISUAL_EXCLUDED_VARIABLES = {
            "BARA",
            "primary_transport",
            "car_dependence",
            "public_transport",
            *{
                f"{prefix}_{rank}"
                for prefix in VISUAL_RANKED_PREFIXES
                for rank in range(2, 4)
            },
        }
        VISUAL_EXCLUDED_PREFIXES = ("NA8F",)
        INTERPRETATION_EXCLUDED_LEVELS = {
            "A4A": {"21"},
            **{
                f"A9A_{question_no}": {"9"}
                for question_no in range(1, 13)
            },
        }
        """
    ),
    code(
        """
        lift_all = pd.read_csv(SOURCE_TABLE_PATH, encoding="utf-8-sig")

        required_columns = {
            STRATUM_COLUMN,
            CLUSTER_COLUMN,
            "변수",
            "변수 라벨",
            "변수 유형",
            "표시 특징",
            "군집 셀 비가중 N",
            "층 내 가중 비율",
            "실질 수준 여부",
            "log2(lift)",
        }
        missing_columns = required_columns - set(lift_all.columns)
        assert not missing_columns, f"입력 결과표에 없는 열: {sorted(missing_columns)}"

        print(f"입력 표: {SOURCE_TABLE_PATH}")
        print(f"전체 lift 행: {len(lift_all):,}")
        display(lift_all.head(3))
        """
    ),
    code(
        """
        def classify_domain(row: pd.Series) -> str:
            \"\"\"변수와 라벨에 따라 해석 도메인을 부여한다.

            방문 동기·여행지 선택 이유는 여행 행태·구조로 분류한다.

            Args:
                row: 변수명과 변수 라벨을 가진 lift 결과 행.

            Returns:
                지정한 도메인명 또는 ``미분류``.
            \"\"\"
            variable = str(row["변수"])
            label = str(row["변수 라벨"])

            if variable.startswith(("A7", "A8", "NA7", "NA8", "ND1_")) or variable in {
                "spend_per_person",
                "spend_per_person_day",
                "destination_spend",
                "share_lodging",
                "share_food",
                "share_transport",
                "share_activity",
                "share_shopping",
                "activity_spend",
                "local_food_shopping_spend",
            }:
                return "지출·소비구성"

            if variable.startswith(("A4", "A4A")) or variable in {
                "activity_nature",
                "activity_food",
                "activity_culture",
                "activity_experience",
                "activity_leisure",
                "activity_wellness",
                "activity_shopping",
                "activity_event",
                "activity_group_count",
            }:
                return "활동·경험"

            if variable.startswith(("A1", "A2", "A5", "A5A")) or variable in {
                "package_used",
                "reservation_any",
                "reservation_count",
                "information_none",
                "information_online",
                "information_people",
            }:
                return "정보·예약·상품"

            if variable.startswith(("B", "DQ")) and variable not in {
                "BMON",
                "BARA",
            }:
                return "인구사회학"

            if variable.startswith(("A9", "A10", "A11", "A12")):
                return "여행 평가·영향"

            structure_variables = {
                "nights",
                "overnight_trip",
                "party_size",
                "party_group",
                "with_child",
                "visit_region_count",
                "multi_region",
                "family_home_lodging",
                "commercial_lodging",
                "primary_transport",
                "car_dependence",
                "public_transport",
                "trip_days",
            }
            if (
                variable.startswith(("D_TRA", "A3", "A6"))
                or variable in structure_variables
                or "선택한 이유" in label
                or "방문 동기" in label
            ):
                return "여행 행태·구조"

            return "미분류"


        excluded_level_mask = lift_all.apply(
            lambda row: str(row["수준 코드"])
            in INTERPRETATION_EXCLUDED_LEVELS.get(row["변수"], set()),
            axis=1,
        )
        candidate_mask = (
            lift_all["군집 셀 비가중 N"].ge(MIN_CLUSTER_N)
            & lift_all["실질 수준 여부"].eq(True)
            & lift_all["log2(lift)"].notna()
            & (
                lift_all["변수 유형"].eq("연속·계수형")
                | lift_all["층 내 가중 비율"].ge(MIN_BASE_SHARE)
            )
            & ~lift_all["변수"].isin(VISUAL_EXCLUDED_VARIABLES)
            & ~lift_all["변수"].str.startswith(VISUAL_EXCLUDED_PREFIXES)
            & ~lift_all["변수"].str.contains("_RANKW_", regex=False)
            & ~excluded_level_mask
        )
        domain_lift = lift_all.loc[candidate_mask].copy()
        domain_lift[DOMAIN_COLUMN] = domain_lift.apply(classify_domain, axis=1)
        domain_lift = domain_lift.loc[
            domain_lift[DOMAIN_COLUMN].isin(DOMAIN_ORDER)
        ].copy()

        domain_counts = (
            domain_lift.groupby([STRATUM_COLUMN, DOMAIN_COLUMN], observed=True)["변수"]
            .nunique()
            .reindex(DOMAIN_ORDER, level=DOMAIN_COLUMN, fill_value=0)
            .rename("후보 변수 수")
            .reset_index()
        )
        display(domain_counts)
        assert not domain_lift.empty
        """
    ),
    code(
        """
        feature_scores = (
            domain_lift.assign(_absolute_log2=domain_lift["log2(lift)"].abs())
            .groupby(
                [STRATUM_COLUMN, DOMAIN_COLUMN, "표시 특징"],
                observed=True,
                as_index=False,
            )["_absolute_log2"]
            .max()
            .sort_values(
                [STRATUM_COLUMN, DOMAIN_COLUMN, "_absolute_log2", "표시 특징"],
                ascending=[True, True, False, True],
            )
        )
        selected_features = (
            feature_scores.groupby(
                [STRATUM_COLUMN, DOMAIN_COLUMN], observed=True
            )
            .head(TOP_FEATURES_PER_DOMAIN)
            .rename(columns={"_absolute_log2": "선정 기준 |log2(lift)|"})
        )
        domain_summary = domain_lift.merge(
            selected_features[[STRATUM_COLUMN, DOMAIN_COLUMN, "표시 특징"]],
            on=[STRATUM_COLUMN, DOMAIN_COLUMN, "표시 특징"],
            how="inner",
            validate="many_to_one",
        )
        domain_summary[DOMAIN_COLUMN] = pd.Categorical(
            domain_summary[DOMAIN_COLUMN],
            categories=DOMAIN_ORDER,
            ordered=True,
        )
        domain_summary.to_csv(
            TABLES_DIR / "domain_cluster_lift_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        selected_features.to_csv(
            TABLES_DIR / "domain_cluster_lift_selection.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("도메인별로 선정한 특징")
        display(selected_features)
        """
    ),
    code(
        """
        def plot_domain_heatmap(
            frame: pd.DataFrame,
            stratum: str,
            domain: str,
        ) -> None:
            \"\"\"한 여행층·도메인의 log2(lift) 히트맵을 저장한다.

            Args:
                frame: 선정된 도메인별 lift 결과.
                stratum: 당일여행 또는 숙박여행 층 이름.
                domain: 시각화할 해석 도메인 이름.
            \"\"\"
            subset = frame.loc[
                (frame[STRATUM_COLUMN] == stratum)
                & (frame[DOMAIN_COLUMN] == domain)
            ].copy()

            if subset.empty:
                return

            feature_order = (
                subset.assign(_absolute_log2=subset["log2(lift)"].abs())
                .groupby(
                    "표시 특징",
                    observed=True,
                    as_index=False,
                )["_absolute_log2"]
                .max()
                .sort_values(
                    ["_absolute_log2", "표시 특징"],
                    ascending=[False, True],
                )["표시 특징"]
            )

            heatmap_data = (
                subset.pivot_table(
                    index="표시 특징",
                    columns=CLUSTER_COLUMN,
                    values="log2(lift)",
                    aggfunc="first",
                )
                .reindex(feature_order)
            )

            annotations = heatmap_data.map(
                lambda value: "" if pd.isna(value) else f"{value:+.2f}"
            )

            n_features = len(heatmap_data)
            n_clusters = len(heatmap_data.columns)

            # 행과 열 개수에 따라 그림 크기를 유동적으로 조정합니다.
            figure_width = max(7.5, 1.45 * n_clusters + 5.5)
            figure_height = max(4.2, 0.62 * n_features + 1.8)

            figure, axis = plt.subplots(
                figsize=(figure_width, figure_height),
                constrained_layout=False,
            )

            lift_cmap = LinearSegmentedColormap.from_list(
                "domain_log2_lift",
                [
                    COLORS["primary"],
                    COLORS["background"],
                    COLORS["negative"],
                ],
            )

            sns.heatmap(
                heatmap_data,
                cmap=lift_cmap,
                norm=TwoSlopeNorm(
                    vmin=-COLOR_LIMIT,
                    vcenter=0,
                    vmax=COLOR_LIMIT,
                ),
                annot=annotations,
                fmt="",
                linewidths=0.5,
                linecolor=COLORS["background"],
                square=False,
                cbar_kws={
                    "label": "log2(lift)",
                    "ticks": [-3, -2, -1, 0, 1, 2, 3],
                    "shrink": 0.92,
                    "pad": 0.05,
                },
                ax=axis,
            )

            # suptitle 대신 축 제목을 사용하면 위치 제어가 더 쉽습니다.
            axis.set_title(
                f"[{stratum}] {domain}",
                fontsize=17,
                fontweight="bold",
                pad=24,
            )

            axis.set_xlabel(
                "군집",
                labelpad=10,
            )
            axis.set_ylabel(
                "층 평균 대비 특징",
                labelpad=12,
            )

            axis.tick_params(
                axis="x",
                rotation=0,
                pad=6,
            )
            axis.tick_params(
                axis="y",
                labelsize=11,
                rotation=0,
                pad=4,
            )

            axis.set_yticklabels(
                [
                    fill(
                        str(label.get_text()),
                        width=FEATURE_LABEL_WRAP_WIDTH,
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                    for label in axis.get_yticklabels()
                ]
            )

            figure.text(
                0.99,
                0.015,
                "해석: 0=층 평균 · 양수=과대표 · 음수=과소대표",
                ha="right",
                va="bottom",
                fontsize=10,
                color=COLORS["neutral"],
            )

            # 긴 y축 라벨을 위한 왼쪽 공간만 확보하고 나머지는 자동 정리합니다.
            figure.subplots_adjust(
                left=0.37,
                right=0.91,
                top=0.88,
                bottom=0.14,
            )

            file_name = (
                f"{stratum}_{domain.replace('·', '_')}"
                "_cluster_lift_heatmap.png"
            )

            figure.savefig(
                FIGURES_DIR / file_name,
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.12,
                facecolor=figure.get_facecolor(),
            )

            plt.show()
            plt.close(figure)


        for stratum in domain_summary[STRATUM_COLUMN].unique():
            for domain in DOMAIN_ORDER:
                plot_domain_heatmap(domain_summary, stratum, domain)
        """
    ),
    code(
        """
        def plot_domain_heatmap(
            frame: pd.DataFrame,
            stratum: str,
            domain: str,
        ) -> None:
            \"\"\"한 여행층·도메인의 log2(lift) 히트맵을 저장한다.

            Args:
                frame: 선정된 도메인별 lift 결과.
                stratum: 당일여행 또는 숙박여행 층 이름.
                domain: 시각화할 해석 도메인 이름.
            \"\"\"
            subset = frame.loc[
                (frame[STRATUM_COLUMN] == stratum)
                & (frame[DOMAIN_COLUMN] == domain)
            ].copy()
            if subset.empty:
                return

            feature_order = (
                subset.assign(_absolute_log2=subset["log2(lift)"].abs())
                .groupby("표시 특징", observed=True, as_index=False)[
                    "_absolute_log2"
                ]
                .max()
                .sort_values(
                    ["_absolute_log2", "표시 특징"],
                    ascending=[False, True],
                )["표시 특징"]
            )
            heatmap_data = subset.pivot_table(
                index="표시 특징",
                columns=CLUSTER_COLUMN,
                values="log2(lift)",
                aggfunc="first",
            ).reindex(feature_order)
            annotations = heatmap_data.apply(
                lambda column: column.map(
                    lambda value: "" if pd.isna(value) else f"{value:+.2f}"
                )
            )

            figure, axis = plt.subplots(
                figsize=(10, max(4.5, 0.72 * len(heatmap_data)))
            )
            lift_cmap = LinearSegmentedColormap.from_list(
                "domain_log2_lift",
                [COLORS["primary"], COLORS["background"], COLORS["negative"]],
            )
            sns.heatmap(
                heatmap_data,
                cmap=lift_cmap,
                norm=TwoSlopeNorm(
                    vmin=-COLOR_LIMIT,
                    vcenter=0,
                    vmax=COLOR_LIMIT,
                ),
                annot=annotations,
                fmt="",
                linewidths=0.4,
                linecolor=COLORS["background"],
                cbar_kws={
                    "label": "log2(lift)",
                    "ticks": [-3, -2, -1, 0, 1, 2, 3],
                },
                ax=axis,
            )
            figure.suptitle(
                f"[{stratum}] {domain}",
                x=0.5,
                y=0.95,
                ha="center",
            )
            axis.set_xlabel("군집")
            axis.set_ylabel("층 평균 대비 특징")
            axis.tick_params(axis="x", rotation=0)
            axis.tick_params(axis="y", labelsize=11, rotation=0)
            axis.set_yticklabels(
                [
                    fill(
                        str(label.get_text()),
                        width=FEATURE_LABEL_WRAP_WIDTH,
                        break_long_words=True,
                        break_on_hyphens=False,
                    )
                    for label in axis.get_yticklabels()
                ]
            )
            figure.text(
                0.95,
                0.035,
                "해석: 0=층 평균 · 양수=과대표 · 음수=과소대표",
                ha="right",
                va="bottom",
                fontsize=10,
                color=COLORS["neutral"],
            )
            figure.subplots_adjust(left=0.42, bottom=0.15, top=0.92, right=0.95)
            file_name = (
                f"{stratum}_{domain.replace('·', '_')}_cluster_lift_heatmap.png"
            )
            with mpl.rc_context({"savefig.bbox": None}):
                figure.savefig(
                    FIGURES_DIR / file_name,
                    dpi=900,
                )
            plt.show()


        for stratum in domain_summary[STRATUM_COLUMN].unique():
            for domain in DOMAIN_ORDER:
                plot_domain_heatmap(domain_summary, stratum, domain)
        """
    ),
    markdown(
        """
        ## 해석 시 유의점

        - 각 그림은 같은 여행층 안에서 도메인별 절대 `log2(lift)`가 큰 특징 최대
          12개를 보여 준다. 한 도메인에서 선택된 특징은 그 층의 모든 군집에 같은
          순서로 제시한다.
        - 색 범위는 `-3`부터 `+3`까지로 고정했다. 절대값 3보다 큰 값은 색이
          포화되며, 정확한 값은 셀의 숫자로 확인한다.
        - 범주형의 희소 수준은 층 내 가중 비율 1% 이상, 군집 셀 비가중 N 30 이상인
          경우만 후보로 사용했다. `WT_DOM` 가중 lift의 분모와 분석 단위는 기존
          전체변수 lift 분석과 같다.
        """
    ),
    code(
        """
        from src.notebook_sync import sync_script_from_notebook

        print("최종 산출물")
        for path in sorted(TABLES_DIR.glob("domain_cluster_lift_*.csv")):
            print(f"- 표: {path.name}")
        for path in sorted(FIGURES_DIR.glob("*_cluster_lift_heatmap.png")):
            print(f"- 그림: {path.name}")

        sync_script_from_notebook(
            "notebooks/04_Clustering/260726_연고관광_도메인별_클러스터_lift.ipynb"
        )
        """
    ),
]



def create_notebook() -> nbf.NotebookNode:
    """현재 셀 목록으로 노트북 객체를 만든다.

    Returns:
        저장 가능한 노트북 객체다.
    """
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "tourism",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )


def main() -> None:
    """명시적 요청이 있을 때만 노트북 파일을 생성한다."""
    parser = ArgumentParser(description='260726_연고관광_도메인별_클러스터_lift.ipynb' + " 생성")
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
    print(f"생성 완료: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
