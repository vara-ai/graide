"""
Generates Figure 2 from the paper. This script requires that `run_screening.py` has been run for all strategies on
all datasets: [germany/sweden/uk]_[val/test].
"""
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

from graide import constants as gc
from graide.artefacts.plot_util import (
    COLOR_AI,
    COLOR_REF,
    plot_roc_curve,
    Result2Plot,
    seaborn_context,
    STRATEGY_COLOR_MAP,
)
from graide.artefacts.util import get_best_results_by_strategy, get_ref_result, PRETTY_STRATEGY_NAMES
from graide.conf.lib import compose_one
from graide.stats import rates_curve_compute
from graide.strategy import ScreeningStrategy
from graide.util import path_util
from graide.util.logging import logger
from graide.util.types import StrategyResult, PredsAndMetadata, ReferenceResult

sns.set_style("whitegrid")

DECISION_LEVEL = gc.DecisionLevel.PROGRAM

AXIS_LABEL_FONTSIZE = 12
MARKER_SIZE = 20
MARKER_SIZE_INSERT = 50


@seaborn_context(context="paper", font_scale=1.0)
def plot_figure_2(
    preds_and_md: PredsAndMetadata,
    ref_result: ReferenceResult,
    strategy_results: dict[str, dict[str, StrategyResult]],
    ax: Axes,
    n_bootstrap: int = 1000,
    show_legend: bool = True,
    title: str | None = None,
    title_fontsize: int = 14,
) -> Figure:

    RESULT_MARKERS = {"CDR": "o", "Recall": "^", "Balanced": "s"}

    # Create a list of results to plot as points, including ...
    create_result2plot = partial(
        Result2Plot.from_result,
        with_error_bars=n_bootstrap > 0,
        label_with_metrics=max([len(_strategy_results) for _strategy_results in strategy_results.values()]) == 1,
    )

    # ... the reference result ...
    results2plot = [create_result2plot(ref_result, label=DECISION_LEVEL.value, color=COLOR_REF, fmt="o")]

    # ... and all strategy results (only including a label / legend entry for the first of each strategy)
    result_names_list = []
    for strategy_name, results in strategy_results.items():
        for result_idx, (result_name, result) in enumerate(results.items()):
            results2plot.append(
                create_result2plot(
                    result,
                    label=strategy_name if result_idx == 0 else None,
                    color=STRATEGY_COLOR_MAP[strategy_name],
                    fmt=RESULT_MARKERS[result_name],
                )
            )
            if result_name not in result_names_list:
                result_names_list.append(result_name)

    def _draw_results(ax: Axes, marker_size: float = 4, elinewidth: float = 0.5):
        """A helper to that plots the above results2plot on a given Axes and returns the list of plotted points."""
        return [result2plot.plot(ax, marker_size, elinewidth) for result2plot in results2plot]

    # Add the ROC curve
    recall, cdr, cdr_low, cdr_high = rates_curve_compute(
        preds_and_md.y_true, preds_and_md.y_score_bal, n_bootstrap, preds_and_md.weights
    )
    plot_roc_curve(recall, cdr, cdr_low, cdr_high, label="AI", color=COLOR_AI, ax=ax, clip_on=False)

    # Draw results with errorbars on the main chart
    _draw_results(ax, marker_size=MARKER_SIZE, elinewidth=1)
    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(0.0, 9.5)
    ax.grid(False)

    # Set axis labels & create legend
    ax.set_xlabel("Recall", fontdict={"fontsize": AXIS_LABEL_FONTSIZE})
    ax.set_ylabel("CDR [‰]", fontdict={"fontsize": AXIS_LABEL_FONTSIZE})

    # Find the figure from the axis we were passed
    fig: Figure = ax.get_figure()

    if show_legend:

        # 1) Add a legend distinguishing the strategies. We default to placing this legend toward the upper right of the
        #    single axis figure (i.e. the plot we generate using `run_plot_results`). Any of the default configuration
        #    values can be overridden or any additional values passed via the `l1_kwargs` argument above.

        colors = [COLOR_REF] + [STRATEGY_COLOR_MAP[p_name] for p_name in strategy_results.keys()]
        handles = [
            Line2D([0], [0], linestyle="none", marker="_", markersize=8, color=c, markeredgewidth=3) for c in colors
        ]
        labels = [DECISION_LEVEL.value] + list(strategy_results.keys())

        fig.add_artist(
            fig.legend(
                handles,
                labels,
                title="Screening Strategy",
                fontsize=12,
                title_fontsize=14,
                loc="lower left",
                bbox_to_anchor=(0.32, -0.1),
            )
        )

        # 2) ... & maybe an additional legend distinguishing the different result types ('CDR', 'Recall', 'Balanced').
        #    This smaller legend usually appears in the upper right corner of the system performance plot.
        if len(result_names_list) > 1:

            handles = [
                Line2D([0], [0], linestyle="none", marker=m, markersize=3, color="gray")
                for m in RESULT_MARKERS.values()
            ]
            labels = result_names_list

            fig.add_artist(
                fig.legend(
                    handles,
                    labels,
                    title="optimised for",
                    fontsize=12,
                    title_fontsize=14,
                    loc="lower left",
                    bbox_to_anchor=(0.54, -0.1),
                )
            )

    # Now add an insert to zoom into the results
    axins = ax.inset_axes((0.3, 0.08, 0.69, 0.48))
    _draw_results(axins, marker_size=MARKER_SIZE_INSERT, elinewidth=1.5)
    ax.indicate_inset_zoom(axins)
    axins.grid(True)

    # Maybe display a title
    if title is not None:
        ax.set_title(title, fontdict={"fontsize": title_fontsize})

    return fig


def _get_results_by_strategy(
    preds_and_md: PredsAndMetadata,
    preds_and_md_result_selection: PredsAndMetadata,
    strategies: list[ScreeningStrategy],
    screening_system: gc.ScreeningSystem = gc.ScreeningSystem.GERMANY,
) -> dict[str, dict[str, StrategyResult]]:
    """A wrapper around `get_best_results_by_strategy` that selects only the results we want to display."""

    results2display_by_strategy = get_best_results_by_strategy(
        strategies, preds_and_md_result_selection, preds_and_md, screening_system=screening_system
    )

    results2plot = {}
    for strategy_name, (cdr_result, balanced_result, recall_result) in results2display_by_strategy.items():
        strategy_name_pretty = PRETTY_STRATEGY_NAMES[strategy_name]

        results2plot[strategy_name_pretty] = {}
        if strategy_name.endswith("DD") or strategy_name in {"DeferralToASingleReader", "Graide"}:
            results2plot[strategy_name_pretty]["CDR"] = cdr_result.result
            results2plot[strategy_name_pretty]["Recall"] = recall_result.result
        if strategy_name == "NormalTriaging":
            results2plot[strategy_name_pretty]["Recall"] = recall_result.result
        else:
            results2plot[strategy_name_pretty]["Balanced"] = balanced_result.result

    return results2plot


def main():

    # Load German, UK, & Swedish data, & for each load the three results for each (relevant) strategy
    logger.info("Loading data")
    data_dir = path_util.data_dir()
    preds_and_md_germany_val = PredsAndMetadata.load(data_dir, "germany_val")
    preds_and_md_germany_test = PredsAndMetadata.load(data_dir, "germany_test")
    preds_and_md_sweden_val = PredsAndMetadata.load(data_dir, "sweden_val")
    preds_and_md_sweden_test = PredsAndMetadata.load(data_dir, "sweden_test")
    preds_and_md_uk_val = PredsAndMetadata.load(data_dir, "uk_val")
    preds_and_md_uk_test = PredsAndMetadata.load(data_dir, "uk_test")

    # Load & select the results2plot for each strategy on each dataset

    logger.info("Selecting results to plot for each strategy on each dataset")

    strategy_names = list(PRETTY_STRATEGY_NAMES.keys())
    strategies = [compose_one("strategy", strategy_name, instantiate=True) for strategy_name in strategy_names]
    results2plot_germany = _get_results_by_strategy(preds_and_md_germany_test, preds_and_md_germany_val, strategies)
    results2plot_uk = _get_results_by_strategy(
        preds_and_md_uk_test, preds_and_md_uk_val, strategies, screening_system=gc.ScreeningSystem.UK
    )

    strategy_names_sweden = ["ProgramLevelDD", "NormalTriaging", "StandaloneAI"]
    strategies_sweden = [
        compose_one("strategy", strategy_name, instantiate=True) for strategy_name in strategy_names_sweden
    ]
    results2plot_sweden = _get_results_by_strategy(preds_and_md_sweden_test, preds_and_md_sweden_val, strategies_sweden)

    # Plot the figure !

    logger.info("Plotting !")
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(3, 2, wspace=0.25, hspace=0.25)  # 2x2 grid
    ax0 = fig.add_subplot(gs[0:2, :])  # 1st row, full
    ax1 = fig.add_subplot(gs[2, 0])  # 2nd row, 1st col
    ax2 = fig.add_subplot(gs[2, 1])  # 2nd row, 2nd col

    n_bootstrap = 1000
    plot_figure_2(
        preds_and_md_germany_test,
        ref_result=get_ref_result(preds_and_md_germany_test),
        strategy_results=results2plot_germany,
        ax=ax0,
        title="a. Germany",
        n_bootstrap=n_bootstrap,
        show_legend=True,
    )
    plot_figure_2(
        preds_and_md_uk_test,
        ref_result=get_ref_result(preds_and_md_uk_test),
        strategy_results=results2plot_uk,
        ax=ax1,
        title="b. UK",
        n_bootstrap=n_bootstrap,
        show_legend=False,
    )
    plot_figure_2(
        preds_and_md_sweden_test,
        ref_result=get_ref_result(preds_and_md_sweden_test),
        strategy_results=results2plot_sweden,
        ax=ax2,
        title="c. Sweden",
        n_bootstrap=n_bootstrap,
        show_legend=False,
    )

    # Save the figure after first accounting for the legends that are outside any of the axes
    filename = f"{path_util.analysis_dir()}/figure2"
    bbox_extra_artists = [c for c in fig.get_children() if isinstance(c, Legend)]
    fig.savefig(f"{filename}.png", format="png", dpi=300, bbox_extra_artists=bbox_extra_artists, bbox_inches="tight")
    fig.savefig(f"{filename}.svg", format="svg", dpi=300, bbox_extra_artists=bbox_extra_artists, bbox_inches="tight")


if __name__ == "__main__":
    main()
