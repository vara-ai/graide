import dataclasses
import functools

import numpy as np
import seaborn as sns
from matplotlib.axes import Axes

from graide.artefacts.util import PRETTY_STRATEGY_NAMES
from graide.util.types import Result, ValueWithCI


MatplotlibColor = str | tuple[float, float, float]

# from https://mikemol.github.io/technique/colorblind/2018/02/11/color-safe-palette.html
COLORBLIND_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
COLOR_AI = "#CC79A7"
COLOR_REF = "#222130"

STRATEGY_NAMES = list(PRETTY_STRATEGY_NAMES.values())
STRATEGY_COLOR_MAP = {sname: COLORBLIND_COLORS[i] for i, sname in enumerate(STRATEGY_NAMES)}
assert STRATEGY_COLOR_MAP["Standalone AI"] == COLOR_AI


def seaborn_context(context=None, font_scale: float = 1.0, rc=None):
    """
    Allows us to specify a seaborn context as a decorator.

    Usage:

        @seaborn_context(context="paper", font_scale=1.5)
        def plotting_function(...):

        is equivalent to

        def plotting_function(...):
            with sns.plotting_context(context="paper", font_scale=1.5):
                ...

        but saves indentation levels.
    """

    def decorator_sns_context(plot_func):
        @functools.wraps(plot_func)
        def wrapper_sns_context(*args, **kwargs):
            with sns.plotting_context(context=context, font_scale=font_scale, rc=rc):
                value = plot_func(*args, **kwargs)
            return value

        return wrapper_sns_context

    return decorator_sns_context


def plot_roc_curve(
    recall: np.ndarray,
    cdr: np.ndarray,
    cdr_low: np.ndarray,
    cdr_high: np.ndarray,
    label: str,
    color: MatplotlibColor,
    ax: Axes,
    clip_on: bool = True,
):

    line = ax.plot(recall, cdr, color=color, label=label, linewidth=2, zorder=3)

    # Display CI as a fill if the interval data exists (likely if n_bootstrap > 0 somewhere in the pipeline).
    fill = None
    if cdr_low is not None and cdr_high is not None:
        fill = ax.fill_between(recall, cdr_low, cdr_high, color=color, alpha=0.1)

    # as the roc curves usually touch the frame of the plot, we don't want to clip it
    if not clip_on:
        for plot_item in line:
            plot_item.set_clip_on(False)

    return line, fill


def _format_error_bar(value_with_ci: ValueWithCI | list[ValueWithCI]) -> np.ndarray:
    """Format error bar(s) as required by plt.errorbar"""

    if not isinstance(value_with_ci, list):
        value_with_ci = [value_with_ci]

    lower = [x.value - x.ci_low for x in value_with_ci]
    upper = [x.ci_upp - x.value for x in value_with_ci]
    return np.array([lower, upper])


@dataclasses.dataclass
class Result2Plot:
    """A helper class that encodes everything from a Result that we need to plot an errorbar object."""

    label: str | None
    x: float
    y: float
    xerr: np.ndarray | None
    yerr: np.ndarray | None
    color: MatplotlibColor
    fmt: str
    alpha: float = 1

    @classmethod
    def from_result(
        cls,
        result: Result,
        label: str | None,
        color: MatplotlibColor,
        fmt: str,
        alpha: float = 1,
        with_error_bars: bool = True,
        label_with_metrics: bool = True,
    ) -> "Result2Plot":

        x_value_with_ci = result.metrics.recall * 100
        y_value_with_ci = result.metrics.cdr * 1000

        # Add the metrics to the legend label
        if label is not None and label_with_metrics:
            label = f"{label}: ({y_value_with_ci.value:.3f}, {x_value_with_ci.value:.3f})"

        return cls(
            label=label,
            x=x_value_with_ci.value,
            y=y_value_with_ci.value,
            xerr=_format_error_bar(x_value_with_ci) if with_error_bars else None,
            yerr=_format_error_bar(y_value_with_ci) if with_error_bars else None,
            color=color,
            alpha=alpha,
            fmt=fmt,
        )

    def plot(self, ax: Axes, marker_size: float = 4, elinewidth: float = 0.5):
        ax.errorbar(
            self.x,
            self.y,
            yerr=self.yerr,
            xerr=self.xerr,
            fmt="none",
            color=self.color,
            elinewidth=elinewidth,
            alpha=self.alpha,
        )
        ax.scatter(self.x, self.y, s=marker_size, fc=self.color, marker=self.fmt, alpha=self.alpha, linewidths=0)
