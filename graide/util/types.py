import dataclasses
import numbers
import os
from functools import cached_property
from typing import Any, Callable

import numpy as np
import pandas as pd

from graide import constants as gc
from graide.util import io_util

# Used to indicate that an array contains booleans
BooleanArray = np.ndarray

# A boolean mask representing a 'bucket', i.e. a set of studies that will be dealt with similarly by a given strategy.
Bucket = BooleanArray

# Any number of thresholds (between 0 and 1) used to partition studies into buckets
Thresholds = tuple[float, ...]

# A function to be applied to a set of thresholds to determine whether they're 'valid'
ThresholdConstraint = Callable[[Thresholds], bool]

# An alias for an array of decisions, i.e. a boolean array of either shape (n_studies, ) or (n_studies, 2)
Decisions = BooleanArray

# An alias for a workload array, i.e. an integer array of either shape (n_studies, ) or (n_studies, 2) where each
# entry represents the number of reads required to predict that study following a given strategy.
Workload = np.ndarray


@dataclasses.dataclass(frozen=True)
class ValueWithCI:

    value: float
    ci_low: float | None
    ci_upp: float | None

    def to_dict(self):
        return dataclasses.asdict(self)

    def __mul__(self, factor: int | float) -> "ValueWithCI":
        assert isinstance(factor, numbers.Number), "Your multiplication factor must be a valid number"
        return ValueWithCI(
            value=self.value * factor,
            ci_low=None if self.ci_low is None else self.ci_low * factor,
            ci_upp=None if self.ci_upp is None else self.ci_upp * factor,
        )

    def __str__(self) -> str:
        str_repr = f"{self.value:1%}"
        if self.ci_low is not None and self.ci_upp is not None:
            str_repr += f" ({self.ci_low:.1%}, {self.ci_upp:.1%})"
        return str_repr


@dataclasses.dataclass
class ValueWithPValue:
    value: float
    p_value: float | None

    def __str__(self) -> str:
        p_value = "" if self.p_value is None else f" (p={self.p_value:.3f})"
        return f"{self.value:.1%}{p_value}"


@dataclasses.dataclass
class Delta:
    """
    Stores the deltas, incl. p values, of our 4 core metrics. This object only makes sense in the context of a
    given `StrategyResult` as we need to know the reference we are comparing against.
    """

    sens: ValueWithPValue
    spec: ValueWithPValue
    cdr: ValueWithPValue
    recall: ValueWithPValue

    def get(self, metric_name: str) -> ValueWithPValue:
        return getattr(self, metric_name)


@dataclasses.dataclass
class PermTestResult:
    delta: float
    p_value: float


@dataclasses.dataclass
class ScreeningMetrics:

    sens: ValueWithCI
    spec: ValueWithCI
    cdr: ValueWithCI
    recall: ValueWithCI

    # Optional dictionaries of (positive) metrics stratified by anything
    stratified_sens: dict[Any, ValueWithCI] | None = None
    stratified_cdr: dict[Any, ValueWithCI] | None = None

    def get(self, metric_name: str) -> ValueWithCI:
        return getattr(self, metric_name)

    def get_value(self, metric_name: str) -> float:
        return self.get(metric_name).value

    def __sub__(self, other: "ScreeningMetrics") -> "Delta":
        return Delta(
            **{
                metric_name: ValueWithPValue(
                    value=self.get_value(metric_name) - other.get_value(metric_name), p_value=None
                )
                for metric_name in gc.METRIC_NAMES
            }
        )


@dataclasses.dataclass
class ReferenceResult:
    """
    An analogue to the below `StrategyResult` that stores the simpler information needed for a reference. The first
    two fields are the only two that need to be set; the others are included only for compatibility with the
    `StrategyResult` type below.
    """

    decision_level: gc.DecisionLevel
    metrics: ScreeningMetrics

    strategy_name: str = "Reference"
    workload_reduction: float = 0
    thresholds = None
    delta = None

    def __str__(self) -> str:
        return (
            f"{self.strategy_name} ({self.decision_level.value})"
            f"cdr={self.metrics.cdr.value * 1000:.2f}/1000 "
            f"recall={self.metrics.recall.value * 100:.1f}/100 "
        )

    def to_dict(self) -> dict[str, str]:
        """See `StrategyResult.to_dict` for more info; this function is more self-explanatory."""

        _dict = {"strategy": f"{self.strategy_name} ({self.decision_level.name})", "thresholds": None}
        for m_name in gc.METRIC_NAMES:
            _dict[f"{m_name} (95% CI)"] = str(self.metrics.get(m_name))
            _dict[f"Δ {m_name} (p-value)"] = None
        _dict["workload reduction"] = f"{self.workload_reduction:+.0%}"
        return _dict


@dataclasses.dataclass
class StrategyResult:
    """
    Stores the screening metrics we care about for a given strategy & set of AI thresholds.
    This object only makes sense in the context of a given `ScreeningResults` object as we need to know the reference
    we're working with and comparing against.
    """

    # The name of the `ScreeningStrategy` subclass used to compute this result (see `strategy.py`).
    strategy_name: str

    # The specific thresholds used by the given screening strategy, to achieve the given results.
    thresholds: Thresholds

    # The set of metrics to store
    metrics: ScreeningMetrics

    # The workload reduction, i.e. how many reads were saved by using this strategy rather than the reference, as a
    # fraction (see `ScreeningResults` below for more info).
    workload_reduction: float

    # The delta between this result and the reference's `ReferenceResult`. This field is optional as we'll often
    # only compute the delta for a (promising) subset of strategy results in a given `ScreeningResults` object.
    delta: Delta | None = None

    def __str__(self) -> str:

        thresholds_str = ""
        if self.thresholds is not None:
            thresholds_str = "thresholds=" + "(" + ", ".join([f"{t:.2f}" for t in self.thresholds]) + ") "

        return (
            f"{self.strategy_name} {thresholds_str}"
            f"cdr={self.metrics.cdr.value * 1000:.2f}/1000 "
            f"recall={self.metrics.recall.value * 100:.1f}/100 "
            f"workload_reduction={self.workload_reduction:.0%}"
        )

    def to_dict(self) -> dict[str, str]:
        """
        Converts this StrategyResult object into a flattened dict representing the key information, namely an
        identifier (strategy & thresholds) and the value of each metric.
        """

        _dict = {"strategy": self.strategy_name, "thresholds": self.thresholds}

        for m_name in gc.METRIC_NAMES:
            metric = self.metrics.get(m_name)

            # Add entry for the metric value, possibly with CIs
            _dict[f"{m_name} (95% CI)"] = str(metric)

            # Add entry for the metric delta, possibly with delta
            _dict[f"Δ {m_name} (p-value)"] = str(self.delta.get(m_name)) if self.delta is not None else None

        # Add entry for the workload reduction
        _dict["workload reduction"] = f"{self.workload_reduction:+.0%}"

        return _dict

    def is_super_reference(
        self,
        reference_result: ReferenceResult | None = None,
        use_cdr_and_recall: bool = True,
        require_significance: bool = False,
    ) -> bool:
        """
        A small helper to encode in one place the way we determine whether a given result is super-reference. Care must
        be taken when using this function that the passed reference result was evaluated at the same output decision
        level as this strategy result.

        Args:
            reference_result:
                The reference result with which to compare, only required if this strategy result doesn't have a delta.
                IMPORTANT: if the result has a delta then the passed `reference_result` will be ignored.
            use_cdr_and_recall:
                A boolean indicating whether to use cdr / recall or sens / spec in determining super-referenceness.
            require_significance:
                A boolean indicating whether or not we require statistical significance in our determination of
                super-reference-ness.

        Returns: a boolean indicating whether this result is super-reference.
        """

        metric_pos = gc.CDR if use_cdr_and_recall else gc.SENS
        metric_neg = gc.RECALL if use_cdr_and_recall else gc.SPEC

        if self.delta is None:

            # If we require significance but we don't have a delta, we simply exit with False. We don't want to start
            # computing a delta here else the `ScreeningResults.super_reference` function below could easily trigger
            # delta computation for all strategy results.
            # The only likely use-case for `require_significance=True` is when calling this method on selected
            # test set results.
            if require_significance:
                return False

            # If we have no delta we determine super-referenceness along each axis by just checking the absolute deltas
            if reference_result is None:
                raise ValueError(
                    "Since this `StrategyResult` doesn't have a delta, in order to call is_super_reference you must "
                    "pass a reference_result against which one can be computed."
                )

            pos_is_super = self.metrics.get_value(metric_pos) >= reference_result.metrics.get_value(metric_pos)

            neg_value = self.metrics.get_value(metric_neg)
            neg_value_ref = reference_result.metrics.get_value(metric_neg)
            neg_is_super = neg_value >= neg_value_ref

            # If we're using cdr / recall then we want our recall to be lower than the refernece, => we flip the > to <
            if use_cdr_and_recall:
                neg_is_super = neg_value <= neg_value_ref

        # If we have a delta then we first check the absolute deltas, and only if we require significance do we also
        # enforce sufficiently small p values.
        else:
            d_pos, d_neg = self.delta.get(metric_pos), self.delta.get(metric_neg)
            pos_is_super = d_pos.value > 0
            neg_is_super = d_neg.value < 0 if use_cdr_and_recall else d_neg.value > 0

            if require_significance:
                if d_pos.p_value is None or d_neg.p_value is None:
                    raise ValueError(
                        "In order to evaluate super-referenceness with significance, we need the delta to have "
                        "non-None p values."
                    )

                pos_is_super &= d_pos.p_value < 0.05
                neg_is_super &= d_neg.p_value < 0.05

        return pos_is_super and neg_is_super

    @classmethod
    def load(cls, output_dir: str, file_name: str) -> "StrategyResult":
        return io_util.load_from_pickle(os.path.join(output_dir, file_name))

    def save(self, output_dir: str, file_name: str):
        io_util.save_to_pickle(self, os.path.join(output_dir, file_name))


# These two types have the same fields; they're defined separately to skirt annoying dataclass inheritance
Result = ReferenceResult | StrategyResult


@dataclasses.dataclass
class ScreeningResults:

    FILENAME = "results.pklz"

    # The reference level to compare against. This will determine both the `ReferenceResult` that is used as a baseline
    # comparison point (e.g. to compute deltas), as well as how the metrics are computed for each strategy result, i.e.
    # by specifying how far to "push" the decisions through the screening program before analysing the results.
    decision_level: gc.DecisionLevel
    reference_result: ReferenceResult

    # A list of results using different thresholds
    strategy_results: list[StrategyResult]

    screening_system: gc.ScreeningSystem = gc.ScreeningSystem.GERMANY

    def __len__(self):
        return len(self.strategy_results)

    def __iter__(self):
        return iter(self.strategy_results)

    def save(self, output_dir: str, alt_filename: str | None = None):
        io_util.save_to_pickle(self, os.path.join(output_dir, alt_filename or self.FILENAME))

    @classmethod
    def load(cls, output_dir: str, alt_filename: str | None = None) -> "ScreeningResults":
        return io_util.load_from_pickle(os.path.join(output_dir, cls.FILENAME))

    def to_df(self) -> pd.DataFrame:
        """Converts the `ScreeningResults` to a dataframe of flattened reference & strategy results."""

        return pd.DataFrame([self.reference_result.to_dict()] + [r.to_dict() for r in self.strategy_results])

    def to_csv(self, output_dir: str):
        """Saves the DF created by the above function as a CSV."""

        io_util.save_df_as_csv(self.to_df(), os.path.join(output_dir, "results.csv"))

    @cached_property
    def strategy_result_by_threshold(self) -> dict[Thresholds, StrategyResult]:
        return {r.thresholds: r for r in self.strategy_results}

    def get_result_by_thresholds(self, thresholds: Thresholds) -> StrategyResult:
        return self.strategy_result_by_threshold[thresholds]

    def override_results_by_thresholds(self, new_results: list[StrategyResult]) -> "ScreeningResults":
        """Overrides any number of strategy results based on their thresholds, leaving all others as they were."""

        new_results_by_thresh: dict[Thresholds, StrategyResult] = {r.thresholds: r for r in new_results}
        return self._override_strategy_results(
            [new_results_by_thresh.get(result.thresholds, result) for result in self.strategy_results]
        )

    def _override_strategy_results(self, new_strategy_results: list[StrategyResult]) -> "ScreeningResults":
        """Returns a copy of the ScreeningResults with the strategy results replaced by those in the new list."""

        return ScreeningResults(
            decision_level=self.decision_level,
            reference_result=self.reference_result,
            strategy_results=new_strategy_results,
        )

    def sort(self, key: Callable[[StrategyResult], Any]) -> "ScreeningResults":
        """Returns a copy of the screening results with the strategy results sorted according to the given key."""

        return self._override_strategy_results(sorted(self.strategy_results, key=key))

    def filter(self, condition: Callable[[StrategyResult], bool]) -> "ScreeningResults":
        """Returns a copy of the screening results containing only those strategy results that satisfy the condition."""

        return self._override_strategy_results([r for r in self.strategy_results if condition(r)])

    def map(self, mapper: Callable[[StrategyResult], StrategyResult]) -> "ScreeningResults":
        """Returns a copy of the screening results with some mapping function applied to each strategy result."""

        return self._override_strategy_results([mapper(r) for r in self.strategy_results])

    def get_result(self, condition: Callable[[StrategyResult], bool]) -> StrategyResult:
        """Returns the first strategy result satisfying the given condition."""

        return next(r for r in self.strategy_results if condition(r))

    def super_reference(
        self,
        use_cdr_and_recall: bool = False,
        require_significance: bool = False,
    ) -> "ScreeningResults":
        """
        Filters this ScreeningResults object to include only those StrategyResults entities that are super-reference.

        Args:
            use_cdr_and_recall:
                A boolean indicating whether to use cdr / recall or sens / spec in determining super-referenceness.
            require_significance:
                A boolean indicating whether or not we require statistical significance when determining
                super-referenceness.

        Returns: another `ScreeningResults` instance, probably with fewer `StrategyResults`, but otherwise unchanged.
        """

        return self._override_strategy_results(
            [
                r
                for r in self.strategy_results
                if r.is_super_reference(
                    reference_result=self.reference_result,
                    use_cdr_and_recall=use_cdr_and_recall,
                    require_significance=require_significance,
                )
            ]
        )

    def optimal_result(
        self,
        key: Callable[[StrategyResult], Any],
        use_cdr_and_recall: bool = True,
        only_super: bool = True,
    ) -> StrategyResult:
        """
        A helper function that selects the 'optimal' result according to some key that sorts the list. The `only_super`
        flag can be set to False to sort the entire list of results, not only those that are super-reference.
        """

        to_sort = self.super_reference(use_cdr_and_recall=use_cdr_and_recall) if only_super else self
        return to_sort.sort(key).strategy_results[0]

    @property
    def cdr_optimising_result(self) -> StrategyResult:
        """
        Get the (non-statistically-significant) super-reference result that maximises CDR. If there are no
        super-reference results then we resort to the logic described by `best_non_super_results`.
        """

        try:
            return self.optimal_result(lambda r: -r.metrics.cdr.value)
        except IndexError:
            return self.best_non_super_results[0]

    @property
    def recall_optimising_result(self) -> StrategyResult:
        """
        Get the (non-statistically-significant) super-reference result that minimises recall. If there are no
        super-reference results then we resort to the logic described by `best_non_super_results`.
        """

        try:
            return self.optimal_result(lambda r: r.metrics.recall.value)
        except IndexError:
            return self.best_non_super_results[1]

    def oracle_result(
        self,
        ref_result: ReferenceResult | None = None,
        use_cdr_and_recall: bool = True,
    ) -> StrategyResult:
        """
        Selects a result using the following three steps:

            1. If there are any super-reference results, we only search within those, else we search within all results
            2. We filter to only those results that are within some distance margin of the 'oracle' line, i.e. the line
               from the reference performance to (sens, spec) = (1, 1). We use sens / spec instead of CDR / recall to
               avoid the difficulties that come with the latter two rates being different orders of magnitude.
            3. Of those, we seect the one that minimises the distance to the top-left, (sens, spec) = (1, 1).

        The idea behind these steps is to show the result that has the largest delta to the reference but is roughly
        along the oracle line, i.e. the result that constitutes a balanced increase in metrics. We can't simply take
        the point that minimises the distance to the oracle line as there are sometimes much better points that are
        just slightly further away.

        Args:
            ref_result:
                You can optionally pass a reference result with which the oracle line will be defined. This is useful
                as the Rad. and Program references sit at very different points in the sens / spec tradeoff, so we may
                want different oracle lines in different settings.
            use_cdr_and_recall:
                A boolean indicating whether to use cdr / recall or sens / spec in determining super-referenceness.
        """

        # Reference p1 = (x1, y1) and perfect performance p2 = (x2, y2) = (1, 1) define the line from reference
        # performance to perfect performance (the 'oracle' line).
        ref_result = ref_result or self.reference_result
        y1, x1 = ref_result.metrics.sens.value, ref_result.metrics.spec.value
        y2, x2 = 1.0, 1.0
        distance_p1_p2 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # If there's at least one super-reference result then we only consider those, else we consider any points.
        self_super = self.super_reference(use_cdr_and_recall=use_cdr_and_recall)
        to_filter = self if len(self_super) == 0 else self_super

        # Filter to only those results that are within 1e-4 of the oracle line (empirically determined on validation
        # data, see section eMethods 5 in the Appendix for more info).
        for margin in [1e-4, 1e-3, 1e-2]:
            close_to_oracle_results = to_filter.filter(
                lambda r: (
                    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
                    np.abs((x2 - x1) * (y1 - r.metrics.sens.value) - (x1 - r.metrics.spec.value) * (y2 - y1))
                    / distance_p1_p2
                )
                < margin
            )
            if len(close_to_oracle_results) > 0:
                break

        # ... and within those, choose the one that minimises the distance to the oracle.
        return close_to_oracle_results.optimal_result(
            lambda r: np.sqrt((x2 - r.metrics.spec.value) ** 2 + (y2 - r.metrics.sens.value) ** 2),
            use_cdr_and_recall=use_cdr_and_recall,
            only_super=False,
        )

    @property
    def best_non_super_results(self) -> tuple[StrategyResult, StrategyResult]:
        """
        A simple heuristic function for finding the 'optimal' CDR & recall rate results on a dataset where we have no
        super-reference results. All we do is sort the results by decreasing CDR (or increasing recall) and then find
        the point at which the metric in question changes from better to worse as compared to the program. By selecting
        exactly the result where we are _just_ better than the program we're finding out how much that increase will
        cost us in terms of recall (or CDR). Put simply, the two results returned will be on either side of
        super-referenceness, thereby giving us a clear idea of how far away we are.
        """

        cdr_result, recall_result = None, None

        results_sorted_cdr = self.sort(lambda r: -r.metrics.cdr.value).strategy_results
        for i, r in enumerate(results_sorted_cdr[:-1]):
            r_next = results_sorted_cdr[i + 1]
            if r.metrics.cdr.value - self.reference_result.metrics.cdr.value > 0:  # good
                if r_next.metrics.cdr.value - self.reference_result.metrics.cdr.value <= 0:  # bad
                    cdr_result = r

        results_sorted_recall = self.sort(lambda r: r.metrics.recall.value).strategy_results
        for i, r in enumerate(results_sorted_recall[:-1]):
            r_next = results_sorted_recall[i + 1]
            if r.metrics.recall.value - self.reference_result.metrics.recall.value < 0:  # good
                if r_next.metrics.recall.value - self.reference_result.metrics.recall.value >= 0:  # bad
                    recall_result = r

        assert (
            cdr_result is not None or recall_result is not None
        ), "Neither a CDR nor a recall-optimising result was found, a situation we haven't seen before."

        # If we didn't find a result here it means that we're working with a 'one-sided' strategy, i.e. one that's only
        # capable of improving one of the two metrics over the reference. If the CDR result is None it means there was
        # no result with a better CDR than the reference, => we select the result with the highest CDR (likely equal to
        # reference performance) as the CDR-optimising result and use our one-sided heuristic to find the
        # recall-optimising result. And VV for the recall-result.
        if cdr_result is None:
            cdr_result = results_sorted_cdr[0]
            recall_result = self._best_one_sided_result("recall")
        if recall_result is None:
            recall_result = results_sorted_recall[0]
            cdr_result = self._best_one_sided_result("cdr")

        return cdr_result, recall_result

    def _best_one_sided_result(self, metric_name: str) -> StrategyResult:
        """
        A heuristic function used to find the 'optimal' CDR or recall result for a 'one-sided' strategy, a strategy that
        is only capable of improving one of the two metrics over the reference. For example, `NormalTriaging` is
        capable of decreasing recall but it can never increase CDR.

        The heuristic we use here is to allow a small margin in one of the two metrics. For example, if we're selecting
        the recall-optimising result on a set of results, none of whom have a higher _CDR_ than the reference, we allow
        a 1% margin on CDR, filter the results to those who are 'super' after this margin has been applied, and then
        select the recall-optimising result among those remaining.
        """

        assert metric_name in {"cdr", "recall"}, "The metric_name passed must be in {'cdr', 'recall'}"

        # Compute the reference metrics against which to compare, one of which will have had a 1% margin applied ...
        ref_cdr, ref_recall = self.reference_result.metrics.cdr.value, self.reference_result.metrics.recall.value
        if metric_name == "cdr":
            ref_recall *= 1.01
        if metric_name == "recall":
            ref_cdr *= 0.99

        # ... & then filter the results to only those that are "super" now that we've applied the margin.
        supers = [
            r for r in self.strategy_results if r.metrics.cdr.value > ref_cdr and r.metrics.recall.value < ref_recall
        ]

        # Finally, we just return the highest CDR / lowest recall result of those left in `supers`.
        if metric_name == "cdr":
            return sorted(supers, key=lambda r: -r.metrics.cdr.value)[0]
        else:
            return sorted(supers, key=lambda r: r.metrics.recall.value)[0]


@dataclasses.dataclass
class PredsAndMetadata:
    """
    An object that stores metadata values and AI model scores (between 0 and 1) for each study in a dataset.
    All values are arrays, and all should be the same length.
    """

    # The name of the dataset this entity represents, derived from the file it's loaded from.
    dataset_name: str

    # Ground-truth labels & decision arrays at the various decision levels.
    y_true: BooleanArray
    y_rad: BooleanArray
    y_send2cc: BooleanArray
    y_program: BooleanArray

    # Cancer sub-type masks
    is_sdc: BooleanArray
    is_ic: BooleanArray
    is_nrsdc: BooleanArray

    # AI model scores
    y_score_bal: np.ndarray
    y_score_spec: np.ndarray

    # Per-study weights
    weights: np.ndarray

    # Additional metadata arrays
    cancer_grade: np.ndarray
    cancer_stage: np.ndarray
    patient_age: np.ndarray

    @property
    def is_cancer(self) -> BooleanArray:
        return self.is_sdc | self.is_ic | self.is_nrsdc

    @property
    def is_missed_cancer(self) -> BooleanArray:
        return self.is_ic | self.is_nrsdc

    def __len__(self) -> int:
        return len(self.y_true)

    @classmethod
    def load(cls, dir_path: str, dataset_name: str) -> "PredsAndMetadata":
        """Loads a PredsAndMetadata object from a CSV file."""

        csv_path = os.path.join(dir_path, f"{dataset_name}.csv")
        df = pd.read_csv(csv_path)

        # Determine class labels
        is_sdc = np.array(df["is_sdc"], dtype=bool)
        is_ic = np.array(df["is_ic"], dtype=bool)
        is_nrsdc = np.array(df["is_nrsdc"], dtype=bool)

        return cls(
            dataset_name=dataset_name,
            y_true=np.array(df["y_true"], dtype=bool),
            y_rad=np.c_[np.array(df["y_rad_1"], dtype=bool), np.array(df["y_rad_2"], dtype=bool)],
            y_send2cc=np.array(df["y_send2cc"], dtype=bool),
            y_program=np.array(df["y_program"], dtype=bool),
            is_sdc=is_sdc,
            is_ic=is_ic,
            is_nrsdc=is_nrsdc,
            y_score_bal=np.array(df["y_score_bal"], dtype=float),
            y_score_spec=np.array(df["y_score_spec"], dtype=float),
            weights=np.array(df["weights"], dtype=float),
            cancer_grade=np.array(df["cancer_grade"], dtype=object),
            cancer_stage=np.array(df["cancer_stage"], dtype=object),
            patient_age=np.array(df["age"], dtype=int),
        )
