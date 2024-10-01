"""
See `ScreeningStrategy` below.
"""
import numpy as np

from graide import constants as gc, screening_path
from graide.stats import compute_metrics, compute_stratified_cdr, compute_stratified_sens
from graide.util import screening_util
from graide.util.types import (
    BooleanArray,
    Bucket,
    Decisions,
    PredsAndMetadata,
    ReferenceResult,
    ScreeningMetrics,
    StrategyResult,
    ThresholdConstraint,
    Thresholds,
    Workload,
)

RAD, SEND2CC, PROGRAM = gc.DecisionLevel.RAD, gc.DecisionLevel.SEND2CC, gc.DecisionLevel.PROGRAM
ScreeningPath = screening_path.ScreeningPath


def _buckets_are_valid(buckets: tuple[Bucket, ...], num_studies: int) -> bool:
    """Validates that every study is placed in exactly one bucket."""

    assert all(
        len(b) == num_studies for b in buckets
    ), "len(bucket) != num_studies... Please double-check your `buckets_from_thresholds` implementation"

    # validate that each study is in no more than one bucket and at least one bucket
    return (max(np.sum(buckets, axis=0)) == 1) and (np.sum(buckets) == num_studies)


class ScreeningStrategy:
    """
    This class supports the definition of arbitrary AI-integration strategies for (German or other) screening systems.

    Each strategy must define at least the three core properties — `input_decision_level`, `min_output_decision_level`,
    & `n_buckets` — and the following methods: `buckets_from_thresholds`, `screening_paths`, `positive_ai_buckets`, &
    `ai_buckets`.
    """

    # The number of buckets that this strategy operates on, i.e. the number of different ways that studies are dealt
    # with to arrive at a decision
    n_buckets: int

    # The interval used to create this strategy's threshold search space (see `_thresholds_iter` in `run_screening.py`).
    threshold_interval_size: float = 0.01

    def init_data(self, preds_and_md: PredsAndMetadata):
        """
        These operations are in a separate constructor s.t. we can instantiate the strategy classes & inspect their
        other attributes (e.g. `.name` and `.has_decision_dimension`) without requiring a preds_and_md instance.
        """
        self.preds_and_md = preds_and_md
        self.y_true = preds_and_md.y_true
        self.y_score_bal = preds_and_md.y_score_bal
        self.y_score_spec = preds_and_md.y_score_spec

    @property
    def n_thresholds(self) -> int:
        return self.n_buckets - 1

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def min_output_decision_level(self) -> gc.DecisionLevel:
        """Returns the minimum decision level at which this strategy can operate, below which it can't be evaluated."""
        raise NotImplementedError

    @property
    def input_decision_level(self) -> gc.DecisionLevel | tuple[gc.DecisionLevel, ...]:
        """
        Returns the decision-level at which this strategy operates, either one level for all buckets or a tuple of
        per-bucket levels. You can think about this as the level at which the AI makes a decision in (a given bucket
        within) this strategy, i.e. whether it replaces one reader, the two readers together (send2cc), or the program.
        These values are used to control the "pushing" of decisions, i.e. the process by which decisions at an earlier
        level in the screening system are transferred to later level decisions.
        """
        raise NotImplementedError

    @property
    def input_decision_levels(self) -> tuple[gc.DecisionLevel, ...]:
        """Convenience function to convert the above to a tuple format. This should not be overridden."""
        return (
            self.input_decision_level
            if isinstance(self.input_decision_level, tuple)
            else tuple([self.input_decision_level for _ in range(self.n_buckets)])
        )

    @property
    def has_decision_dimension(self) -> bool:
        """
        A "decision dimension" is a second dimension in the decisions & workload arrays that corresponds to multiple
        sets of decisions. This is only required when any bucket in the strategy operates on a rad-level because,
        in that case, we need to simulate the impact of working with each of the two radiologists separately.
        """
        return RAD in self.input_decision_levels

    def threshold_constraints(self) -> list[ThresholdConstraint]:
        """
        An optional function that returns a list of threshold constraints: functions that take the AI thresholds
        needed by this strategy and return a bool indicating whether that tuple of thresholds is 'valid'. So far we
        only use these constraints in order to restrict the threshold search space.
        """
        return []

    def buckets_from_thresholds(self, thresholds: Thresholds) -> tuple[Bucket, ...]:
        """
        Given a set of thresholds (one fewer than the desired number of buckets), this function returns a tuple of
        buckets, each mask indicating the studies that belong in that bucket. A valid set of buckets must place every
        study into _exactly_ one bucket, though an invalid set of buckets can be caused not only by an incorrect
        implementation but also by a non-partitioning set of thresholds.

        There are two reasons why this function should be maintained separately from `decisions_from_buckets`:
            1) in order to allow the calling process to handle the validation of the buckets, and
            2) to allow for inheritance of one or other of these functions between various strategy implementations.
        """
        raise NotImplementedError

    def screening_paths(self) -> list[ScreeningPath]:
        """
        Returns a list of ScreeningPath functions, one for each bucket returned by the above, that each map a single
        bucket to the boolean decisions & workload arrays of the studies in that bucket at a given decision level.
        """

        raise NotImplementedError

    def positive_ai_buckets(self, buckets: tuple[Bucket, ...]) -> list[Bucket]:
        """Returns a list of the buckets that constitute a positive AI decision, e.g. for localization analysis."""
        raise NotImplementedError

    def ai_buckets(self, buckets: tuple[Bucket, ...]) -> list[Bucket]:
        """Returns a list of the buckets that involve an AI decision, at any decision level."""
        raise NotImplementedError

    def decisions_from_buckets(
        self,
        buckets: tuple[Bucket, ...],
        output_decision_level: gc.DecisionLevel,
        screening_system: gc.ScreeningSystem = gc.ScreeningSystem.GERMANY,
    ) -> tuple[Decisions, Workload]:
        """
        Given a set of buckets, this strategy's screening paths, & the decision-level of the decisions that should be
        output, this function returns the boolean decisions array at that decision level as well as an integer array
        representing the amount of 'work' (in units of reads) done to reach those decisions.
        This function should not be overridden by the strategy subclasses.
        """

        has_decision_dimension = output_decision_level is RAD or (
            self.has_decision_dimension and screening_system is gc.ScreeningSystem.GERMANY
        )

        # We start with empty arrays for both decisions & workload, stacked into an array of shape (n_studies, 2) if
        # any of our buckets is operating on a rad-level.
        decisions = np.zeros_like(self.y_true, dtype=bool)
        workload = np.zeros_like(self.y_true, dtype=int)
        if has_decision_dimension:
            decisions = np.c_[decisions, decisions]
            workload = np.c_[workload, workload]

        # For each bucket & screening path defined by this strategy class ...
        for bucket_idx, _screening_path in enumerate(self.screening_paths()):
            bucket = buckets[bucket_idx]

            # ... get the decisions & workload arrays for the bucket, pushed through to the target decision-level ...
            bucket_decisions, bucket_workload = _screening_path(
                bucket=bucket,
                preds_and_md=self.preds_and_md,
                input_decision_level=self.input_decision_levels[bucket_idx],
                output_decision_level=output_decision_level,
                screening_system=screening_system,
            )

            if has_decision_dimension and bucket_decisions.ndim == 1:
                bucket_decisions = np.c_[bucket_decisions, bucket_decisions]
            if has_decision_dimension and bucket_workload.ndim == 1:
                bucket_workload = np.c_[bucket_workload, bucket_workload]

            # ... & assign the values.
            decisions[bucket] = bucket_decisions
            workload[bucket] = bucket_workload

        return decisions, workload

    def compute_result_at_thresholds(
        self,
        thresholds: Thresholds,
        decision_level: gc.DecisionLevel,
        reference_result: ReferenceResult | None = None,
        reference_decisions: Decisions | None = None,
        program_workload: float | None = None,
        metric_stratifications: dict[str, BooleanArray] | None = None,
        with_stats: bool = False,
        screening_system: gc.ScreeningSystem = gc.ScreeningSystem.GERMANY,
    ) -> StrategyResult | None:
        """
        This function is the entire reason behind the `ScreeningStrategy` class. For a given strategy, for a given
        set of thresholds, this function computes the screening metrics we care about.

        This function should not be overridden by the strategy subclasses.

        Args:
            thresholds: the set of thresholds at which to evaluate the current strategy
            decision_level: the decision level at which to evaluate the result
            reference_result: (optional) the performance of the reference
            reference_decisions: (optional) the decisions of the reference at the specified decision_level
            program_workload: (optional) the total program workload, used to compute workload reduction
            metric_stratifications: a (optional) dict of named masks determining stratified metric computation
            with_stats: whether or not to compute CIs for the metrics
            screening_system: the screening system

        Returns: a StrategyResult, or None if the provided thresholds don't partition the dataset
        """

        if not hasattr(self, "preds_and_md"):
            raise ValueError(
                "Please call `init_data()` on your ScreeningStrategy before running `compute_result_at_thresholds`"
            )

        preds_and_md = self.preds_and_md

        reference_result = reference_result or screening_util.get_reference_result(preds_and_md, decision_level)
        program_workload = program_workload or screening_util.get_program_workload(preds_and_md, decision_level)

        if reference_decisions is None:
            reference_decisions = screening_util.get_reference_decisions(preds_and_md, decision_level)

        # Determine the strategy's buckets & continue iff each study is placed in exactly one bucket
        buckets = self.buckets_from_thresholds(thresholds)
        if not _buckets_are_valid(buckets, num_studies=len(preds_and_md)):
            return None

        # Get decisions & workload of the given strategy at the given thresholds at the given output decision-level
        # in the given screening system.
        decisions, workload = self.decisions_from_buckets(buckets, decision_level, screening_system)

        # Compute fractional workload reduction
        workload_reduction = screening_util.compute_workload_reduction(workload, program_workload, preds_and_md.weights)

        # Compute metrics, with CIs if with_stats=True
        sens_with_ci, spec_with_ci, cdr_with_ci, recall_with_ci = compute_metrics(
            preds_and_md.y_true, decisions, preds_and_md.weights, n_resamples=1000 if with_stats else 0
        )
        metrics = ScreeningMetrics(sens_with_ci, spec_with_ci, cdr_with_ci, recall_with_ci)

        # Here we (maybe) add per-cancer-type sens & CDR values s.t. we can later stratify the delta improvement of
        # a given strategy over the different kinds of cancers.
        if metric_stratifications is not None:
            metrics.stratified_sens = compute_stratified_sens(
                metric_stratifications, preds_and_md.y_true, decisions, preds_and_md.weights
            )
            metrics.stratified_cdr = compute_stratified_cdr(
                metric_stratifications, preds_and_md.y_true, decisions, preds_and_md.weights
            )

        # Compute a delta (maybe) including p values for the difference between the strategy result & the reference
        delta = (
            None
            if not with_stats
            else screening_util.compute_delta(
                metrics, reference_result.metrics, decisions, reference_decisions, preds_and_md
            )
        )

        return StrategyResult(
            strategy_name=self.name,
            thresholds=thresholds,
            metrics=metrics,
            workload_reduction=workload_reduction,
            delta=delta,
        )


class DecisionDeferral(ScreeningStrategy):
    """
    The decision deferral strategies are those in which the AI makes some positive and some negative decisions,
    deferring the rest to the screening system. The deferral_level passed determines the level at which the deferral
    operates. See `screening_paths` below for a clear in-code explanation of this point.
    """

    n_buckets: int = 3

    def __init__(self, deferral_level: gc.DecisionLevel):
        super().__init__()
        self.deferral_level = deferral_level

    @property
    def name(self) -> str:
        return f"{self.deferral_level.name.lower().capitalize()}LevelDD"

    @property
    def input_decision_level(self) -> gc.DecisionLevel:
        """Rad-level DD makes decisions on a radiologist level, program-level DD on a program level, etc."""
        return self.deferral_level

    @property
    def min_output_decision_level(self) -> gc.DecisionLevel:
        return self.input_decision_level

    def threshold_constraints(self) -> list[ThresholdConstraint]:
        """
        Constrains our balanced AI model threshold to be less than 0.25 and our high-specificity AI model threshold
        to be less than 0.75.
        """
        return [lambda thresholds: thresholds[0] <= 0.25 and thresholds[1] <= 0.75]

    def buckets_from_thresholds(self, thresholds: tuple[float, float]) -> tuple[Bucket, Bucket, Bucket]:
        """
        These same three buckets will be used for all DD strategies, only the process by which the buckets
        are used to generate decisions will change.
        """

        neg_bucket = self.y_score_bal < thresholds[0]
        dd_bucket = (self.y_score_bal >= thresholds[0]) & (self.y_score_spec <= thresholds[1])
        pos_bucket = self.y_score_spec > thresholds[1]

        return neg_bucket, dd_bucket, pos_bucket

    def positive_ai_buckets(self, buckets: tuple[Bucket, Bucket, Bucket]) -> list[Bucket]:
        return [buckets[-1]]

    def ai_buckets(self, buckets: tuple[Bucket, Bucket, Bucket]) -> list[Bucket]:
        return [buckets[0], buckets[-1]]

    def screening_paths(self) -> list[screening_path.ScreeningPath]:
        """
        In the deferral regime we send to both readers (i.e. to the regular program), while in the negative and
        positive buckets we need to do slightly different things depending on the deferral_level of the DD we  are
        running. If we're on rad-level then we either send to the other reader or to CC, if we're on send2cc-level
        then we send either to negative or to CC, & if we're on program-level then we send to either negative or recall.
        """

        def_path = screening_path.screening_system_2_readers

        if self.deferral_level is RAD:
            neg_path = screening_path.negative_ai_read
            pos_path = screening_path.suspicious_ai_read
        elif self.deferral_level is PROGRAM:
            neg_path = screening_path.direct_no_recall
            pos_path = screening_path.direct_recall

        return [neg_path, def_path, pos_path]


class StandaloneAI(ScreeningStrategy):
    """In this strategy the AI makes all the decisions, with no involvement from the screening system."""

    n_buckets: int = 2

    @property
    def input_decision_level(self) -> gc.DecisionLevel:
        """
        The input decision level doesn't actually matter here since we only use the `direct_no_recall` and
        `direct_recall` bucket functions, neither of which require any "pushing" through the program.
        """
        return PROGRAM

    @property
    def min_output_decision_level(self) -> gc.DecisionLevel:
        return RAD

    def buckets_from_thresholds(self, thresholds: tuple[float]) -> tuple[Bucket, Bucket]:
        ai_threshold = thresholds[0]
        bucket_1 = self.y_score_bal < ai_threshold
        bucket_2 = self.y_score_bal >= ai_threshold
        return bucket_1, bucket_2

    def screening_paths(self) -> list[screening_path.ScreeningPath]:
        return [
            screening_path.direct_no_recall,
            screening_path.direct_recall,
        ]

    def positive_ai_buckets(self, buckets: tuple[Bucket, Bucket]) -> list[Bucket]:
        return [buckets[-1]]

    def ai_buckets(self, buckets: tuple[Bucket, Bucket]) -> list[Bucket]:
        return list(buckets)


class ReaderReplacement(StandaloneAI):
    """
    This strategy represents the replacement of one of two readers with an AI. The AI makes positive and negative
    decisions, but those have to be combined with the decisions of the other reader to evaluate send2cc- or program-
    level results.
    """

    @property
    def input_decision_level(self) -> gc.DecisionLevel:
        return RAD

    @property
    def min_output_decision_level(self) -> gc.DecisionLevel:
        """Evaluating reader-replacement at the rad-level would be equivalent to evaluating standalone AI."""
        return SEND2CC

    def screening_paths(self) -> list[screening_path.ScreeningPath]:
        return [
            screening_path.negative_ai_read,
            screening_path.suspicious_ai_read,
        ]


class NormalTriaging(StandaloneAI):
    """This strategy makes some confident negative decisions and refers the rest to the screening system."""

    def screening_paths(self) -> list[screening_path.ScreeningPath]:
        return [
            screening_path.direct_no_recall,
            screening_path.screening_system_2_readers,
        ]

    def positive_ai_buckets(self, buckets: tuple[Bucket, Bucket]) -> list[Bucket]:
        return []

    def ai_buckets(self, buckets: tuple[Bucket, Bucket]) -> list[Bucket]:
        return [buckets[0]]


class Graide(ScreeningStrategy):
    """
    Generalised AI Deferral is the more general form of `DecisionDeferral`, allowing assignment to all five screening
    paths. This strategy can only be evaluated on the program-level as it operates at multiple levels in the screening
    system.
    """

    n_buckets: int = 5

    @property
    def input_decision_level(self) -> tuple[gc.DecisionLevel, ...]:
        return PROGRAM, RAD, RAD, RAD, PROGRAM

    @property
    def min_output_decision_level(self) -> gc.DecisionLevel:
        """GrAIde only makes sense for program-level evaluations as it operates at all decision levels."""
        return PROGRAM

    def threshold_constraints(self) -> list[ThresholdConstraint]:
        return [
            # The first constraints prevents some nonsensical buckets
            # FYI: we don't require t[1] <= t[2] because the two models' scores are not comparable.
            lambda t: t[0] <= t[1] and t[2] <= t[3],
            # The second restricts our two AI models to interesting threshold ranges
            lambda t: t[0] <= 0.25 and t[1] <= 0.25 and t[2] <= 0.75,
        ]

    def buckets_from_thresholds(
        self, thresholds: tuple[float, float, float, float]
    ) -> tuple[Bucket, Bucket, Bucket, Bucket, Bucket]:
        """
        We partition the studies into 5 buckets using the four thresholds, where the balanced AI model scores are used
        to determine the first three buckets and the high-specificity AI model scores to determine the last three
        (both are involved in determining the deferral bucket).
        """

        np_p_bucket = self.y_score_bal < thresholds[0]
        np_r_bucket = (thresholds[0] <= self.y_score_bal) & (self.y_score_bal < thresholds[1])
        dd_bucket = (thresholds[1] <= self.y_score_bal) & (self.y_score_spec < thresholds[2])
        sn_r_bucket = (thresholds[2] <= self.y_score_spec) & (self.y_score_spec < thresholds[3])
        sn_p_bucket = thresholds[3] <= self.y_score_spec

        return np_p_bucket, np_r_bucket, dd_bucket, sn_r_bucket, sn_p_bucket

    def positive_ai_buckets(self, buckets: tuple[Bucket, Bucket, Bucket, Bucket, Bucket]) -> list[Bucket]:
        return list(buckets[-2:])

    def ai_buckets(self, buckets: tuple[Bucket, Bucket, Bucket, Bucket, Bucket]) -> list[Bucket]:
        return [buckets[0], buckets[1], buckets[3], buckets[4]]

    def screening_paths(self) -> list[screening_path.ScreeningPath]:
        return [
            screening_path.direct_no_recall,
            screening_path.negative_ai_read,
            screening_path.screening_system_2_readers,
            screening_path.suspicious_ai_read,
            screening_path.direct_recall,
        ]


class DeferralToASingleReader(DecisionDeferral):
    """
    This strategy is very similar to program-level DD except that studies in the deferral bucket get sent to a single
    reader for a final decision rather than to the entire two-read-plus-CC program.

    We inherit from the DD class here s.t. we can reuse the `threshold_constraints`, `buckets_from_thresholds`, and
    `positive_ai_buckets` functions.

    In Germany there is no "first-reader" so we evaluate with both, but in other non-independent-read screening
    systems, like the UK, there is, and in those cases we always evaluate this strategy using the first reader.
    """

    def __init__(self):
        """Must be defined in order to remove the argument requirements of the DecisionDeferral constructor."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def input_decision_level(self) -> tuple[gc.DecisionLevel, ...]:
        return PROGRAM, RAD, PROGRAM

    @property
    def min_output_decision_level(self) -> gc.DecisionLevel:
        """
        DeferralToASingleReader is equivalent to RadLevelDD on a rad-level, but different for all higher-level
        evaluations.
        """
        return SEND2CC

    def screening_paths(self) -> list[screening_path.ScreeningPath]:
        return [
            screening_path.direct_no_recall,
            screening_path.screening_system_1_reader,
            screening_path.direct_recall,
        ]
