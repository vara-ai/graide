import numpy as np

from graide import constants as gc
from graide.stats import (
    compute_cdr,
    compute_metrics,
    compute_recall_rate,
    compute_sens,
    compute_spec,
    compute_stratified_cdr,
    compute_stratified_sens,
    permutation_test,
)
from graide.util.types import (
    BooleanArray,
    Decisions,
    Delta,
    PredsAndMetadata,
    ReferenceResult,
    ScreeningMetrics,
    ValueWithPValue,
    Workload,
)


def has_decision_dim(decisions: Decisions) -> bool:
    """We sometimes need an extra dimension to account for the decisions from two different radiologists."""

    return decisions.ndim == 2


def align_decisions_shapes(
    strategy_decisions: Decisions,
    reference_decisions: Decisions,
) -> tuple[Decisions, Decisions]:
    """Ensures that both decisions arrays have a decision dimension if either of them does."""

    strategy_has_decision_dim = has_decision_dim(strategy_decisions)
    reference_has_decision_dim = has_decision_dim(reference_decisions)

    if strategy_has_decision_dim and not reference_has_decision_dim:
        reference_decisions = np.c_[reference_decisions, reference_decisions]
    elif reference_has_decision_dim and not strategy_has_decision_dim:
        strategy_decisions = np.c_[strategy_decisions, strategy_decisions]

    return strategy_decisions, reference_decisions


def get_program_workload(data: PredsAndMetadata, decision_level: gc.DecisionLevel) -> float:
    """Computes the program workload at a given decision-level."""

    # We start with one (weighted) unit of work per study.
    workload = data.weights.sum()

    # If our reference is at a 'higher' decision level than rad-level then we multiply by 2 to account for the two reads
    if decision_level is not gc.DecisionLevel.RAD:
        workload *= 2

    # And if we're evaluating at the program-level we add one (weighted) unit of work for each CC read
    if decision_level is gc.DecisionLevel.PROGRAM:
        workload += data.weights[data.y_send2cc].sum()

    return workload


def compute_workload_reduction(strategy_workload: Workload, program_workload: float, weights: np.ndarray) -> float:
    """
    Computes the fractional workload reduction achieved via using a given strategy instead of the program. If the
    `strategy_workload` array has a decision dimension then we first take the mean over that to get a single averaged
    value per study.

    Args:
        strategy_workload:
            workload array, either one or two values per study (depending on whether we have a decision dimension).
        program_workload: a float representing the program's total workload (at some decision level).
        weights: the per-study weights, required to compute a weighted sum of the strategy's workload
    """

    if has_decision_dim(strategy_workload):
        strategy_workload = strategy_workload.mean(axis=1)

    return (program_workload - (strategy_workload * weights).sum()) / program_workload


def get_reference_decisions(data: PredsAndMetadata, decision_level: gc.DecisionLevel) -> Decisions:
    """Returns the reference decisions at a given decision level."""

    return {
        gc.DecisionLevel.RAD: data.y_rad,
        gc.DecisionLevel.SEND2CC: data.y_send2cc,
        gc.DecisionLevel.PROGRAM: data.y_program,
    }[decision_level]


def get_reference_result(
    data: PredsAndMetadata,
    decision_level: gc.DecisionLevel,
    metric_stratifications: dict[str, BooleanArray] | None = None,
) -> ReferenceResult:
    """Gets the `ReferenceResult` object for the reference at a given decision level."""

    # Compute standard metrics using the reference decisions
    reference_decisions = get_reference_decisions(data, decision_level)
    metrics = ScreeningMetrics(*compute_metrics(data.y_true, reference_decisions, data.weights))

    # (Optionally) add stratified stratified metrics
    if metric_stratifications:
        metrics.stratified_sens = compute_stratified_sens(
            metric_stratifications, data.y_true, reference_decisions, data.weights
        )
        metrics.stratified_cdr = compute_stratified_cdr(
            metric_stratifications, data.y_true, reference_decisions, data.weights
        )

    return ReferenceResult(decision_level=decision_level, metrics=metrics)


def compute_delta(
    strategy_metrics: ScreeningMetrics,
    reference_metrics: ScreeningMetrics,
    strategy_decisions: Decisions,
    reference_decisions: Decisions,
    preds_and_md: PredsAndMetadata,
    with_stats: bool = True,
) -> Delta:
    """
    Computes a Delta object between a given strategy result & a given reference result via running a
    `permutation_test`.

    Args:
        [strategy/reference]_metrics: the metrics from the strategy & reference from which to compute absolute deltas
        [strategy/reference]_decisions:
            the boolean decisions of the strategy and the reference with which to run our permutation test
        preds_and_md: paired predictions & metadata
        with_stats:
            whether or not to compute a p value indicating whether the delta computed by this function constitutes a
            statistically significant difference.
    """

    # Compute absolute deltas for each metric
    deltas = {m: strategy_metrics.get(m).value - reference_metrics.get(m).value for m in gc.METRIC_NAMES}

    # Compute p values for each metric's delta using permutation tests
    p_values = {m: None for m in gc.METRIC_NAMES}
    if with_stats:

        strategy_decisions, reference_decisions = align_decisions_shapes(strategy_decisions, reference_decisions)
        test_result = permutation_test(
            y_true=preds_and_md.y_true,
            y_treatment=strategy_decisions,
            y_control=reference_decisions,
            weights=preds_and_md.weights,
            metric_fns={
                gc.SENS: compute_sens,
                gc.SPEC: compute_spec,
                gc.CDR: compute_cdr,
                gc.RECALL: compute_recall_rate,
            },
            n_permutations=1000,
        )

        # Validates that `permutation_test` returns (almost) equivalent absolute deltas to those computed above.
        # NB: these assertions can be raised if using `StrategyResults`s that have been loaded from disk => the
        # `ref_result` argument passed should be computed on the fly (using `dr_util.compute_ref_result`) rather
        # than from the loaded entity.
        for m in gc.METRIC_NAMES:
            assert np.isclose(test_result[m].delta, deltas[m]), (
                f"Unexpected difference in delta {m}: " f"{test_result[m].delta} vs. {deltas[m]}"
            )

        p_values = {m: test_result[m].p_value for m in gc.METRIC_NAMES}

    return Delta(**{m: ValueWithPValue(deltas[m], p_values[m]) for m in gc.METRIC_NAMES})
