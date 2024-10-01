import copy
import os
from dataclasses import dataclass
from typing import Type

from graide import constants as gc
from graide.conf.lib import compose_one
from graide.stats import compute_metrics
from graide.strategy import ScreeningStrategy
from graide.util import io_util, path_util
from graide.util.logging import logger
from graide.util.screening_util import compute_delta, get_reference_decisions, get_reference_result
from graide.util.types import (
    BooleanArray,
    Decisions,
    PredsAndMetadata,
    ReferenceResult,
    ScreeningMetrics,
    ScreeningResults,
    StrategyResult,
    Workload,
)

PROG = gc.DecisionLevel.PROGRAM


def _stratifications(preds_and_md: PredsAndMetadata) -> dict[str, BooleanArray]:
    return {"SDC": preds_and_md.is_sdc, "IC": preds_and_md.is_ic, "NRSDC": preds_and_md.is_nrsdc}


def get_ref_result(preds_and_md: PredsAndMetadata) -> ReferenceResult:
    """A wrapper to quickly get the reference we care about for all artefacts in the paper."""

    return get_reference_result(preds_and_md, PROG, _stratifications(preds_and_md))


def frac_diff(strategy_value: float, ref_value: float) -> float:
    """Computes the fractional difference of `strategy_value` from `ref_value`."""

    return (strategy_value - ref_value) / ref_value


def fmt_pct(val: float, num_dp: int = 2) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{100 * val:.{num_dp}f}"


@dataclass
class Result2Display:
    name: str
    result: StrategyResult
    test_for_superiority_cdr: bool
    test_for_superiority_recall: bool

    def with_new_result(self, new_result: StrategyResult) -> "Result2Display":
        return Result2Display(
            name=self.name,
            result=new_result,
            test_for_superiority_cdr=self.test_for_superiority_cdr,
            test_for_superiority_recall=self.test_for_superiority_recall,
        )


BestResultsByStrategy = dict[str, list[Result2Display]]


def get_best_results_by_strategy(
    strategies: list[ScreeningStrategy],
    val_data: PredsAndMetadata,
    test_data: PredsAndMetadata,
    screening_system: gc.ScreeningSystem = gc.ScreeningSystem.GERMANY,
) -> BestResultsByStrategy:
    """
    For each passed strategy, this function selects the three "best" results on the validation dataset (CDR-maximising,
    recall-minimising, and oracle-like), transfers those thresholds to the test dataset, and adds "statistical rigour"
    to the test results, i.e. CIs and deltas with p values. The dict returned by this function is used by each
    artefact-creating script.

    NB: This function requires that `run_screening` has already been run for each strategy passed.
    """

    analysis_dir = path_util.analysis_dir()

    ref_result_val = get_ref_result(val_data)
    ref_result_test = get_ref_result(test_data)
    dataset_name_val = val_data.dataset_name
    dataset_name_test = test_data.dataset_name

    # This loop selects a CDR-optimising, a recall-optimising, and an oracle-like result for each strategy on the
    # validation dataset and finds and stores the result on test data that uses those thresholds.
    test_results_by_strategy = {}
    for strategy in strategies:

        # 1. First we select three exemplar results on validation data, namely a CDR-optimising, a recall-optimising,
        #    and an oracle-like result. For each result we also determine, for each of CDR & recall, which statistical
        #    test we'd like to run — superiority or non-inferiority — depending on both which metric the given result
        #    is optimising and whether the result is super-reference.

        strategy_dir_val = path_util.strategy_dir(PROG, dataset_name_val, strategy.name)
        if not io_util.file_exists(strategy_dir_val):
            logger.warning(f"No results found for the {strategy.name} strategy on {dataset_name_val}.")
            continue

        val_ds_name = dataset_name_val.lower()
        best_results_val_path = os.path.join(analysis_dir, val_ds_name, f"best_results_{strategy.name}.pklz")
        try:
            best_results_val = io_util.load_from_pickle(best_results_val_path)
            logger.info(
                f"Loading val results for the {strategy.name} strategy on {dataset_name_val} from "
                f"{best_results_val_path}. Please delete that file if you want them to be computed anew."
            )
        except FileNotFoundError:
            logger.info(f"Selecting val results for the {strategy.name} strategy on {dataset_name_val}")

            cdr_result = StrategyResult.load(strategy_dir_val, gc.CDR_RESULT_FN)
            recall_result = StrategyResult.load(strategy_dir_val, gc.RECALL_RESULT_FN)
            oracle_result = StrategyResult.load(strategy_dir_val, gc.ORACLE_RESULT_FN)
            oracle_result_is_super = oracle_result.is_super_reference(ref_result_val, use_cdr_and_recall=True)

            best_results_val = [
                Result2Display(
                    name="CDR",
                    result=cdr_result,
                    test_for_superiority_cdr=cdr_result.is_super_reference(ref_result_val, use_cdr_and_recall=True),
                    test_for_superiority_recall=False,
                ),
                Result2Display(
                    name="Balanced",
                    result=oracle_result,
                    test_for_superiority_cdr=oracle_result_is_super,
                    test_for_superiority_recall=oracle_result_is_super,
                ),
                Result2Display(
                    name="Recall",
                    result=recall_result,
                    test_for_superiority_cdr=False,
                    test_for_superiority_recall=recall_result.is_super_reference(
                        ref_result_val, use_cdr_and_recall=True
                    ),
                ),
            ]
            io_util.save_to_pickle(best_results_val, best_results_val_path)

            # If we don't already have it we add statistical rigour to our three chosen results in order to evaluate
            # super-referenceness.
            logger.info(f"Adding statistical rigour to {strategy.name} val results on {dataset_name_val}")
            best_results_val = [
                result2display.with_new_result(
                    add_statistical_rigour_to_result(
                        result2display.result, ref_result_val, val_data, PROG, screening_system=screening_system
                    )
                )
                if result2display.result.delta is None
                else result2display
                for result2display in best_results_val
            ]
            io_util.save_to_pickle(best_results_val, best_results_val_path)

        # 2. If we haven't previously transferred our validation results to test data, we evaluate each of their
        # thresholds to determine test results & save the artefact.

        strategy_dir_test = path_util.strategy_dir(PROG, dataset_name_test, strategy.name)
        if not io_util.file_exists(strategy_dir_test):
            logger.warning(f"No results found for the {strategy.name} strategy on {dataset_name_test}.")
            continue

        test_ds_name = dataset_name_test.lower()
        best_results_test_path = os.path.join(
            analysis_dir, val_ds_name, test_ds_name, f"best_results_{strategy.name}.pklz"
        )
        try:
            best_results_test = io_util.load_from_pickle(best_results_test_path)
            logger.info(
                f"Loading test results for the {strategy.name} strategy on {dataset_name_test} from "
                f"{best_results_test_path}. Please delete that file if you want them to be computed anew."
            )
        except FileNotFoundError:
            logger.info(f"Selecting test results for the {strategy.name} strategy on {dataset_name_test}")
            strategy_results_test = ScreeningResults.load(strategy_dir_test)
            best_results_test = [
                result2display.with_new_result(
                    strategy_results_test.get_result_by_thresholds(result2display.result.thresholds)
                )
                for result2display in best_results_val
            ]
            io_util.save_to_pickle(best_results_test, best_results_test_path)

            # If we don't already have it we add statistical rigour to our three chosen results in order to evaluate
            # super-referenceness.
            logger.info(f"Adding statistical rigour to {strategy.name} test results on {dataset_name_test}")
            best_results_test = [
                result2display.with_new_result(
                    add_statistical_rigour_to_result(
                        result2display.result, ref_result_test, test_data, PROG, screening_system=screening_system
                    )
                )
                if result2display.result.delta is None
                else result2display
                for result2display in best_results_test
            ]
            io_util.save_to_pickle(best_results_test, best_results_test_path)

        test_results_by_strategy[strategy.name] = best_results_test

    return test_results_by_strategy


def _all_subclasses(cls: Type) -> set[Type]:
    """Returns a set of the subclasses of a given type, as `.__subclasses__()` only returns immediate subclasses."""

    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in _all_subclasses(c)])


def _get_strategy_from_name(name: str) -> ScreeningStrategy:
    """Gets a ScreeningStrategy class instance from its string name, usually from a StrategyResult."""

    from graide.strategy import DecisionDeferral, ScreeningStrategy

    for subclass in _all_subclasses(ScreeningStrategy):
        if name == subclass.__name__:
            return subclass()

    if name.endswith("DD"):
        deferral_level = gc.DecisionLevel[name.split("Level")[0].upper()]
        return DecisionDeferral(deferral_level=deferral_level)

    raise ValueError(f"The strategy name '{name}' doesn't correspond to any defined `ScreeningStrategy` subclass.")


def _recover_decisions_and_workload_from_result(
    result: StrategyResult,
    preds_and_md: PredsAndMetadata,
    decision_level: gc.DecisionLevel,
    screening_system: gc.ScreeningSystem = gc.ScreeningSystem.GERMANY,
) -> tuple[Decisions, Workload]:
    """Recovers the decisions & workload arrays of a given strategy result at a given decision level."""

    strategy = _get_strategy_from_name(result.strategy_name)
    strategy.init_data(preds_and_md)
    buckets = strategy.buckets_from_thresholds(result.thresholds)
    return strategy.decisions_from_buckets(buckets, decision_level, screening_system)


def add_statistical_rigour_to_result(
    strategy_result: StrategyResult,
    reference_result: ReferenceResult,
    preds_and_md: PredsAndMetadata,
    decision_level: gc.DecisionLevel,
    screening_system: gc.ScreeningSystem = gc.ScreeningSystem.GERMANY,
) -> StrategyResult:
    """Adds confidence intervals and computes deltas (maybe incl. p values) for all metrics in the given StrategyResult.

    Args:
        strategy_result: the result to which we want to add statistical rigour.
        reference_result:
            the reference against which we want to compute a delta. This and the `strategy_result` should have had
            their metrics computed at the same decision level.
        preds_and_md: paired predictions & metadata from which we can extract program metadata.
        decision_level: the decision level of the strategy & reference results, needed to recover decisions arrays.
        screening_system: the screening system we're evaluating

    Returns: A copy of the input StrategyResult with CIs & deltas, the latter computed w.r.t. the given reference.
    """

    # Get the reference decisions array and 'recover' the strategy result's decisions array
    reference_decisions = get_reference_decisions(preds_and_md, decision_level)
    decisions, _ = _recover_decisions_and_workload_from_result(
        strategy_result, preds_and_md, decision_level, screening_system
    )

    # Re-compute the screening metrics, this time with bootstrapping (transferring the stratified_cdr values).
    stratified_cdr = copy.deepcopy(strategy_result.metrics.stratified_cdr)
    strategy_result.metrics = ScreeningMetrics(
        *compute_metrics(preds_and_md.y_true, decisions, preds_and_md.weights, n_resamples=1000)
    )
    strategy_result.metrics.stratified_cdr = stratified_cdr

    # Compute a delta (incl. p values) for the difference between the strategy result & the reference
    strategy_result.delta = compute_delta(
        strategy_result.metrics,
        reference_result.metrics,
        decisions,
        reference_decisions,
        preds_and_md,
        with_stats=True,
    )

    return strategy_result


PRETTY_STRATEGY_NAMES = {
    "Graide": "GrAIde",
    "ProgramLevelDD": "Decision Referral (program-level)",
    "RadLevelDD": "Decision Referral (reader-level)",
    "DeferralToASingleReader": "Deferral to a Single Reader",
    "ReaderReplacement": "Single Reader Replacement",
    "NormalTriaging": "Normal Triaging",
    "StandaloneAI": "Standalone AI",
}


def validate_strategy_names(strategy_names: list[str]) -> list[str]:
    """Validates that the names passed are valid strategies (unfortunately Hydra can't compose lists of configs)."""

    for strategy_name in strategy_names:
        try:
            compose_one("strategy", strategy_name)
        except KeyError:
            raise ValueError(f"The strategy {strategy_name} wasn't found in the config store.")
    return strategy_names
