"""
This is the primary entrypoint of this repository. For a given screening strategy on a given dataset, evaluated at a
given decision-level, this script computes metrics (sens, spec, CDR, recall, etc.) for a number of different possible
combinations of AI thresholds. Once these results are computed for all strategies on all datasets, the various script
in the `artefacts` directory can be used to recreate the figures and tables in the paper.
"""
import itertools
import logging
from functools import partial
from typing import Iterator

import numpy as np
from hydra.utils import instantiate
from omegaconf import MISSING
from pydantic import root_validator

from graide import constants as gc
from graide.conf.entrypoint import entrypoint
from graide.conf.lib import BaseConfig, register_config, save_config
from graide.conf.strategy import StrategyBaseConfig
from graide.strategy import ScreeningStrategy
from graide.util import path_util, screening_util, system_util
from graide.util.types import PredsAndMetadata, ScreeningResults, ThresholdConstraint, Thresholds


@register_config(group="entrypoint")
class RunScreeningEntrypointConfig(BaseConfig):

    # The strategy we wish to evaluate
    strategy: StrategyBaseConfig = MISSING

    # The output decision level, i.e. the level at which we wish to evaluate the performance of the specified strategy
    decision_level: gc.DecisionLevel = MISSING

    # Which screening system to evaluate
    screening_system: gc.ScreeningSystem = MISSING

    # The name of the dataset to evaluate (for which a CSV should exist in the data directory).
    dataset_name: str = MISSING

    @root_validator
    def _validate_valid_strategy(cls, values):
        """
        Validates that the given strategy makes sense for the given decision level. For example, ProgramLevelDD isn't
        possible if running decision_level=RAD, but standalone AI can be evaluated at any level.
        """

        decision_level = values["decision_level"]
        strategy: ScreeningStrategy = instantiate(values["strategy"])
        if strategy.min_output_decision_level.is_greater_than(decision_level):
            raise ValueError(
                "You must specify a strategy whose minimum output decision level is less than or equal to the "
                f"decision level you wish to evaluate. The {strategy.name} strategy can only be evaluated at "
                f"{strategy.min_output_decision_level} or higher (you specified `decision_level={decision_level}`)."
            )

        return values


def _artefact_dir(config: RunScreeningEntrypointConfig) -> str:
    """Returns the directory in which all artefacts output by this script will be saved."""

    strategy: ScreeningStrategy = instantiate(config.strategy)
    return path_util.strategy_dir(config.decision_level, config.dataset_name, strategy.name)


def _thresholds_iter(
    n_thresholds: int,
    threshold_constraints: list[ThresholdConstraint],
    interval_size: float = 0.01,
) -> Iterator[Thresholds]:
    """The threshold iterator that, given some constraints and the number to generate, yields tuples of thresholds."""

    for thresholds in itertools.product(np.arange(0, 1 + interval_size, interval_size), repeat=n_thresholds):

        # We only consider sets of thresholds that satisfy all constraints
        if all(constraint(thresholds) for constraint in threshold_constraints):
            yield thresholds


def run_screening(
    data: PredsAndMetadata,
    strategy: ScreeningStrategy,
    decision_level: gc.DecisionLevel,
    output_dir: str,
    screening_system: gc.ScreeningSystem = gc.ScreeningSystem.GERMANY,
) -> ScreeningResults:

    strategy.init_data(data)

    logging.info("Computing reference result & defining entities")
    metric_stratifications = {"SDC": data.is_sdc, "IC": data.is_ic, "NRSDC": data.is_nrsdc}
    reference_result = screening_util.get_reference_result(data, decision_level, metric_stratifications)

    # Define the callable that will compute results for the given strategy for each set of thresholds
    result_computer = partial(
        strategy.compute_result_at_thresholds,
        decision_level=decision_level,
        reference_decisions=screening_util.get_reference_decisions(data, decision_level),
        reference_result=reference_result,
        program_workload=screening_util.get_program_workload(data, decision_level),
        metric_stratifications=metric_stratifications,
        screening_system=screening_system,
        # In order to prevent unnecessary computation, we only need to compute CIs for the best results. This is
        # handled later within the `artefacts` scripts.
        with_stats=False,
    )

    # Iterate through our sets of thresholds, generating one `StrategyResult` for each.
    thresholds = _thresholds_iter(
        strategy.n_thresholds, strategy.threshold_constraints(), strategy.threshold_interval_size
    )
    strategy_results = [
        result
        for result in system_util.compute_parallel(
            result_computer, thresholds, tqdm_desc=f"Computing {strategy.name} results"
        )
        if result is not None  # filter out the Nones from sets of thresholds that didn't partition the data
    ]

    # Create ScreeningResults object, (maybe) save to disk, & return.
    screening_results = ScreeningResults(decision_level, reference_result, strategy_results, screening_system)
    if output_dir is not None:
        screening_results.save(output_dir)
        screening_results.to_csv(output_dir)
        screening_results.cdr_optimising_result.save(output_dir, gc.CDR_RESULT_FN)
        screening_results.recall_optimising_result.save(output_dir, gc.RECALL_RESULT_FN)
        screening_results.oracle_result().save(output_dir, gc.ORACLE_RESULT_FN)
        logging.info(f"Saved {strategy.name} results to '{output_dir}'")

    return screening_results


@entrypoint(config_path=".", config_name="run-screening", cache_artefacts_func=_artefact_dir)
def main(config: RunScreeningEntrypointConfig):
    data = PredsAndMetadata.load(path_util.data_dir(), config.dataset_name)
    strategy: ScreeningStrategy = instantiate(config.strategy)
    strategy_dir = _artefact_dir(config)
    run_screening(data, strategy, config.decision_level, strategy_dir, config.screening_system)
    save_config(strategy_dir, config)


if __name__ == "__main__":
    main()
