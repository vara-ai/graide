"""
This script can be used to generate Table 3 and eTables 5 and 7 from the paper, depending on the dataset used. The
`run_screening.py` entrypoint must have been run for each strategy on each of the two datasets used here.
"""
import itertools
import os

import more_itertools
import numpy as np
import pandas as pd
from omegaconf import MISSING
from pydantic import validator
from tabulate import tabulate

from graide import constants as gc
from graide.artefacts.util import get_best_results_by_strategy, PRETTY_STRATEGY_NAMES, validate_strategy_names
from graide.conf.entrypoint import entrypoint
from graide.conf.lib import BaseConfig, compose_one, register_config
from graide.screening_path import (
    direct_no_recall,
    direct_recall,
    negative_ai_read,
    suspicious_ai_read,
    screening_system_1_reader,
    screening_system_2_readers,
)
from graide.strategy import ScreeningStrategy
from graide.util import io_util, path_util
from graide.util.types import Bucket, PredsAndMetadata

SCREENING_PATH_TO_IDX = {
    direct_no_recall: 0,
    negative_ai_read: 1,
    screening_system_1_reader: 2,
    screening_system_2_readers: 2,
    suspicious_ai_read: 3,
    direct_recall: 4,
}


@register_config(group="entrypoint")
class TablePathAllocationsEntrypointConfig(BaseConfig):
    strategy_names: list[str] = MISSING
    threshold_setting_dataset: str = MISSING
    test_dataset: str = MISSING
    screening_system: gc.ScreeningSystem = MISSING

    @validator("strategy_names")
    def _validate_strategy_names(cls, strategy_names: list[str]):
        return validate_strategy_names(strategy_names)


def _fmt_proportion(proportion: float) -> str:
    return f"{100 * proportion:.1f}"


def _fmt_prevalence(prevalence: float) -> str:
    return f"{1000 * prevalence:.2f}"


def table_path_allocations(config: TablePathAllocationsEntrypointConfig):

    data_dir = path_util.data_dir()
    analysis_dir = os.path.join(data_dir, "analysis")

    # Load threshold-setting and test model preds and metadata
    threshold_setting_data = PredsAndMetadata.load(data_dir, config.threshold_setting_dataset)
    test_data = PredsAndMetadata.load(data_dir, config.test_dataset)

    # Load best results for each strategy on the test dataset
    strategies = [compose_one("strategy", strategy_name, instantiate=True) for strategy_name in config.strategy_names]
    best_results_by_strategy = get_best_results_by_strategy(
        strategies, threshold_setting_data, test_data, screening_system=config.screening_system
    )

    weights = test_data.weights
    is_cancer = test_data.is_cancer

    def _prevalence(bucket: Bucket, mask: np.ndarray) -> str:
        return _fmt_prevalence(weights[bucket & mask].sum() / weights[bucket].sum())

    def _strat_prevalence(bucket: Bucket) -> str:
        sdc_prev = _prevalence(bucket, test_data.is_sdc)
        ic_prev = _prevalence(bucket, test_data.is_ic)
        nrsdc_prev = _prevalence(bucket, test_data.is_nrsdc)
        return f"{sdc_prev}, {ic_prev}, {nrsdc_prev}"

    # Initialise the table with the screening program values (only in the "Screening System" bucket).
    prog_name = "German Screening Program"  # TODO: customise depending on the dataset used ...
    prev = _fmt_prevalence(weights[is_cancer].sum() / weights.sum())
    strat_prev = _strat_prevalence(np.ones_like(test_data.y_true, dtype=bool))
    table = [[prog_name, "—", "—", "—", "—", "—", "-", "-", "100", prev, strat_prev, "-", "-", "—", "—", "—", "—"]]

    strategy: ScreeningStrategy
    for strategy in strategies:
        strategy.init_data(test_data)

        for result2display in best_results_by_strategy[strategy.name]:
            row_names = [PRETTY_STRATEGY_NAMES[strategy.name], result2display.name]

            # We initialise the proportion & prevalence values for each bucket, only to be updated for the buckets that
            # the strategy actually used.
            bucket_values = [["-", "-", "-"] for _ in range(5)]

            # For each bucket in the strategy we compute the proportion of studies that ended up in there & the cancer
            # prevalence within those studies, both overall & stratified by cancer type.
            buckets = strategy.buckets_from_thresholds(result2display.result.thresholds)
            for bucket, screening_path in zip(buckets, strategy.screening_paths()):
                bucket_idx = SCREENING_PATH_TO_IDX[screening_path]
                bucket_size = bucket.sum()
                bucket_values[bucket_idx][0] = _fmt_proportion(weights[bucket].sum() / weights.sum())
                bucket_values[bucket_idx][1] = "-" if bucket_size == 0 else _prevalence(bucket, is_cancer)
                bucket_values[bucket_idx][2] = "-" if bucket_size == 0 else _strat_prevalence(bucket)

                # If we have the 1-reader variant of the screening system screening path we denote with an asterisk that
                # this means something different (namely that we took one reader's decision as the final decision).
                if screening_path == screening_system_1_reader:
                    bucket_values[bucket_idx] = [v + " *" for v in bucket_values[bucket_idx]]

            table.append(row_names + list(more_itertools.flatten(bucket_values)))

    # The names of the screening paths & the values we need to display for each (for each strategy result row).
    bucket_names = ["Direct No Recall", "Negative AI Read", "Screening System", "Suspicious AI Read", "Direct Recall"]
    value_columns = ["Proportion (%)", "Prevalence (/ 1000)", "of which SDC, IC, NRSDC"]
    bucket_value_product = itertools.product(bucket_names, value_columns)

    # Display flat table (since tabulate doesn't support nice multi-indexing) ...
    headers_flat = ["\nScreening Strategy", "\nStrategy optimised for"]
    for bucket_name, value_col_name in bucket_value_product:
        headers_flat.append(f"{bucket_name}\n{value_col_name}")
    print(tabulate(table, headers=headers_flat, tablefmt="fancy_grid"))

    # ... & create a proper multi-indexed table for use in pandas etc.
    headers_multi = pd.MultiIndex.from_product([[""], ["Screening Strategy", "Strategy optimised for"]]).append(
        pd.MultiIndex.from_product([bucket_names, value_columns])
    )
    df = pd.DataFrame(table, columns=headers_multi)
    print(df)

    output_dir = os.path.join(analysis_dir, config.threshold_setting_dataset, config.test_dataset)
    io_util.save_df_as_csv(df, os.path.join(output_dir, "table_path_allocations.csv"))
    io_util.save_df_as_h5(df, os.path.join(output_dir, "table_path_allocations.h5"))


@entrypoint(config_path=".", config_name="table3")
def main(config: TablePathAllocationsEntrypointConfig):
    table_path_allocations(config)


if __name__ == "__main__":
    main()
