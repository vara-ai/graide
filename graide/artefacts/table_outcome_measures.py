"""
This script can be used to generate Table 2 and eTables 4 and 6 from the paper, depending on the dataset used. The
`run_screening.py` entrypoint must have been run for each strategy on each of the two datasets used here.
"""
import os

import pandas as pd
from omegaconf import MISSING
from pydantic import validator
from tabulate import tabulate

from graide import constants as gc
from graide.artefacts.util import (
    BestResultsByStrategy,
    fmt_pct,
    frac_diff,
    get_best_results_by_strategy,
    get_ref_result,
    PRETTY_STRATEGY_NAMES,
    validate_strategy_names,
)
from graide.conf.entrypoint import entrypoint
from graide.conf.lib import BaseConfig, compose_one, register_config
from graide.util import io_util, path_util
from graide.util.types import PredsAndMetadata, ReferenceResult, StrategyResult, ValueWithCI


@register_config(group="entrypoint")
class TableOutcomeMeasuresEntrypointConfig(BaseConfig):
    strategy_names: list[str] = MISSING
    threshold_setting_dataset: str = MISSING
    test_dataset: str = MISSING
    screening_system: gc.ScreeningSystem = MISSING

    @validator("strategy_names")
    def _validate_strategy_names(cls, strategy_names: list[str]):
        return validate_strategy_names(strategy_names)


def create_table(best_results_by_strategy: BestResultsByStrategy, ref_result: ReferenceResult) -> list:
    """Format compiled results into a nice table."""

    strat_names = ["SDC", "IC", "NRSDC"]

    # Initialise the table with program-level performance
    ref_strat_cdr = ", ".join(
        [f"{ref_result.metrics.stratified_cdr[strat_name].value * 1000:.2f}" for strat_name in strat_names]
    )
    table = [
        [
            "German Screening Program",
            "-",
            f"{ref_result.metrics.cdr.value * 1000:.2f}",
            ref_strat_cdr,
            "-",
            "-",
            f"{ref_result.metrics.recall.value * 100:.2f}",
            "-",
            "-",
            f"{fmt_pct(0, 1)}",
        ]
    ]

    for strategy_name in best_results_by_strategy:
        pretty_strategy_string = PRETTY_STRATEGY_NAMES[strategy_name]

        result: StrategyResult
        for result2display in best_results_by_strategy[strategy_name]:
            result = result2display.result

            # (Maybe) show CDR stratified by cancer type (i.e. these should sum to the overall CDR if the
            # stratification masks completely cover the set of studies).
            strat_cdr_string = (
                "-"
                if result.metrics.stratified_cdr is None
                else ", ".join(
                    [f"{result.metrics.stratified_cdr[strat_name].value * 1000:.2f}" for strat_name in strat_names]
                )
            )

            frac_diff_cdr = frac_diff(result.metrics.cdr.value, ref_result.metrics.cdr.value)
            frac_diff_recall = frac_diff(result.metrics.recall.value, ref_result.metrics.recall.value)

            # Display the results of both superiority & non-inferiority tests for all metrics & results.
            cdr_diff_p_value_str = recall_diff_p_value_str = "-"
            if result.delta is not None:
                cdr_p_value, recall_p_value = result.delta.cdr.p_value, result.delta.recall.p_value

                cdr_diff_p_value_str = "<0.0001" if cdr_p_value < 0.0001 else f"{cdr_p_value:.4g}"
                recall_diff_p_value_str = "<0.0001" if recall_p_value < 0.0001 else f"{recall_p_value:.4g}"

            def _value_with_ci_str(value_with_ci: ValueWithCI) -> str:
                return f"{value_with_ci.value:.2f} ({value_with_ci.ci_low:.2f}, {value_with_ci.ci_upp:.2f})"

            table.append(
                [
                    pretty_strategy_string,
                    result2display.name,
                    _value_with_ci_str(result.metrics.cdr * 1000),
                    strat_cdr_string,
                    f"{fmt_pct(frac_diff_cdr, 1)}",
                    cdr_diff_p_value_str,
                    _value_with_ci_str(result.metrics.recall * 100),
                    f"{fmt_pct(frac_diff_recall, 1)}",
                    recall_diff_p_value_str,
                    f"{fmt_pct(result.workload_reduction, 0)}",
                ]
            )

    return table


def table_outcome_measures(config: TableOutcomeMeasuresEntrypointConfig):

    data_dir = path_util.data_dir()
    analysis_dir = os.path.join(data_dir, "analysis")

    # Load threshold-setting and test model preds and metadata
    threshold_setting_data = PredsAndMetadata.load(data_dir, config.threshold_setting_dataset)
    test_data = PredsAndMetadata.load(data_dir, config.test_dataset)
    test_ds_name = test_data.dataset_name.lower()

    # Load best results for each strategy on the test dataset
    strategies = [compose_one("strategy", strategy_name, instantiate=True) for strategy_name in config.strategy_names]
    best_results_by_strategy = get_best_results_by_strategy(
        strategies, threshold_setting_data, test_data, screening_system=config.screening_system
    )

    # Create flat headers for the tabulate table ...
    cdr_header = "CDR (/ 1000)"
    recall_header = "Recall rate (%)"
    headers_flat = [
        "\nScreening Strategy",
        "\nStrategy optimised for;",
        f"{cdr_header}\nOverall (95% CI)",
        f"{cdr_header}\nof which SDC, IC, NRSDC",
        f"{cdr_header}\nChange (%)",
        f"{cdr_header}\np-value (diff.)",
        f"{recall_header}\nOverall (95% CI)",
        f"{recall_header}\nChange (%)",
        f"{cdr_header}\np-value (diff.)",
        "Workload reduction\n",
    ]

    # ... & multi-indexed headers for the dataframe.
    name_headers = pd.MultiIndex.from_product([[""], ["Screening Strategy", "Strategy optimised for"]])
    cdr_headers = pd.MultiIndex.from_product(
        [
            ["Cancer detection rate (/ 1000)"],
            ["Overall (95% CI)", "of which SDC, IC, NRSDC", "Change (%)", "p-value (diff.)"],
        ]
    )
    recall_headers = pd.MultiIndex.from_product(
        [
            ["Recall rate (%)"],
            ["Overall (95% CI)", "Change (%)", "p-value (diff.)"],
        ]
    )
    last_header = pd.MultiIndex.from_product([["Workload Reduction [%]"], [""]])
    headers_multi = name_headers.append(cdr_headers).append(recall_headers).append(last_header)

    # Display & save the results :)
    table_test = create_table(best_results_by_strategy, get_ref_result(test_data))
    df = pd.DataFrame(table_test, columns=headers_multi)
    print(f"\n\nTest dataset: {test_ds_name}")
    print(tabulate(table_test, headers=headers_flat, tablefmt="fancy_grid"))
    print(df)

    output_dir = os.path.join(analysis_dir, config.threshold_setting_dataset, config.test_dataset)
    io_util.save_df_as_csv(df, os.path.join(output_dir, "table_outcome_measures.csv"))
    io_util.save_df_as_h5(df, os.path.join(output_dir, "table_outcome_measures.h5"))


@entrypoint(config_path=".", config_name="table2")
def main(config: TableOutcomeMeasuresEntrypointConfig):
    # TODO: get the config-name here and use it to create output files etc.
    table_outcome_measures(config)


if __name__ == "__main__":
    main()
