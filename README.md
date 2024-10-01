# GrAIde

This repository contains the code and data necessary to reproduce the figures and tables of the Lancet Digital Health paper "Strategies for integrating artificial intelligence into mammography screening programs. A retrospective simulation analysis." or to evaluate any of the strategies on your own data.

If you find GrAIde useful in your research, please use the following BibTeX entry for citation.

```BibTeX
```

## Installation

First, clone the repository
```bash
git clone https://github.com/vara-ai/graide.git
```
then create a conda virtual environment and activate it:
```bash
conda create -n graide python=3.10
source activate graide
```
before installing the required dependencies via
```bash
conda install hdf5
pip install -r requirements.txt
```

## Data

We provide anonymized versions of all _evaluation_ data used, including model predictions, that enable the reproduction of all results. More specifically, for each of the German, UK, and Swedish datasets used in the paper, we provide dataframes for the validation and external test splits. For a full description of these six datasets, please see the _Data Sources and Participants_ section of the main [paper](www.TODOTODOTODO) as well as eMethods 2 and eMethods 3 of the [appendix](www.TODOTODOTODO). 

Each dataframe provided has one row per study, and the following columns:
- is_[sdc/ic/nrsdc]: boolean indicating whether the study is a screen-detected cancer, interval cancer, or next-round screen-detected cancer
- cancer_[grade/stage]: the grade or stage of the study, if present and if the study is a cancer (NaN otherwise)
- y_rad_[1/2]: the two radiologist reads for the study
- y_send2cc: whether or not the study went to the consensus conference (or arbitration in the UK)
- y_program: whether or not the study was recalled
- y_score_[bal/spec]: the suspicious score assigned to the study by each of two AI models
- weights: the sample weight of the study used by inverse probability weighting
- age: patient age, if available (NaN otherwise)

You can download all six datasets into the correct location by running `./setup_data.sh`, but if you prefer to access them manually they must be stored within the `data` directory next to the `graide` directory, as follows:
```console
├── data
|   ├── germany_val.csv
|   ├── germany_test.csv
|   ├── ...
├── graide
|   ├── ...
|   ├── run_screening.py
|   ├── ...
```

To use your own datasets you need to create dataframes in the same format as those described above, with the same columns, and store them in the same `data` directory.

## Usage

### Pipeline

To run everything and reproduce the figures and tables from the paper you can use the provided pipeline
```bash
./run_pipeline.sh
```
though be warned this takes >15 hours on a standard laptop (almost entirely due to the large threshold search space of the GrAIde strategy). The pipeline will generate the following artefacts
```console
├── data
|   ├── analysis
|   │   ├── figure2.png
|   │   ├── figure2.svg
|   │   ├── germany_val
|   │   │   ├── best_results_DeferralToASingleReader.pklz
|   │   │   ├── best_results_Graide.pklz
|   │   │   ├── ...
|   │   │   │   ├── germany_test
|   │   │   │   │   ├── best_results_DeferralToASingleReader.pklz
|   │   │   │   │   ├── best_results_Graide.pklz
|   │   │   │   │   ├── ...
|   │   │   │   │   ├── table_outcome_measures.csv
|   │   │   │   │   ├── table_outcome_measures.h5
|   │   │   │   │   ├── table_path_allocations.csv
|   │   │   │   │   ├── table_path_allocations.h5
|   │   ├── sweden_val
|   │   │   ├── ...
|   │   ├── uk_val
|   │   │   ├── ...
|   ├── program-level
|   │   ├── germany_test
|   │   │   ├── deferral_to_a_single_reader
|   │   │   │   ├── config-*.yaml
|   │   │   │   ├── result_cdr_optimising.pklz
|   │   │   │   ├── result_oracle.pklz
|   │   │   │   ├── result_recall_optimising.pklz
|   │   │   │   ├── results.csv
|   │   │   │   ├── results.pklz
|   │   │   ├── graide
|   │   │   ├── normal_triaging
|   │   │   ├── program_level_dd
|   │   │   ├── rad_level_dd
|   │   │   ├── reader_replacement
|   │   │   ├── standalone_ai
|   │   ├── germany_val
|   │   ├── sweden_test
|   │   ├── sweden_val
```

To run things more systematically, you can use the following entrypoints:

### `run_screening`

This is the primary entrypoint of this repository. For a given strategy, in a given screening system, on a given dataset, this entrypoint computes metrics (CDR, recall rate, and workload reduction) evaluated at a given decision level, for a number of different combinations of AI thresholds. Only once this script has been run for all strategies on all datasets can the tables and figures from the paper be generated. The best way to understand how to interact with this entrypoint is by reading the `run_pipeline.sh` script, as we demonstrate how to run `run_screening.py` for all combinations of strategies and datasets. 

In terms of general usage, 
- we provide implementations for the following seven screening strategies: standalone AI (`StandaloneAI`), normal triaging (`NormalTriaging`), single reader replacement (`ReaderReplacement`), radiologist-level decision referral (`RadLevelDD`), program-level decision referral (`ProgramLevelDD`), deferral to a single reader (`DeferralToASingleReader`), and GrAIde (`Graide`).
- our screening path logic supports two screening systems (as defined in the paper): Germany (`GERMANY`) and the UK (`UK`).
- all results in this paper were generated using `decision_level=PROGRAM`, i.e. measuring program-level metrics. The evaluation style of Leibig et. al 2022 can be achieved via `decision_level=RAD`.

Warning: as mentioned in the previous section, running `run_screening.py` with the GrAIde strategy takes >3 hours on a standard laptop for most provided datasets (>6 hours for the largest, `uk_test.csv`).

### Publication artefacts

Once `run_screening.py` has been run for all strategies on all datasets, the tables and figures from the paper can be generated. 

The following two table-generating scripts require that `run_screening.py` has been run for all strategies on each of the val and test datasets of a single country. More precisely, the two datasets passed should be a threshold-setting dataset and an external test dataset, i.e. the dataset on which AI thresholds will be set and the dataset on which metrics will be evaluated using those thresholds.
- `graide/artefacts/table_outcome_measures.py`: generates Table 2 and eTables 4 and 6 from paper, depending on which datasets are passed.
- `graide/artefacts/table_path_allocations.py`: generates Table 3 and eTables 5 and 7 from paper, depending on which datasets are passed.
Please note that named YAMLs have been provided that can be used to generate each of these artefacts, i.e.
```python
python graide/artefacts/table_outcome_measures.py --config-name="etable4"
```
and
```python
python graide/artefacts/table_path_allocations.py --config-name="table3"
``` 

The figure 2 script (`graide/artefacts/figure2.py`) requires that run_screening has been run for all strategies on _all six_ datasets. This script is run without configuration.

### Configuration

We use [Hydra](https://hydra.cc/) to configure all entrypoints. Please adapt any of the entrypoint YAMLs (e.g.` graide/run-screening.yaml`) or specify command-line overrides (as demonstrated in `run_pipeline.sh`).

### Reproducibility

The confidence intervals and p-values computed using this repository vary (slightly) between runs due to the inherently stochastic nature of those methods. We have never seen a reproduction that led to a different interpretation of the results.

## Contributors

Main contributor for the code used in the publication and for verifying its correctness:
* Michael Ball

Other contributors (alphabetical):
* Benjamin Strauch
* Christian Leibig
* Dominik Schüler
* Stefan Bunk
* Vilim Štih
* Zacharias Fisches

## Contact

At [Vara](https://www.vara.ai), we are committed to reproducible research. Please feel free to reach out to the corresponding author ([firstname] [dot] [lastname] [at] vara.ai) if you have trouble reproducing results or any questions about GrAIde or the paper.
