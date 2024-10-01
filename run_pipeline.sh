#!/bin/bash

# Make sure the script aborts if any of the intermediate steps fail
set -euo pipefail  # https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/

# Run `run_screening.py` on Germany datasets
python graide/run_screening.py -m \
  decision_level=PROGRAM \
  strategy=NormalTriaging,ReaderReplacement,RadLevelDD,ProgramLevelDD,DeferralToASingleReader,Graide,StandaloneAI \
  dataset_name=germany_val,germany_test \
  screening_system=GERMANY

# Run `run_screening.py` on UK datasets
python graide/run_screening.py -m \
  decision_level=PROGRAM \
  strategy=NormalTriaging,ReaderReplacement,RadLevelDD,ProgramLevelDD,DeferralToASingleReader,Graide,StandaloneAI \
  dataset_name=uk_val,uk_test \
  screening_system=UK

# Run `run_screening.py` on Sweden datasets
python graide/run_screening.py -m \
  decision_level=PROGRAM \
  strategy=NormalTriaging,ProgramLevelDD,StandaloneAI \
  dataset_name=sweden_val,sweden_test \
  screening_system=GERMANY  # because the process is the same

# Generate artefacts (unfortunately can't multi-run over config names)
python graide/artefacts/table_outcome_measures.py --config-name="table2"
python graide/artefacts/table_outcome_measures.py --config-name="etable4"
python graide/artefacts/table_outcome_measures.py --config-name="etable6"
python graide/artefacts/table_path_allocations.py --config-name="table3"
python graide/artefacts/table_path_allocations.py --config-name="etable5"
python graide/artefacts/table_path_allocations.py --config-name="etable7"
python graide/artefacts/figure2.py
