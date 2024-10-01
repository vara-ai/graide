#!/bin/bash

# Make sure the script aborts if any of the intermediate steps fail
set -euo pipefail  # https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/

mkdir -p data

# Download datasets
for filename in germany_val.csv germany_test.csv uk_val.csv uk_test.csv sweden_val.csv sweden_test.csv; do
  if ! [[ -e data/$filename ]]; then
      echo "$filename not found, downloading it"
      curl --output data/$filename https://storage.googleapis.com/mx-healthcare-pub/bm2/$filename
  else
      echo "$filename already found, run 'rm data/$filename' if you want to download it again"
  fi
done
