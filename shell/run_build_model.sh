#!/bin/bash

cd ../src
# $1 = exp id
python prepare_dataset_model.py /src/experiments/credit-training-setting$1.json
python check_model_perf.py /src/experiments/credit-training-setting$1.json
