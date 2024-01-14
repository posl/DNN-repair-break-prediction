#!/bin/bash

cd ../src
# tabular datasets
echo tabular datasets...
python prepare_dataset_model.py /src/experiments/care/credit-training-setting1.json --train_model
python check_model_perf.py /src/experiments/care/credit-training-setting1.json
python get_explanatory_metrics.py /src/experiments/care/credit-training-setting1.json

python prepare_dataset_model.py /src/experiments/care/census-training-setting1.json --train_model
python check_model_perf.py /src/experiments/care/census-training-setting1.json
python get_explanatory_metrics.py /src/experiments/care/census-training-setting1.json

python prepare_dataset_model.py /src/experiments/care/bank-training-setting1.json --train_model
python check_model_perf.py /src/experiments/care/bank-training-setting1.json
python get_explanatory_metrics.py /src/experiments/care/bank-training-setting1.json

# image datasets
echo image datasets...
python prepare_dataset_model.py /src/experiments/care/fm-training-setting1.json --train_model
python check_model_perf.py /src/experiments/care/fm-training-setting1.json
python get_explanatory_metrics.py /src/experiments/care/fm-training-setting1.json

python prepare_dataset_model.py /src/experiments/care/c10-training-setting1.json --train_model
python check_model_perf.py /src/experiments/care/c10-training-setting1.json
python get_explanatory_metrics.py /src/experiments/care/c10-training-setting1.json

python prepare_dataset_model.py /src/experiments/care/gtsrb-training-setting1.json --train_model
python check_model_perf.py /src/experiments/care/gtsrb-training-setting1.json
python get_explanatory_metrics.py /src/experiments/care/gtsrb-training-setting1.json

# text datasets
echo text datasets...
python prepare_dataset_model.py /src/experiments/care/rtmr-training-setting1.json --train_model
python check_model_perf.py /src/experiments/care/rtmr-training-setting1.json
python get_explanatory_metrics.py /src/experiments/care/rtmr-training-setting1.json

python prepare_dataset_model.py /src/experiments/care/imdb-training-setting1.json --train_model
python check_model_perf.py /src/experiments/care/imdb-training-setting1.json
python get_explanatory_metrics.py /src/experiments/care/imdb-training-setting1.json
