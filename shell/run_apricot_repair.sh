#!/bin/bash

cd ../src
for ds in credit census bank fm c10 gtsrb rtmr imdb; do
python rDLM_train.py /src/experiments/care/$ds-training-setting1.json
python apricot_repair.py /src/experiments/care/$ds-training-setting1.json
python apricot_check.py /src/experiments/care/$ds-training-setting1.json
