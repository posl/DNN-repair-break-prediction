#!/bin/bash

cd ../src
# $1 = exp id, $2 = sensitve feature name
python judge_repair_outcome.py /src/experiments/credit-fairness-$2-setting$1.json