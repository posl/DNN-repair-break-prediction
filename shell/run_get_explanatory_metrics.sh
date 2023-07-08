#!/bin/bash

for i in 1 2 3 4; do
cd ../src
python get_explanatory_metrics.py /src/experiments/credit-training-setting$i.json
done