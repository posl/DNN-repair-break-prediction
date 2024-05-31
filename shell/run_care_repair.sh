#!/bin/bash

cd ../src
# $1 = exp id, $2 = sensitve feature name

# tabular datasets
echo CARE tabular datasets...
python care_localize.py /src/experiments/care/credit-fairness-gender-setting1.json
python care_repair_fairness.py /src/experiments/care/credit-fairness-gender-setting1.json
python care_check.py /src/experiments/care/credit-fairness-gender-setting1.json
python care_localize.py /src/experiments/care/census-fairness-gender-setting1.json
python care_repair_fairness.py /src/experiments/care/census-fairness-gender-setting1.json
python care_check.py /src/experiments/care/census-fairness-gender-setting1.json
python care_localize.py /src/experiments/care/bank-fairness-age-setting1.json
python care_repair_fairness.py /src/experiments/care/bank-fairness-age-setting1.json
python care_check.py /src/experiments/care/bank-fairness-age-setting1.json

# image datasets
echo CARE image datasets...
python care_localize.py /src/experiments/care/fm-fairness-setting1.json
python care_repair_fairness.py /src/experiments/care/fm-fairness-setting1.json
python care_check.py /src/experiments/care/fm-fairness-setting1.json
python care_localize.py /src/experiments/care/c10-fairness-setting1.json
python care_repair_fairness.py /src/experiments/care/c10-fairness-setting1.json
python care_check.py /src/experiments/care/c10-fairness-setting1.json
python care_localize.py /src/experiments/care/gtsrb-fairness-setting1.json
python care_repair_fairness.py /src/experiments/care/gtsrb-fairness-setting1.json
python care_check.py /src/experiments/care/gtsrb-fairness-setting1.json

# text datasets
echo CARE text datasets...
python care_localize.py /src/experiments/care/rtmr-fairness-setting1.json
python care_repair_fairness.py /src/experiments/care/rtmr-fairness-setting1.json
python care_check.py /src/experiments/care/rtmr-fairness-setting1.json
python care_localize.py /src/experiments/care/imdb-fairness-setting1.json
python care_repair_fairness.py /src/experiments/care/imdb-fairness-setting1.json
python care_check.py /src/experiments/care/imdb-fairness-setting1.json