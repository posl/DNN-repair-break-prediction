#!/bin/bash

cd ../src
for ds in credit census bank fm c10 gtsrb rtmr imdb; do
python convert_torch2keras.py $ds
python arachne_localize.py /src/experiments/care/$ds-training-setting1.json
python run_arachne.py $ds
python arachne_check.py /src/experiments/care/$ds-training-setting1.json