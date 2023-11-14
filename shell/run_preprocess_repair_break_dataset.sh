#!/bin/bash

cd ../src
method="arachne"

for ds in credit census bank fm; do
# $1 = method, $2 = dataset
python preprocess_repair_break_dataset.py $method $ds
done