#!/bin/bash
for ds in credit census bank fm c10 gtsrb; do
echo "START $ds..."
python build_repair_break_model.py $ds
echo "DONE $ds..."
done
