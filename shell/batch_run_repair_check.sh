for i in 1 2 3 4 5 6 7 8; do
for feat in gender; do
bash run_repair_check.sh $i $feat
done
done