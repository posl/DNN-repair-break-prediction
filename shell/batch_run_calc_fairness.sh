for i in 1 2 3 4; do
for feat in age gender; do
bash run_calc_fairness.sh $i $feat
done
done