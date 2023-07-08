for i in 1 2 3 4; do
for feat in gender; do
bash run_repair.sh $i $feat
done
done