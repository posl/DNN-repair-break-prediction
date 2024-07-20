import os, sys, time, argparse
import pandas as pd
import numpy as np
from collections import defaultdict

methods = ["care", "apricot", "arachne"]
datasets = ["credit", "census", "bank", "fm", "c10", "gtsrb", "imdb", "rtmr"]
strategies = [None, "smote", "without_resampling"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("measure", type=str, choices=["acc", "f1", "precision", "recall", "pr_auc", "roc_auc"])
    args = parser.parse_args()
    measure = args.measure

    for rb in ["repair", "break"]:
        measure_for_strategies = defaultdict(list)
        for st in strategies:
            for method in methods:
                exp_dir = f"/src/experiments/{method}/repair_break_model"
                for dataset in datasets:
                    file_base_name = f"{dataset}-{rb}-test"
                    if st is None:
                        file_name = f"{file_base_name}.csv"
                    else:
                        file_name = f"{file_base_name}-{st}.csv"
                    file_path = f"{exp_dir}/{file_name}"
                    # load csv file as df
                    df = pd.read_csv(file_path)
                    # add the mean of each of the 3 impl. to the list of the st.
                    measure_for_strategies[st].append(np.mean(df[measure]))
        # ["RUS", "RUS+SMOTE", "w/o resampling"] という3つの列を持つdfを作成
        df_summary = pd.DataFrame(measure_for_strategies)
        df_summary.columns = ["RUS", "RUS+SMOTE", "w/o resampling"]
        save_dir = "./resampling_eff"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df_summary.to_csv(f"{save_dir}/{rb}_{measure}.csv", index=False, float_format="%.3f")

        