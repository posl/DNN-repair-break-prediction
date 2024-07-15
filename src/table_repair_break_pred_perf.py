import os, sys, argparse
from ast import literal_eval
from collections import defaultdict
from itertools import product
import pandas as pd
import numpy as np

methods = ["care", "apricot", "arachne", "aprnn"]
priority4show = {method: i for i, method in enumerate(methods)}
method4show = {"care": "CARE", "apricot": "Apricot", "arachne": "Arachne", "aprnn": "APRNN"}
dataset4show = {
    "credit": "Credit",
    "census": "Census",
    "bank": "Bank",
    "c10": "C10",
    "c10c": "C10C",
    "fm": "FM",
    "fmc": "FMC",
    "gtsrb": "GTSRB",
    "imdb": "IMDB",
    "rtmr": "RTMR",
    "acasxu_n2_9_prop8": "N29",
    "acasxu_n3_5_prop2": "N35",
}
perspectives2methods = {
    "correctness": ["care", "apricot", "arachne"],
    "robustness": ["care", "apricot", "arachne"],
    "fairness": ["care", "apricot", "arachne"],
    "safety": ["care", "aprnn"],
}
perspectives2datasets = {
    "correctness": ["credit", "census", "bank", "fm", "c10", "gtsrb", "imdb", "rtmr"],
    "robustness": ["c10c", "fmc"],
    "fairness": ["credit", "census", "bank"],
    "safety": ["acasxu_n2_9_prop8", "acasxu_n3_5_prop2"],  # NOTE: skipping "acasxu_n1_9_prop7"
}
clf = ["lr", "lgb", "rf"]
metrics = ["acc", "precision", "recall", "f1", "roc_auc", "pr_auc"]
used_metrics = ["acc", "precision", "recall", "pr_auc"]
# used_metrics = ["roc_auc", "precision", "recall", "pr_auc"]
# メトリクスごとの棒の色
palette = {"acc": "#999999", "precision": "#ffa500", "recall": "#0059ff", "pr_auc": "#00ff26"}

if __name__ == "__main__":
    # 対象の観点からデータセットと手法を設定
    all_perspective = [
        "correctness",
        "robustness",
        "fairness",
        "safety"
    ]

    for rb in ["repair", "break"]:
        df_all_perspectives = defaultdict(pd.DataFrame)
        for perspective in all_perspective:
            datasets = perspectives2datasets[perspective]
            methods = perspectives2methods[perspective]
            df_all_methods = defaultdict(pd.DataFrame)
            for i, method in enumerate(methods):
                print(rb, perspective, method)
                if method == "aprnn" and rb == "repair" and perspective == "safety":
                    continue
                df = pd.DataFrame(columns=["dataset", "metric", "val", "clf"])
                for dataset in datasets:
                    if method == "care" and rb == "break" and dataset == "acasxu_n2_9_prop8":
                        continue
                    if perspective in ["correctness", "safety"]:
                        path = f"/src/experiments/{method}/repair_break_model/{dataset}-{rb}-test.csv"
                    elif perspective in ["fairness"]:
                        path = f"/src/src/{perspective}/repair_break_model/{dataset}/{method}/{dataset}-{rb}-test.csv"
                    elif perspective in ["robustness"]:
                        if dataset == "fmc":
                            path = f"/src/src/{perspective}/repair_break_model/{dataset}/{method}/{dataset}-{rb}-test.csv"
                        elif dataset == "c10c":
                            # c10c-corruption-type.txtをロードする
                            c10c_corruption_type_dir = "/src/src/robustness/c10c-corruption-type.txt"
                            with open(c10c_corruption_type_dir, "r") as f:
                                c10c_corruption_type = f.read().rstrip("\n")
                            file_names_set = list(literal_eval(c10c_corruption_type))
                            path = [
                                f"/src/src/{perspective}/repair_break_model/{dataset}/{method}/{crp_type}-{rb}-test.csv"
                                for crp_type in file_names_set
                            ]
                    else:
                        raise ValueError(f"Invalid perspective {perspective}")
                    if isinstance(path, str):
                        _df = pd.read_csv(path)[used_metrics]
                        for idx, row in _df.iterrows():
                            for metric in used_metrics:
                                df = df.append(
                                    {
                                        "dataset": dataset4show[dataset],
                                        "metric": metric,
                                        "val": row[metric],
                                        "clf": clf[idx],
                                    },
                                    ignore_index=True,
                                )
                    else: # for C10C
                        assert isinstance(path, list)
                        for p in path:
                            _df = pd.read_csv(p)[used_metrics]
                            for idx, row in _df.iterrows():
                                for metric in used_metrics:
                                    df = df.append(
                                        {
                                            "dataset": dataset4show[dataset],
                                            "metric": metric,
                                            "val": row[metric],
                                            "clf": clf[idx],
                                        },
                                        ignore_index=True,
                                    )
                df["val"] = df["val"].astype("float")
                df = df.pivot_table(index=["dataset", "clf"], columns="metric", values="val").reset_index()
                df['method'] = method
                df_all_methods[method] = df
            # concat the each df in defaultdict
            combined_df = pd.concat(df_all_methods.values())
            # Group by the method and calculate the mean and variance of the metrics
            grouped_df = combined_df.groupby('method').agg(['mean', 'var']).reset_index()
            grouped_df["priority"] = grouped_df["method"].map(priority4show)
            grouped_df = grouped_df.sort_values('priority').drop(columns='priority').reset_index(drop=True)
            # Flatten the columns
            grouped_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped_df.columns.values]
            # Create the list of reordered columns
            reordered_columns = ['method'] + [f"{metric}_mean" for metric in used_metrics] + [f"{metric}_var" for metric in used_metrics]
            # Ensure the method column name is correct and reindex the DataFrame
            grouped_df = grouped_df.loc[:, reordered_columns]
            # Create a new column with the combined mean and variance in the desired format
            for metric in used_metrics:
                grouped_df[metric] = grouped_df.apply(
                    lambda row: f"{row[f'{metric}_mean']:.3f}", axis=1)
                    # lambda row: f"{row[f'{metric}_mean']:.3f} ({row[f'{metric}_var']:.3f})", axis=1)
            # Select only the combined columns for display
            final_df = grouped_df[['method'] + [metric for metric in used_metrics]]
            # print(final_df)
            df_all_perspectives[perspective] = final_df
        combined_df = pd.concat(df_all_perspectives.values())
        print(combined_df)
        combined_df.to_csv(f"./{rb}_pred_model_perf.csv", index=False)