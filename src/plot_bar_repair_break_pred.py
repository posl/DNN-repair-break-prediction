import os, sys, argparse
from ast import literal_eval
from collections import defaultdict
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=2)

methods = ["care", "apricot", "arachne", "aprnn"]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("perspective", type=str, help="all, correctness, robustness, fairness, safety", choices=["all", "correctness", "robustness", "fairness", "safety"])
    args = parser.parse_args()
    # 対象の観点からデータセットと手法を設定
    perspective = args.perspective
    datasets = perspectives2datasets[perspective]
    methods = perspectives2methods[perspective]

    # clfのうち最も精度の高い分類器の結果をグラフかしたい
    for rb in ["repair", "break"]:
        fig = plt.figure(figsize=(18, 4), facecolor="w")
        for i, method in enumerate(methods):
            print(rb, method, perspective)
            if method == "aprnn" and rb == "repair" and perspective == "safety":
                continue
            df = pd.DataFrame(columns=["dataset", "metric", "val", "clf"])
            ax = fig.add_subplot(1, len(methods), i + 1)  # メソッドごとにサブプロットを追加
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
                else:
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
                # # 新たな行を追加
                # max_arr = np.max(_df.values, axis=0)  # 最も良かった分類器のメトリクスだけを取得
                # new_rows = np.array(
                #     [[dataset4show[dataset]] * len(used_metrics), used_metrics, max_arr]
                # ).T
                # new_entry = pd.DataFrame(
                #     data=new_rows, columns=["dataset", "metric", "val"]
                # )
                # df = pd.concat([df, new_entry])
            df["val"] = df["val"].astype("float")
            print(df)
            # 全データセット終わったら描画する
            plt.xticks(rotation=45)
            _error_bar = lambda x: (x.min(), x.max()) # エラーバーの表示方法. "se"なら標準誤差, "sd"なら標準偏差になるが今回は最小から最大値までの範囲を示すカスタムのエラーバーにする.
            _estimator = "median" # 中央値をとる. default: "mean"で平均
            sns.barplot(
                data=df, x="dataset", y="val", hue="metric", palette=palette, ax=ax, errorbar=_error_bar, errcolor="black", estimator=_estimator
            )
            ax.set_title(f"{rb.capitalize()+'s'} pred. models ({method4show[method]})")
            if i == 0:
                ax.set_ylabel("Val. of metrics")
                ax.set_xlabel("Datasets")
            else:
                ax.set_ylabel("")
                ax.set_xlabel("")
            ax.set_ylim(0, 1)
            ### spine setting
            ax.spines['top'].set_linewidth(0)
            ax.spines['right'].set_linewidth(0)
            ax.spines['left'].set_linewidth(2)
            ax.spines['left'].set_color('gray')
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['bottom'].set_color('gray')
            ax.legend().set_visible(False)
        # plt.legend(loc="upper left", bbox_to_anchor=(1,1))
        plt.legend(
            bbox_to_anchor=(-0.25, -0.25), loc="upper center", ncol=len(used_metrics), facecolor='white', edgecolor='none'
        )
        fig.tight_layout()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.2)
        # plt.show()
        plt.savefig(f"./bar_pred_perf_{rb}_{perspective[0]}.pdf", bbox_inches="tight", dpi=300)
