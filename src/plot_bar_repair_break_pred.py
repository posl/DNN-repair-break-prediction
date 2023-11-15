import os, sys
from collections import defaultdict
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=2)

methods = ["care", "apricot", "arachne"]
method4show = {"care": "CARE", "apricot": "Apricot", "arachne": "Arachne"}
datasets = ["credit", "census", "bank", "fm", "c10", "gtsrb"]
dataset4show = {
    "credit": "Credit",
    "census": "Census",
    "bank": "Bank",
    "c10": "C10",
    "fm": "FM",
    "gtsrb": "GTSRB",
}
clf = ["lr", "lgb", "rf"]
metrics = ["acc", "precision", "recall", "f1", "roc_auc", "pr_auc"]
used_metrics = ["acc", "precision", "recall", "roc_auc"]

if __name__ == "__main__":
    # clfのうち最も精度の高い分類器の結果をグラフかしたい
    for rb in ["repair", "break"]:
        fig = plt.figure(figsize=(18, 6), facecolor="w")
        for i, method in enumerate(methods):
            df = pd.DataFrame(columns=["dataset", "metric", "val", "clf"])
            ax = fig.add_subplot(1, len(methods), i + 1)  # メソッドごとにサブプロットを追加
            for dataset in datasets:
                # print(method, dataset, rb)
                if method == "arachne" and dataset in ["c10", "gtsrb"]:
                    continue
                path = f"/src/experiments/{method}/repair_break_model/{dataset}-{rb}-test.csv"
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
            # 全データセット終わったら描画する
            palette = sns.color_palette(n_colors=len(used_metrics))
            plt.xticks(rotation=45)
            _error_bar = lambda x: (x.min(), x.max()) # エラーバーの表示方法. "se"なら標準誤差, "sd"なら標準偏差になるが今回は最小から最大値までの範囲を示すカスタムのエラーバーにする.
            _estimator = "median" # 中央値をとる. default: "mean"で平均
            sns.barplot(
                data=df, x="dataset", y="val", hue="metric", palette=palette, ax=ax, errorbar=_error_bar, estimator=_estimator
            )
            ax.set_title(f"{rb.capitalize()+'s'} pred. models ({method4show[method]})")
            ax.set_xlabel("Datasets")
            if i == 0:
                ax.set_ylabel("Val. of metrics")
            else:
                ax.set_ylabel("")
            ax.set_ylim(0, 1)
            ax.legend().set_visible(False)
        # plt.legend(loc="upper left", bbox_to_anchor=(1,1))
        plt.legend(
            bbox_to_anchor=(-0.7, -0.25), loc="upper center", ncol=len(used_metrics)
        )
        fig.tight_layout()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.2)
        # plt.show()
        plt.savefig(f"./bar_pred_perf_{rb}.pdf", bbox_inches="tight", dpi=300)
