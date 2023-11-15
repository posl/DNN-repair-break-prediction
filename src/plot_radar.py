import os, sys
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.5)


# 必要なdfを読んできてスケーリングして返す
def get_df(method, dataset, rb):
    obj_col = "repaired" if rb == "repair" else "broken"
    # 実験のdir名
    exp_dir = f"/src/experiments/{method}"
    # repair/break datasetのファイル名
    if method == "care":
        rb_ds_filename = f"{dataset}-fairness-setting1-{rb}.csv"
    elif method == "apricot" or method == "arachne":
        rb_ds_filename = f"{dataset}-training-setting1-{rb}.csv"
    target_path = os.path.join(
        exp_dir, "repair_break_dataset", "raw_data", rb_ds_filename
    )
    # repair/break datasetのcsvをロード
    df = pd.read_csv(target_path)
    # 説明変数たち
    exp_cols = [col for col in df.columns if col != obj_col]
    # min-max scalingして返す
    scaler = MinMaxScaler()
    df[exp_cols] = scaler.fit_transform(df[exp_cols])
    return df, exp_cols


if __name__ == "__main__":
    rb = sys.argv[1]
    obj_col = "repaired" if rb == "repair" else "broken"

    # 定数たち
    # FIXME: 実行する前に以下の変数はチェックすること
    methods = ["care", "apricot", "arachne"]
    datasets = ["credit", "census", "bank", "fm", "c10", "gtsrb"]
    # datasets = ["credit", "census", "bank", "fm"]

    # 描画時の表示用のラベル名たち
    expmet4show = {"pcs": "PCS", "lps": "LPS", "entropy": "Ent.", "loss": "Los."}
    dataset4show = {
        "credit": "Credit",
        "census": "Census",
        "bank": "Bank",
        "c10": "C10",
        "fm": "FM",
        "gtsrb": "GTSRB",
    }
    method4show = {"care": "CARE", "apricot": "Apricot", "arachne": "Arachne"}
    # 描画用の角度たち
    angles = np.linspace(0.25 * np.pi, 2.25 * np.pi, 5)

    num_col = 3
    num_row = 6

    # 描画ゾーン
    fig = plt.figure(figsize=(9, 18))
    for i, (method, dataset) in enumerate(product(methods, datasets)):
        # FIXME: 一時的なif
        if method == "arachne" and dataset in ["gtsrb"]:
            print(method, dataset)
            continue
        # print(method, dataset)
        ax = fig.add_subplot(num_row, num_col, i + 1, projection="polar")
        df, exp_cols = get_df(method, dataset, rb)
        for tf in [True, False]:
            # 描画用の色
            color = "g" if tf else "r"
            # 表示ラベル
            if i == 0:
                label = obj_col if tf else "non-" + obj_col
            else:
                label = None
            # 目的変数が特定の値のみ取り出し
            df_tf = df[df[obj_col] == tf]
            # 各説明変数の平均を出す
            exp_mean = np.array(df_tf[exp_cols].mean())
            # 描画用のデータ
            values = np.concatenate((exp_mean, [exp_mean[0]]))
            ax.plot(
                angles, values, "o-", color=color, clip_on=False, zorder=10, label=label
            )
            ax.fill(angles, values, alpha=0.25, color=color)
        ax.set_thetagrids(
            np.rad2deg(angles[:-1]), [expmet4show[col] for col in exp_cols]
        )  # theta軸めもり
        ax.set_rlim(0, 1)  # r軸の範囲
        ax.set_rgrids(
            np.linspace(0, 1, num=5), ["0", "", "0.5", "", "1.0"], angle=30
        )  # r軸めもり
        ax.set_title(f"{method4show[method]}, {dataset4show[dataset]}")
    fig.legend(bbox_to_anchor=(0.5, -0.03), loc="upper center", ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2
    )  # この1行を入れる
    # fig.show()
    # plt.savefig(f"./radar_{rb}.pdf", bbox_inches="tight")
    # FIXME: 一時的なファイル名
    plt.savefig(f"./radar_tmp_{rb}.pdf", bbox_inches="tight")
