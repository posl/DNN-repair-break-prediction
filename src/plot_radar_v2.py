import os, sys
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.5)


# 必要なdfを読んできてスケーリングして返す
def get_df(pers, method, dataset, rb):
    obj_col = "repaired" if rb == "repair" else "broken"
    if pers == "correctness" or pers == "safety":
        # 実験のdir名
        exp_dir = f"/src/experiments/{method}"
        # repair/break datasetのファイル名
        if method == "care" or pers == "safety":
            rb_ds_filename = f"{dataset}-fairness-setting1-{rb}.csv"
        elif method == "apricot" or method == "arachne":
            rb_ds_filename = f"{dataset}-training-setting1-{rb}.csv"
        target_path = os.path.join(
            exp_dir, "repair_break_dataset", "raw_data", rb_ds_filename
        )
        df = pd.read_csv(target_path)
    elif pers == "robustness" or pers == "fairness":
        exp_dir = f"/src/src/{pers}"
        target_dir = os.path.join(exp_dir, "rb_datasets", dataset, method)
        # {rb}.csvで終わるcsvファイルを全部読み込んで縦に結合し，1つのdfを作る
        df_list = []
        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.endswith(f"{rb}.csv"):
                    df_list.append(pd.read_csv(os.path.join(root, file)))
        df = pd.concat(df_list)
    elif pers == "safety":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown perspective: {pers}")
    # 説明変数たち
    exp_cols = [col for col in df.columns if col != obj_col]
    # min-max scalingして返す
    scaler = MinMaxScaler()
    df[exp_cols] = scaler.fit_transform(df[exp_cols])
    return df, exp_cols


if __name__ == "__main__":
    rb = sys.argv[1]
    assert rb in ["repair", "break"], "rb must be either 'repair' or 'break'"
    obj_col = "repaired" if rb == "repair" else "broken"

    # 定数たち
    perspectives = [
        # "correctness",
        "robustness",
        "fairness",
        "safety"
    ] # TODO: extend to fairness, robustness, safety
    
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

    # 描画時の表示用のラベル名たち
    expmet4show = {"pcs": "PCS", "lps": "LPS", "entropy": "Ent.", "loss": "Los."}
    dataset4show = {
        "credit": "Credit",
        "census": "Census",
        "bank": "Bank",
        "c10": "C10",
        "fm": "FM",
        "gtsrb": "GTSRB",
        "imdb": "IMDB",
        "rtmr": "RTMR",
        "acasxu_n1_9_prop7": "ACAS N1,9, P7",
        "acasxu_n2_9_prop8": "ACAS N2,9, P8",
        "acasxu_n3_5_prop2": "ACAS N3,5, P2",
    }
    method4show = {"care": "CARE", "apricot": "Apricot", "arachne": "Arachne", "aprnn": "APRNN"}

    # perspectives2methodsのリストの中で最大の長さを設定
    num_col = max([len(methods) for methods in perspectives2methods.values()])
    num_row = len(perspectives)

    # 描画ゾーン
    fig = plt.figure(figsize=(3*num_col, 3*num_row))
    gs = gridspec.GridSpec(num_row, num_col, figure=fig)
    for pi, pers in enumerate(perspectives):
        for mi, method in enumerate(perspectives2methods[pers]):
            print(f"pers={pers}, method={method}")
            # pane_id = num_col * pi + mi + 1
            ax = fig.add_subplot(gs[pi, mi], projection="polar")
            # スペースの都合上, データセットに関しては全てまとめて平均する
            df_list = []
            for ds in perspectives2datasets[pers]:
                _df, exp_cols = get_df(pers, method, ds, rb)
                print(f"_df.shape ({ds}) = {_df.shape}")
                df_list.append(_df)
            df = pd.concat(df_list)
            print(f"df.shape={df.shape}")

            for tf in [True, False]:
                # 描画用の色
                color = "g" if tf else "r"
                # 表示ラベル
                if mi == 0 and pi == 0:
                    label = obj_col if tf else "non-" + obj_col
                else:
                    label = None
                # 目的変数が特定の値のみ取り出し
                df_tf = df[df[obj_col] == tf]
                # 各説明変数の平均を出す
                exp_mean = np.array(df_tf[exp_cols].mean())
                # 描画用のデータ
                values = np.concatenate((exp_mean, [exp_mean[0]]))
                if len(exp_cols) == 4:
                    angles = np.linspace(0.25 * np.pi, 2.25 * np.pi, len(exp_cols) + 1)
                elif len(exp_cols) == 7:
                    angles = np.linspace(0, 2 * np.pi, len(exp_cols) + 1)
                ax.plot(
                    angles, values, "o-", color=color, clip_on=False, zorder=10, label=label
                )
                ax.fill(angles, values, alpha=0.25, color=color)
            ax.set_thetagrids(
                np.rad2deg(angles[:-1]), [expmet4show[col] if col in expmet4show.keys() else col for col in exp_cols], fontsize=15
            )  # theta軸めもり
            ax.set_rlim(0, 1)  # r軸の範囲
            ax.set_rgrids(
                np.linspace(0, 1, num=5), ["0", "", "0.5", "", "1.0"], angle=30, fontsize=10
            )  # r軸めもり
            
            # 観点が1つならそれを表示する必要はない
            if len(perspectives) == 1:
                plt_title = method4show[method]
                bbox_to_anchor_y = 1.10
            else:
                plt_title = f"{method4show[method]}, {pers.capitalize()}"
                bbox_to_anchor_y = 1.06
            if pers == "safety":
                ax.set_title(plt_title, fontsize=15, y=-0.2)
            else:
                ax.set_title(plt_title, fontsize=15)
    fig.legend(bbox_to_anchor=(0.5, bbox_to_anchor_y), loc="lower center", ncol=2, fontsize=15, facecolor='white', edgecolor='none') # cだけなら1.10, rfsなら1.06
    fig.tight_layout()
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.15, wspace=0.18
    )  # この1行を入れる
    # fig.show()
    pers_id = "".join([pers[0] for pers in perspectives])
    save_path = f"./radar_{rb}_{pers_id}.pdf"
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    print(f"saved to {save_path}")
