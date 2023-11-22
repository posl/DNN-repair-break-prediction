import os, sys, csv
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from scipy import stats

sns.set(style="whitegrid", font_scale=1.5)


# 必要なdfを読んできて返す
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
    return df

def judge_test_res(p_value, d_cliff):
    # 統計結果の判断基準
    # 有意水準を設定
    sig_005 = 0.05
    sig_001 = 0.01
    # 効果量の閾値
    eff_large = 0.474
    eff_medium = 0.33
    eff_small = 0.147
    # 検定結果を判断して出力を整形
    # 有意水準の判断
    if p_value < sig_001:
        sig = "**"
    elif p_value < sig_005:
        sig = "*"
    else:
        sig = ""
    # 効果量の判断
    if d_cliff >= eff_large:
        eff = "large"
    elif d_cliff >= eff_medium:
        eff = "med."
    elif d_cliff >= eff_small:
        eff = "small"
    else:
        eff = "neg."
    print(eff + sig)
    return sig, eff

if __name__ == "__main__":
    rb = sys.argv[1]
    obj_col = "repaired" if rb == "repair" else "broken"

    # 定数たち
    # FIXME: 実行する前に以下の変数はチェックすること
    methods = ["care", "apricot", "arachne"]
    datasets = ["credit", "census", "bank", "fm", "c10", "gtsrb", "imdb", "rtmr"]

    # 検定するmetricsの順番のリスト
    exp_cols = ["entropy", "pcs", "lps", "loss"]

    # 結果を入れるarr
    res_arr = []


    for i, dataset in enumerate(datasets):
        res_arr_ds = []
        for method in methods:
            print(f"{method}, {dataset}\n==========================")
            # FIXME: 一時的なif
            if method == "arachne" and dataset in ["imdb", "rtmr"]:
                continue
            df = get_df(method, dataset, rb)
            print(df.shape)
            print(df[obj_col].value_counts())
            # repaired/non-repaires (broken/non-broken) に分ける
            df_t = df[df[obj_col] == True]
            df_f = df[df[obj_col] == False]
            # 各説明変数に対してu-testを行う
            for exp_col in exp_cols:
                print(f"test for exp_col: {exp_col}")
                group_t, group_f = df_t[exp_col], df_f[exp_col]
                statistic, p_value = stats.mannwhitneyu(group_t, group_f, alternative="greater")
                # 効果量の算出
                d_cliff = - (2*statistic/(len(group_t)*len(group_f)) - 1)
                d_cliff = np.abs(d_cliff)
                sig, eff = judge_test_res(p_value, d_cliff)
                res_arr_ds.append(eff + sig)
        res_arr.append(res_arr_ds)
    # CSVファイルへの書き込み
    file_path = f"mannwhitneyu_{rb}.csv"
    with open(file_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(res_arr)
