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
    sign = ""
    if d_cliff > 0:
        sign = "(+) "
    elif d_cliff < 0:
        sign = "(-) "
    d_cliff = np.abs(d_cliff)
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
    return sig, eff, sign

if __name__ == "__main__":
    rb = sys.argv[1]
    obj_col = "repaired" if rb == "repair" else "broken"

    # 定数たち
    methods = ["care", "apricot", "arachne"]
    datasets = ["credit", "census", "bank", "fm", "c10", "gtsrb", "imdb", "rtmr"]

    # 検定するmetricsの順番のリスト
    exp_cols = ["entropy", "pcs", "lps", "loss"]

    # 結果を入れるarr
    res_arr = []
    res_arr_tex = [] # tex用
    mean_arr = []
    mean_std_arr = []
    eff_size_arr = []

    alternative = "two-sided"


    for i, dataset in enumerate(datasets):
        mean_arr_ds = []
        mean_std_arr_ds = []
        res_arr_ds = []
        res_arr_ds_tex = []
        eff_size_arr_ds = []
        for method in methods:
            print(f"{method}, {dataset}\n==========================")
            df = get_df(method, dataset, rb)
            print(df.shape)
            print(df[obj_col].value_counts())
            # repaired/non-repaires (broken/non-broken) に分ける
            df_t = df[df[obj_col] == True]
            df_f = df[df[obj_col] == False]
            
            # 検定と効果量の前に平均と分散も計算しておきたい
            print("conducting mean and std calculation...")
            mean_t, std_t = df_t.describe().loc["mean"], df_t.describe().loc["std"]
            mean_f, std_f = df_f.describe().loc["mean"], df_f.describe().loc["std"]
            for exp_col in exp_cols:
                entry_t = f"{mean_t[exp_col]:.2f} ({std_t[exp_col]:.2f})"
                entry_f = f"{mean_f[exp_col]:.2f} ({std_f[exp_col]:.2f})"
                mean_std_arr_ds.extend([entry_t, entry_f])
                mean_arr_ds.extend([f"{mean_t[exp_col]:.2f}", f"{mean_f[exp_col]:.2f}"])

            # 各説明変数に対してu-testを行う
            print("conducting mann-whitneyu test...")
            for exp_col in exp_cols:
                print(f"test for exp_col: {exp_col}")
                group_t, group_f = df_t[exp_col], df_f[exp_col]
                statistic, p_value = stats.mannwhitneyu(group_t, group_f, alternative=alternative)
                # 効果量の算出
                d_cliff = 2*statistic/(len(group_t)*len(group_f)) - 1
                sig, eff, sign = judge_test_res(p_value, d_cliff)
                res_arr_ds.append(sign + eff + sig)
                # tex用の処理
                if sign == "(+) ":
                    color = "ForestGreen"
                elif sign == "(-) ":
                    color = "red"
                else:
                    color = "white"
                res_arr_ds_tex.append(f"\\textcolor{{{color}}}{{{eff + sig}}}")
                eff_size_arr_ds.append(d_cliff)
        
        mean_arr.append(mean_arr_ds)
        mean_std_arr.append(mean_std_arr_ds)
        res_arr.append(res_arr_ds)
        res_arr_tex.append(res_arr_ds_tex)
        eff_size_arr.append(eff_size_arr_ds)
    # CSVファイルへの書き込み
    res_path = f"mannwhitneyu_{rb}_{alternative}.csv"
    res_path_tex = f"mannwhitneyu_{rb}_{alternative}_tex.csv"
    mean_std_path = f"mean_std_{rb}.csv"
    mean_path = f"mean_{rb}.csv"
    eff_size_arr_path = f"eff_size_{rb}.csv"
    for file_path, arr in zip([res_path, res_path_tex, mean_std_path, mean_path, eff_size_arr_path], [res_arr, res_arr_tex, mean_std_arr, mean_arr, eff_size_arr]):
        with open(file_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(arr)
