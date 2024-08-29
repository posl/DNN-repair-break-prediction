import os, sys
import re
from collections import defaultdict
from itertools import combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# plot setting
sns.set()

def reorder_method(mA, mB):
    remaining = [m for m in methods if m not in [mA, mB]]
    assert len(remaining) == 1, f"remaining: {remaining}"
    return [mA, mB, remaining[0]]

def get_val_and_percent(cell):
    # cellという文字列にnanが含まれる場合
    if "nan" in cell:
        return np.nan, np.nan
    match = re.match(r"([0-9.]+) \((-?[0-9.]+)%\)", cell)
    if match:
        value = float(match.group(1))
        percentage = float(match.group(2))
        return value, percentage
    else:
        raise ValueError(f"cell: {cell}")

def get_avg_val_and_percent(res_arr):
    values = []
    percentages = []
    for cell in res_arr.flatten():
        value, percentage = get_val_and_percent(cell)
        values.append(value)
        percentages.append(percentage)
    # valuesとpercentagesの形状を変換
    values = np.array(values).reshape(res_arr.shape)
    percentages = np.array(percentages).reshape(res_arr.shape)
    avg_value = np.nanmean(values, axis=0)
    avg_percentage = np.nanmean(percentages, axis=0)
    return avg_value, avg_percentage

methods = ["care", "apricot", "arachne"]
datasets = ["credit", "census", "bank", "fm", "c10", "gtsrb", "imdb", "rtmr"]
methods4show = {
    "care": "CARE",
    "apricot": "Apricot",
    "arachne": "Arachne"
}
perf_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


if __name__ == "__main__":
    # このプログラムのファイル名を取得
    file_name = os.path.splitext(sys.argv[0])[0]
    used_perf_met = str(sys.argv[1])
    print(f"used_perf_met: {used_perf_met}")
    # perf_metricsの中に無かったらエラー終了
    if used_perf_met != "all" and used_perf_met not in perf_metrics:
        raise ValueError(f"used_perf_met: {used_perf_met}")

    for rb in ["repair", "break"]:
        # 結果を保存するarr
        res_arr = []

        for dataset in datasets:
            print(f"rb: {rb}, dataset: {dataset}")
            # 目的変数
            obj_col = "repaired" if rb == "repair" else "broken"
            
            # 学習に使ったmethodのペアごとの繰り返し
            for mA, mB in combinations(methods, 2):
                combine_dir = os.path.join("/src/src/combine_repair_history", f"{mA}-{mB}")
                tmp_arr = []
                reordered_methods = reorder_method(mA, mB)
                remained_method = reordered_methods[-1]
                
                # 予測対象のmethodの繰り返し
                for tgt_method in reordered_methods:
                    print(f"mA: {mA}, mB: {mB}, tgt_method: {tgt_method}")
                    # combined datasetで学習したモデルの予測結果
                    file_name = f"{dataset}-{rb}-test-{tgt_method}.csv"
                    tgt_res_df = pd.read_csv(os.path.join(combine_dir, file_name))
                    tgt_res_arr = tgt_res_df.values
                    # 対象の手法のdatasetで学習したモデルの予測結果
                    org_res_path = os.path.join(f"/src/experiments/{tgt_method}", "repair_break_model", f"{dataset}-{rb}-test.csv")
                    org_res_df = pd.read_csv(org_res_path)
                    org_res_arr = org_res_df.values
                    org_res_arr[org_res_arr == 0] = np.nan # 0除算回避
                    if used_perf_met == "all":
                        pp_rate = np.nanmean(tgt_res_arr / org_res_arr)
                    else:
                        pp_rate = np.nanmean(tgt_res_arr[:, perf_metrics.index(used_perf_met)] / org_res_arr[:, perf_metrics.index(used_perf_met)])
                    if tgt_method == remained_method:
                        tmp_arr.append(pp_rate)
                    else:
                        mean_perf = np.nanmean(tgt_res_arr) if used_perf_met == "all" else np.nanmean(tgt_res_arr[:, perf_metrics.index(used_perf_met)])
                        tmp_arr.append(f"{mean_perf:.3f} ({(pp_rate-1):.1%})") # pp_rate-1 はパーセンテージの増減を表すため
                
                # mAのモデルからremained_method, mBのモデルからremained_methodへのtransf. score
                transferability_dir = os.path.join("/src/experiments/", "method-transferability")
                if used_perf_met == "all":
                    transferability_path = os.path.join(transferability_dir, f"{dataset}-{rb}.csv")
                else:
                    transferability_path = os.path.join(transferability_dir, f"{dataset}-{rb}-{used_perf_met}.csv")
                org_transf_df = pd.read_csv(transferability_path)
                transpp_mA2remained = org_transf_df[remained_method][methods.index(mA)]
                transpp_mB2remained = org_transf_df[remained_method][methods.index(mB)]
                transpp_ex = max(transpp_mA2remained, transpp_mB2remained)
                transpp_rate = pp_rate / transpp_ex - 1
                tmp_arr[-1] = f"{tmp_arr[-1]:.3f} ({transpp_rate:.1%})"
                # res_arrに追加
                res_arr.append(tmp_arr)
        # res_arrをndarrayに変換
        res_arr = np.asarray(res_arr)
        # res_arrの最後の行に平均を追加する
        avg_value, avg_percentage = get_avg_val_and_percent(res_arr)
        last_row = [f"{av:.3f} ({ap:.1f}%)" for av, ap in zip(avg_value, avg_percentage)]
        res_arr = np.vstack([res_arr, last_row])
        res_arr = np.where(np.char.find(res_arr, "nan") != -1, "N/A", res_arr) # nanが含まれるセルは "N/A" に置換
        print(res_arr)
        print(res_arr.shape)
        # res_arrをcsvで保存
        # 保存先のディレクトリ
        res_dir = os.path.join("/src/src/combine_repair_history")
        if used_perf_met == "all":
            res_path = os.path.join(res_dir, f"{rb}.csv")
        else:
            res_path = os.path.join(res_dir, f"{rb}-{used_perf_met}.csv")
        np.savetxt(res_path, res_arr, delimiter=",", fmt='%s')