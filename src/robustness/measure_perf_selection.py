import os, sys, time, argparse
from ast import literal_eval
import pickle
from collections import defaultdict
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# plot setting
sns.set()

# =====================================================
# 対象とするrepair手法のリスト
methods = ["care", "apricot", "arachne"]
# 対象とするdatasets
datasets = ["fmc", "c10c"]
model_impl = ["lr", "rf", "lgb"]
methods4show = ["CARE", "Apricot", "Arachne", "NOP"]
ID_NOP = 3 # NOPを表すID
# =====================================================

# 与えられた行の最頻値を返す
def make_decision(row):
    uniq, count = np.unique(row, return_counts=True)
    max_count = np.max(count)
    if max_count == 1:
        return np.nan
    return uniq[np.argmax(count)]

if __name__ == "__main__":
    exp_dir = f"/src/src/robustness/"

    # ファイル名の接頭辞を取得
    key_for_ds = defaultdict(set)
    for ds in ["c10c", "fmc"]:
        if ds == "c10c":
            # c10c-corruption-type.txtをロードする
            c10c_corruption_type_dir = "./c10c-corruption-type.txt"
            with open(c10c_corruption_type_dir, "r") as f:
                c10c_corruption_type = f.read().rstrip("\n")
            key_for_ds[ds] = literal_eval(c10c_corruption_type)
        elif ds == "fmc":
            key_for_ds[ds] = set(["fmc"])

    # c1の閾値はコマンドラインで受け取り，のこりはc1から計算
    parser = argparse.ArgumentParser()
    parser.add_argument("--c1_th", type=float, default=0.4, help="Threshold variable for c1")
    parser.add_argument("--c2_th", type=float, default=0.8, help="Threshold variable for c2")
    args = parser.parse_args()
    nop_ths = [args.c1_th, args.c2_th, args.c1_th - (1-args.c2_th)]
    print(f"nop_ths = {nop_ths} ([c1, c2, c3])")
    # model_perfの結果をロード
    save_dir = os.path.join(exp_dir, "method_selection")
    arr = np.loadtxt(os.path.join(save_dir, f"selection_values.csv"), delimiter=",")

    # 3列毎にarrをチェックして, i列目の値が最大ならmethods[i]をセレクションの結果とする
    # NOPの決定のためにCiの最大値も取っておく
    selection_results = np.empty((arr.shape[0], arr.shape[1]//3))
    selection_values = np.empty((arr.shape[0], arr.shape[1]//3))
    for j in range(0, arr.shape[1], 3):
        arr_impl = arr[:, j:j+3]
        selection_results[:, j//3] = np.argmax(arr_impl, axis=1)
        selection_values[:, j//3] = np.max(arr_impl, axis=1)

    # 閾値を下回る場合はNOPにする
    for row_idx, (sv, sr) in enumerate(zip(selection_values, selection_results)):
        ci_idx = row_idx % 3
        th = nop_ths[ci_idx]
        selection_results[row_idx] = np.where(sv < th, ID_NOP, sr)
    selection_results = np.vectorize(lambda x: methods4show[int(x)])(selection_results) # np.vectorizeは関数の入力をベクトルかできるやつらしい


    # 各列の中で最も多い要素を選ぶ
    # majority = np.apply_along_axis(func1d=make_decision, axis=1, arr=selection_results) # awsでやったらなぜか文字列の長さが全部3に統一されちゃった
    majority = []
    for sr in selection_results:
        majority.append(make_decision(sr))
    majority = np.array(majority)

    selection_results = np.concatenate([selection_results, majority.reshape(-1, 1)], axis=1)

    # selection_resultsの中に"nan"が含まれる行のインデックスを取得
    nan_row_idx, _ = np.where(selection_results == "nan")
    scores_for_nan = arr[nan_row_idx]
    # 3列毎にarrをチェックして, i列目の値が最大ならmethods[i]をセレクションの結果とする
    decisions_id = [] # 多数決が決まらなかった各行に対して,どのimplのいうことを聞くか
    for row in scores_for_nan:
        # 各implの自信の配列
        conf_scores = []
        for j in range(0, len(row), 3):
            row_impl = row[j:j+3]
            # row_implの中の最大値と2番目に大きい値の差を取得
            conf_scores.append(np.max(row_impl) - np.sort(row_impl)[-2])
        decisions_id.append(np.argmax(conf_scores))
    # 上のdecisions_idを使ってnanの部分を変更していく
    for i, ni in enumerate(nan_row_idx):
        selection_results[ni, -1] = methods4show[decisions_id[i]]
    print(f"selection_results: {selection_results}")

    # 実際に良かったやつと比較
    # method, datasetごとにtest setをロードしてきてまとめればいいんちゃう
    arr_for_method = [[], [], []]
    for mi, method in enumerate(methods):
        for di, ds in enumerate(datasets):
            rb_save_dir = os.path.join(exp_dir, f"preprocessed_rb_datasets/{ds}/{method}")
            for key in key_for_ds[ds]:
                # 対象のrepairs/breaks datasetsのtestのcsvファイル名
                test_rep = f"{key}-repair-test.csv"
                test_bre = f"{key}-break-test.csv"
                # 必要なデータセットをロードする
                df_rep = pd.read_csv(os.path.join(exp_dir, rb_save_dir, test_rep))
                df_bre = pd.read_csv(os.path.join(exp_dir, rb_save_dir, test_bre))
                # s1: repair_ratio
                repair_ratio = len(df_rep[df_rep["repaired"]==True]) / len(df_rep)
                arr_for_method[mi].append(repair_ratio)
                # s2: retain_ratio
                retain_ratio = len(df_bre[df_bre["broken"]==False]) / len(df_bre)
                arr_for_method[mi].append(retain_ratio)
                break_ratio = len(df_bre[df_bre["broken"]==True]) / len(df_bre)
                # s3: balance
                balance = repair_ratio - break_ratio
                arr_for_method[mi].append(balance)
    correct_arr = np.array(arr_for_method).T
    correct_values = np.max(correct_arr, axis=1)
    correct_results = np.argmax(correct_arr, axis=1)

    # 閾値を下回る場合はNOPにする
    for row_idx, (sv, sr) in enumerate(zip(correct_values, correct_results)):
        ci_idx = row_idx % 3
        th = nop_ths[ci_idx]
        correct_results[row_idx] = np.where(sv < th, ID_NOP, sr)

    correct_results = np.vectorize(lambda x: methods4show[int(x)])(correct_results) # np.vectorizeは関数の入力をベクトルかできるやつらしい
    res_arr = np.concatenate([selection_results, correct_results.reshape(-1, 1)], axis=1)
    np.savetxt(os.path.join(save_dir, f"selection_results_robustness.csv"), res_arr, delimiter=",", fmt="%s")