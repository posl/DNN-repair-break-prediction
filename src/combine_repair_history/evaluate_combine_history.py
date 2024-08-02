import os, sys
import pickle
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

methods = ["care", "apricot", "arachne"]
datasets = ["credit", "census", "bank", "fm", "c10", "gtsrb", "imdb", "rtmr"]
methods4show = {
    "care": "CARE",
    "apricot": "Apricot",
    "arachne": "Arachne"
}

if __name__ == "__main__":
    # このプログラムのファイル名を取得
    file_name = os.path.splitext(sys.argv[0])[0]

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
                    pp_rate = np.nanmean(tgt_res_arr / org_res_arr)
                    if tgt_method == remained_method:
                        tmp_arr.append(pp_rate)
                    else:
                        mean_perf = np.nanmean(tgt_res_arr)
                        tmp_arr.append(f"{mean_perf:.3f} ({(pp_rate-1):.1%})")
                
                # mAのモデルからremained_method, mBのモデルからremained_methodへのtransf. score
                transferability_dir = os.path.join("/src/experiments/", "method-transferability")
                org_transf_df = pd.read_csv(os.path.join(transferability_dir
                , f"{dataset}-{rb}.csv"))
                transpp_mA2remained = org_transf_df[remained_method][methods.index(mA)]
                transpp_mB2remained = org_transf_df[remained_method][methods.index(mB)]
                transpp_ex = max(transpp_mA2remained, transpp_mB2remained)
                transpp_rate = pp_rate / transpp_ex - 1
                tmp_arr[-1] = f"{tmp_arr[-1]:.3f} ({transpp_rate:.1%})"
                # res_arrに追加
                res_arr.append(tmp_arr)
        # res_arrをndarrayに変換
        res_arr = np.asarray(res_arr)
        print(res_arr)
        print(res_arr.shape)
        # res_arrをcsvで保存
        # 保存先のディレクトリ
        res_dir = os.path.join("/src/src/combine_repair_history")
        res_file = os.path.join(res_dir, f"{rb}.csv")
        np.savetxt(res_file, res_arr, delimiter=",", fmt='%s')