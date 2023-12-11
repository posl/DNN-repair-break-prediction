"""
repair method R, dataset Dに対する予測モデルを, 別のデータセットD'に対して適用した際の精度を計測する.
Rは標準入力で与える
"""

import os, sys
import pickle
from collections import defaultdict
from itertools import permutations
import pandas as pd
import numpy as np
from lib.log import set_exp_logging
import matplotlib.pyplot as plt
import seaborn as sns
# plot setting
sns.set()

# build_repair_break_model.pyからprint_perfをインポート
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from build_repair_break_model import print_perf

datasets = ["credit", "census", "bank", "fm", "c10", "gtsrb", "imdb", "rtmr"]
exp_metrics = ["pcs", "lps", "loss", "entropy"]

def get_models(model_dir, ds, rb):
    model_dic = {}
    for model_name in ["lr", "rf", "lgb"]:
        model_path = os.path.join(model_dir, f"{ds}-{rb}-{model_name}.pkl")
        with open(model_path, "rb") as f:
            model_dic[model_name] = pickle.load(f)
    return model_dic


def get_test_df(data_dir, method, ds, rb):
    # 対象のcsvファイル名
    if method == "care":
        test_csv_name = f"{ds}-fairness-setting1-{rb}-test.csv"
    elif method == "apricot" or method == "arachne":
        test_csv_name = f"{ds}-training-setting1-{rb}-test.csv"
    # 必要なデータセットをロードする
    test_ds_dir = os.path.join(data_dir, test_csv_name)
    return pd.read_csv(test_ds_dir)

if __name__ == "__main__":
    # このプログラムのファイル名を取得
    file_name = os.path.splitext(sys.argv[0])[0]
    # 対象となるmethod
    method = sys.argv[1]
    # ディレクトリの指定
    exp_dir = f"/src/experiments/{method}"
    model_save_dir = os.path.join(exp_dir, "repair_break_model")
    data_save_dir = os.path.join(exp_dir, "repair_break_dataset", "preprocessed_data")
    transferability_dir = os.path.join(exp_dir, "transferability")
    # ディレクトリがなければ作成
    if not os.path.exists(transferability_dir):
        os.makedirs(transferability_dir, exist_ok=True)

    # 各組み合わせについて実行
    for rb in ["repair", "break"]:
        # 目的変数
        obj_col = "repaired" if rb == "repair" else "broken"
        # transferabilityの結果を保存するdf
        transferability_df = pd.DataFrame(columns=[datasets], index=[datasets], data=[])
        for ds_src, ds_tar in permutations(datasets, 2):
            print(f"ds_src={ds_src}, ds_tar={ds_tar}, rb={rb}")
            # ds_srcの予測モデルを読み込み
            model_dic = get_models(model_save_dir, ds_src, rb)
            # ds_tarのデータを読み込み
            df_test = get_test_df(data_save_dir, method, ds_tar, rb)
            X_test, y_test = df_test[exp_metrics], df_test[obj_col]
            # ds_srcのモデルでds_tarのテストデータを予測して結果をarrayにまとめる
            test_res_arr = []
            for model_name in ["lr", "rf", "lgb"]:
                test_res_arr.append(print_perf(model_dic[model_name], X_test, y_test, output_log=False))
            test_res_arr = np.asarray(test_res_arr)
            
            # 元々のデータセットでの精度と比較する
            org_res_path = os.path.join(model_save_dir, f"{ds_tar}-{rb}-test.csv")
            org_res_df = pd.read_csv(org_res_path)
            org_res_arr = org_res_df.values

            # test_res_arrをorg_res_arrで割ることで，originalとの差分を確認
            # test_res_arr, org_res_arrに0があればnanにする
            test_res_arr[test_res_arr == 0] = np.nan
            org_res_arr[org_res_arr == 0] = np.nan
            # nanを除いて平均してdfに格納
            transferability_df.loc[ds_src, ds_tar] = np.nanmean(test_res_arr / org_res_arr)
        # 結果を保存
        transferability_df.to_csv(os.path.join(transferability_dir, f"{file_name}-{rb}.csv"))
