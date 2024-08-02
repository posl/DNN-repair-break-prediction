"""
repair method R, dataset Dに対する予測モデルを, 別のmethod R'に対して適用した際の精度を計測する.
Dは標準入力で与える
"""

import os, sys
import pickle
from collections import defaultdict
from itertools import permutations, combinations
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

methods = ["care", "apricot", "arachne"]
methods4show = {
    "care": "CARE",
    "apricot": "Apricot",
    "arachne": "Arachne"
}
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

def get_compatibility(clf1, clf2, X):
    """
    clf1, clf2の予測結果の互換性を計測する
    """
    y1 = clf1.predict(X)
    y2 = clf2.predict(X)
    return np.mean(y1 == y2)

if __name__ == "__main__":
    # このプログラムのファイル名を取得
    file_name = os.path.splitext(sys.argv[0])[0]
    # 対象となるmethod
    dataset = sys.argv[1]
    # 保存先のディレクトリ
    transferability_dir = os.path.join("/src/experiments/", "method-transferability")
    compatibility_dir = os.path.join("/src/experiments/", "method-compatibility")
    # ディレクトリがなければ作成
    if not os.path.exists(transferability_dir):
        os.makedirs(transferability_dir, exist_ok=True)
    if not os.path.exists(compatibility_dir):
        os.makedirs(compatibility_dir, exist_ok=True)

    # 各組み合わせについて実行
    for rb in ["repair", "break"]:
        # 目的変数
        obj_col = "repaired" if rb == "repair" else "broken"
        # transferabilityの結果を保存するdf
        transferability_df = pd.DataFrame(columns=[methods], index=[methods], data=[])
        compatibility_df = pd.DataFrame(columns=[methods], index=[methods], data=[])
        
        for mt_src, mt_tar in permutations(methods, 2):
            print(f"mt_src={mt_src}, mt_tar={mt_tar}, rb={rb}")
            # ディレクトリの指定
            src_model_save_dir = os.path.join(f"/src/experiments/{mt_src}", "repair_break_model") # srcの修正手法のためのモデル
            tar_model_save_dir = os.path.join(f"/src/experiments/{mt_tar}", "repair_break_model") # targetの修正手法のためのモデル
            data_save_dir = os.path.join(f"/src/experiments/{mt_tar}", "repair_break_dataset", "preprocessed_data") # tgtの修正手法のためのモデル
            # datasetに対するmt_srcの予測モデルを読み込み
            src_model_dic = get_models(src_model_save_dir, dataset, rb)
            tar_model_dic = get_models(tar_model_save_dir, dataset, rb)
            # datasetに対するmt_tarのデータを読み込み
            # print(data_save_dir)
            df_test = get_test_df(data_save_dir, mt_tar, dataset, rb)
            X_test, y_test = df_test[exp_metrics], df_test[obj_col]
            # print(X_test.head())
            # ds_srcのモデルでds_tarのテストデータを予測して結果をarrayにまとめる
            src_test_res_arr = []
            tar_test_res_arr = []
            for model_name in ["lr", "rf", "lgb"]:
                src_test_res_arr.append(print_perf(src_model_dic[model_name], X_test, y_test, output_log=False))
                tar_test_res_arr.append(get_compatibility(src_model_dic[model_name], tar_model_dic[model_name], X_test))
            src_test_res_arr = np.asarray(src_test_res_arr)
            tar_test_res_arr = np.asarray(tar_test_res_arr)
            
            # 元々のデータセットでの精度と比較する
            org_res_path = os.path.join(f"/src/experiments/{mt_tar}", "repair_break_model", f"{dataset}-{rb}-test.csv")
            org_res_df = pd.read_csv(org_res_path)
            org_res_arr = org_res_df.values

            # test_res_arrをorg_res_arrで割ることで，originalとの差分を確認
            # test_res_arr, org_res_arrに0があればnanにする
            src_test_res_arr[src_test_res_arr == 0] = np.nan
            org_res_arr[org_res_arr == 0] = np.nan
            # nanを除いて平均してdfに格納
            transferability_df.loc[mt_src, mt_tar] = np.nanmean(src_test_res_arr / org_res_arr)
            # compatibility_dictをdfに追加
            compatibility_df.loc[mt_src, mt_tar] = np.mean(tar_test_res_arr)
        # 結果を保存
        transferability_df.to_csv(os.path.join(transferability_dir, f"{dataset}-{rb}.csv"))
        compatibility_df.to_csv(os.path.join(compatibility_dir, f"{dataset}-{rb}.csv"), index=False)
