import os, sys, time
import pickle
from collections import defaultdict
from itertools import product
import pandas as pd
import numpy as np
# from imblearn.over_sampling import SMOTE
from lib.log import set_exp_logging
import matplotlib.pyplot as plt
import seaborn as sns

# plot setting
sns.set()

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    precision_recall_curve,
)
import lightgbm as lgb


def calc_aucs(clf, X, y):
    """モデルのデータセットに対するROC-AUC, PR-AUCを計算して返す

    Args:
        clf : sklearn等で作成した学習済みモデル
        X (array-like): データセット
        y (array-like): Xに対応するラベル

    Returns:
        roc_auc, pr_auc: ROC-AUC, PR-AUCの値
    """
    pred = clf.predict(X).astype(int)
    proba = clf.predict_proba(X)
    # ROC-AUCの計算
    fpr, tpr, _ = roc_curve(y_true=y, y_score=proba[:, 1])
    roc_auc = auc(fpr, tpr)
    # PR-AUCの計算
    pr, re, _ = precision_recall_curve(y_true=y, probas_pred=proba[:, 1])
    pr_auc = auc(re, pr)
    return roc_auc, pr_auc


def print_perf_with_output(clf, X, y):
    """モデルのデータセットに対する各種メトリクス (acc, pre, rec, f1, roc-auc, pr-auc) を表示する

    Args:
        clf : sklearn等で作成した学習済みモデル
        X (array-like): データセット
        y (array-like): Xに対応するラベル

    Returns:
        メトリクスのリスト
    """
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc, pr_auc = calc_aucs(clf, X, y)
    return y_pred, np.array([acc, precision, recall, f1, roc_auc, pr_auc])

# 空のリストで初期化する関数
def initialize_list_cell():
    return []


# =====================================================
# 対象とするrepair手法のリスト
methods = ["care", "apricot", "arachne"]
# 対象とするdatasets
datasets = ["credit", "census", "bank", "fm", "c10", "gtsrb", "imdb", "rtmr"]
model_impl = ["lr", "rf", "lgb"]
# =====================================================

if __name__ == "__main__":
    # このプログラムのファイル名を取得
    file_name = os.path.splitext(sys.argv[0])[0]

    res_arr = []
    for di, dataset in enumerate(datasets):
        arr_for_ds = [[], [], []]
        for mi, method in enumerate(methods):
            # 実験のディレクトリと実験名を取得
            exp_dir = f"/src/experiments/{method}"
            save_dir = os.path.join(exp_dir, "repair_break_model")
            print(f"method = {method}")
            # 対象のrepairs/breaks datasetsのtestのcsvファイル名
            if method == "care":
                test_rep = f"{dataset}-fairness-setting1-repair-test.csv"
                test_bre = f"{dataset}-fairness-setting1-break-test.csv"
            elif method == "apricot" or method == "arachne":
                test_rep = f"{dataset}-training-setting1-repair-test.csv"
                test_bre = f"{dataset}-training-setting1-break-test.csv"
            # 必要なデータセットをロードする
            df_rep = pd.read_csv(os.path.join(exp_dir, "repair_break_dataset/preprocessed_data", test_rep))
            df_bre = pd.read_csv(os.path.join(exp_dir, "repair_break_dataset/preprocessed_data", test_bre))
            arr_for_ds_method = np.zeros((3, len(model_impl))) # 各データセット, methodごとの, s1, s2, s3の指標の値 (行) をmodel implごとに (列) 入れる配列
            # 学習済みのpred modelのロード
            for ii, impl in enumerate(model_impl):
                rep_model_dir = os.path.join(save_dir, f"{dataset}-repair-{impl}.pkl")
                with open(rep_model_dir, "rb") as f:
                    clf_rep = pickle.load(f)
                bre_model_dir = os.path.join(save_dir, f"{dataset}-break-{impl}.pkl")
                with open(bre_model_dir, "rb") as f:
                    clf_bre = pickle.load(f)
                # メトリクスの計算
                y_pred_rep, perf_rep = print_perf_with_output(clf_rep, df_rep.drop("repaired", axis=1), df_rep["repaired"])
                y_pred_bre, perf_bre = print_perf_with_output(clf_bre, df_bre.drop("broken", axis=1), df_bre["broken"])
                # repaired ratioの見込み数
                pre_repaired = np.sum(y_pred_rep) / len(y_pred_rep)
                arr_for_ds_method[0][ii] = pre_repaired
                # 全体-brokenのratioの見込み数 (retained ration)
                pre_retained = (len(y_pred_bre) - np.sum(y_pred_bre)) / len(y_pred_bre)
                arr_for_ds_method[1][ii] = pre_retained
                # repair_ratio - broken_ratio
                ratio = (np.sum(y_pred_rep) / len(y_pred_rep)) - (np.sum(y_pred_bre) / len(y_pred_bre))
                arr_for_ds_method[2][ii] = ratio
            arr_for_ds[mi] = arr_for_ds_method
        res_arr.append(np.concatenate(arr_for_ds, axis=1)) # methodで横に繋げる (列の長さが *methodsの数 になる)
    res_arr = np.concatenate(res_arr, axis=0) # datasetで縦につなげる (行の長さが *datasetsの数 になる)
    # 列がLRに対する各メソッドの値, RF..., LGB,...となるように列の順番を入れ替える
    new_order = []
    for start_idx in [0, 1, 2]:
        new_order += [i for i in range(start_idx, len(model_impl)*len(methods), len(model_impl))]
    res_arr = res_arr[:, new_order]
    # res_arrをcsvで保存
    save_dir = "/src/experiments/method-selection"
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(os.path.join(save_dir, f"selection_values.csv"), res_arr, delimiter=",")