import os, sys, time
from ast import literal_eval
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
import pickle
import pandas as pd
import numpy as np

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

# 与えられた行の最頻値を返す
def make_decision(row):
    uniq, count = np.unique(row, return_counts=True)
    max_count = np.max(count)
    if max_count == 1:
        return np.nan
    return uniq[np.argmax(count)]


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
methods = ["care", "aprnn"]
# 対象とするdatasets
datasets = ["acasxu_n3_5_prop2"]
model_impl = ["lr", "rf", "lgb"]
num_pers = 1 # 修正手法の選択基準の数. 今回はbroken rateだけなので1.
# =====================================================

if __name__ == "__main__":

    res_arr = []
    for di, ds in enumerate(datasets):
            arr_for_ds = [[] for _ in methods]
            for mi, method in enumerate(methods):
                print(f"ds = {ds}, method = {method}")
                exp_dir = f"/src/experiments/{method}"
                model_save_dir = os.path.join(exp_dir, f"repair_break_model")
                rb_save_dir = os.path.join(exp_dir, f"repair_break_dataset/preprocessed_data")
                # 対象のrepairs/breaks datasetsのtestのcsvファイル名
                test_bre = f"{ds}-fairness-setting1-break-test.csv"
                # 必要なデータセットをロードする
                df_bre = pd.read_csv(os.path.join(rb_save_dir, test_bre))
                arr_for_ds_method = np.zeros((num_pers, len(model_impl))) # 各データセット, methodごとの, s1, s2, s3の指標の値 (行) をmodel implごとに (列) 入れる配列
                # 学習済みのpred modelのロード
                for ii, impl in enumerate(model_impl):
                    bre_model_dir = os.path.join(model_save_dir, f"{ds}-break-{impl}.pkl")
                    with open(bre_model_dir, "rb") as f:
                        clf_bre = pickle.load(f)
                    # メトリクスの計算
                    y_pred_bre, perf_bre = print_perf_with_output(clf_bre, df_bre.drop("broken", axis=1), df_bre["broken"])
                    # 全体-brokenのratioの見込み数 (retained ration)
                    pre_retained = (len(y_pred_bre) - np.sum(y_pred_bre)) / len(y_pred_bre)
                    arr_for_ds_method[0][ii] = pre_retained
                arr_for_ds[mi] = arr_for_ds_method
            res_arr.append(np.concatenate(arr_for_ds, axis=1)) # methodで横に繋げる (列の長さが *methodsの数 になる)
    res_arr = np.concatenate(res_arr, axis=0) # datasetで縦につなげる (行の長さが *datasetsの数 になる)
    # 列がLRに対する各メソッドの値, RF..., LGB,...となるように列の順番を入れ替える
    new_order = []
    for start_idx in [0, 1, 2]:
        new_order += [i for i in range(start_idx, len(model_impl)*len(methods), len(model_impl))]
    res_arr = res_arr[:, new_order]
    # res_arrをcsvで保存
    res_save_dir = os.path.join("./method_selection")
    os.makedirs(res_save_dir, exist_ok=True)
    print(f"res_arr.shape = {res_arr.shape}")
    np.savetxt(os.path.join(res_save_dir, f"selection_values.csv"), res_arr, delimiter=",")

    ##################################################
    # 以降で上で求めた値から最適なメソッドを選択する #
    ##################################################
    
    # len(methods)列毎にarrをチェックして, i列目の値が最大ならmethods[i]をセレクションの結果とする
    selection_results = np.empty((res_arr.shape[0], res_arr.shape[1]//len(methods)))
    for j in range(0, res_arr.shape[1], len(methods)):
        arr_impl = res_arr[:, j:j+len(methods)]
        selection_results[:, j//len(methods)] = np.argmax(arr_impl, axis=1)

    # selection_resultsの各要素をインデックスとしてmethodsに置き換える
    methods4show = ["CARE", "APRNN"]
    selection_results = np.vectorize(lambda x: methods4show[int(x)])(selection_results) # np.vectorizeは関数の入力をベクトルかできるやつらしい
    # 各列の中で最も多い要素を選ぶ
    majority = np.apply_along_axis(func1d=make_decision, axis=1, arr=selection_results)
    selection_results = np.concatenate([selection_results, majority.reshape(-1, 1)], axis=1)
    # 多数決で決まらなかった (4列目がnan) のものは，スコアの自信（最大値-2番目に大きい値）をとってそれが最大の分類器のいうことを聞く
    # selection_resultsの中に"nan"が含まれる行のインデックスを取得
    nan_row_idx, _ = np.where(selection_results == "nan")
    scores_for_nan = res_arr[nan_row_idx]
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
    
    # 実際に良かったやつと比較
    # method, datasetごとにtest setをロードしてきてまとめればいいんちゃう
    arr_for_method = [[] for _ in methods]
    for mi, method in enumerate(methods):
        for di, ds in enumerate(datasets):
            exp_dir = f"/src/experiments/{method}"
            model_save_dir = os.path.join(exp_dir, f"repair_break_model")
            rb_save_dir = os.path.join(exp_dir, f"repair_break_dataset/preprocessed_data")
            # 対象のrepairs/breaks datasetsのtestのcsvファイル名
            test_bre = f"{ds}-fairness-setting1-break-test.csv"
            # 必要なデータセットをロードする
            df_bre = pd.read_csv(os.path.join(exp_dir, rb_save_dir, test_bre))
            # s2: retain_ratio
            retain_ratio = len(df_bre[df_bre["broken"]==False]) / len(df_bre)
            arr_for_method[mi].append(retain_ratio)
    correct_arr = np.array(arr_for_method).T
    print(correct_arr.shape)
    correct_arr = np.argmax(correct_arr, axis=1)
    print(correct_arr.shape)
    correct_arr = np.vectorize(lambda x: methods4show[int(x)])(correct_arr)
    print(selection_results.shape)
    print(correct_arr.shape)
    res_arr = np.concatenate([selection_results, correct_arr.reshape(-1, 1)], axis=1)
    print(f"res_arr.shape = {res_arr.shape}")
    np.savetxt(os.path.join(res_save_dir, f"selection_results.csv"), res_arr, delimiter=",", fmt="%s")