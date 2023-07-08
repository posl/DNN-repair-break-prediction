import os, sys
import pickle
from collections import defaultdict
from itertools import product
import pandas as pd
import numpy as np
from lib.util import json2dict
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


def print_perf(clf, X, y):
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
    logger.info(f"Accuracy: {acc:.3}")
    logger.info(f"Precision: {precision:.3}")
    logger.info(f"Recall: {recall:.3}")
    logger.info(f"F1: {f1:.3}")
    logger.info(f"ROC-AUC: {roc_auc:.3}")
    logger.info(f"PR-AUC: {pr_auc:.3}")
    return np.array([acc, precision, recall, f1, roc_auc, pr_auc])


def plot_conf_mat(clf, X, y):
    """モデルのデータセットに対するconfusion matrixを表示する

    Args:
        clf : sklearn等で作成した学習済みモデル
        X (array-like): データセット
        y (array-like): Xに対応するラベル
    """
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, square=True, cbar=True, annot=True, cmap="Reds", fmt="d")
    plt.ylim(0, cm.shape[0])
    plt.ylabel("true")
    plt.xlabel("pred")
    plt.show()


def under_sampling(df, target_column, minority=True, sample_size_ratio=1):
    """dfのある列target_columnにおいて, 少数派の値minorityの行数にmajorityの行数を合わせたdfを返す

    Args:
        df (pd.DataFrame) :対象となるdf
        target_column (string) : 対象となる列名
        minority (boolean) : 少数派のラベル（True, Falseのみ）
        sample_size_ratio (float) : 多数派のラベルのデータ数を，少数派のラベルのデータ数の何倍にするか

    Returns:
        pd.DataFrame : 間引いた後のdf
    """
    # 少ない方の行数を取得
    df_only_minority = df[df[target_column] == minority]
    minority_cnt = df_only_minority.shape[0]
    # 多い方の行から，少ない方の数 * sample_size_ratioだけサンプリング
    df_only_majority = df[df[target_column] != minority]
    sample_size = int(sample_size_ratio * minority_cnt)
    df_sampled = df_only_majority.sample(n=sample_size, random_state=629)
    return pd.concat([df_only_minority, df_sampled])


# TODO: これらの定数を外部化
# =====================================================
# 対象とするdivision (train, repair, testのいずれか)
divisions = ["train", "repair", "test"]
# 対象とするdataset
# dataset = "credit"
# dataset = "census"
# dataset = "bank"
# 対象とする観点
task_name = "correctness"
# fairnessの場合のsensitive feature
sens_name = "gender"
# sens_name = "age"
# fold数
num_folds = 5
# 説明変数の列名のリスト
exp_metrics = ["pcs", "lps", "loss", "entropy"]
# =====================================================


if __name__ == "__main__":
    # このプログラムのファイル名を取得
    file_name = os.path.splitext(sys.argv[0])[0]
    # 対象となるデータセット
    dataset = sys.argv[1]
    # 実験のディレクトリと実験名を取得
    exp_dir = "/src/experiments"
    # ログファイルの生成
    logger = set_exp_logging(exp_dir, f'{dataset}-{file_name.replace("_", "-")}')
    logger.info(f"dataset={dataset}, task_name={task_name}")
    save_dir = os.path.join(exp_dir, "repair_break_model")
    os.makedirs(save_dir, exist_ok=True)

    for rb in ["repair", "break"]:
        logger.info(f"repair or break = {rb}")
        obj_col = "repaired" if rb == "repair" else "broken"
        # 対象のcsvファイル名
        if task_name == "correctness":
            train_csv_name = f"{dataset}-{task_name}-{rb}-trainval.csv"
            test_csv_name = f"{dataset}-{task_name}-{rb}-test.csv"
        elif task_name == "fairness":
            train_csv_name = f"{dataset}-{task_name}-{sens_name}-{rb}-trainval.csv"
            test_csv_name = f"{dataset}-{task_name}-{sens_name}-{rb}-test.csv"
        # 必要なデータセットをロードする
        train_ds_dir = os.path.join(exp_dir, "repair_break_dataset/preprocessed_data", train_csv_name)
        test_ds_dir = os.path.join(exp_dir, "repair_break_dataset/preprocessed_data", test_csv_name)
        df_train = pd.read_csv(train_ds_dir)
        df_test = pd.read_csv(test_ds_dir)
        logger.info(f"df_train.shape={df_train.shape}, df_test.shape={df_test.shape}")
        df_train = under_sampling(df_train, obj_col, True, 1)
        logger.info(f"(AFTER USAMP.) df_train.shape={df_train.shape}, df_test.shape={df_test.shape}")

        # 説明変数と目的変数に分割
        X_train, y_train = df_train[exp_metrics], df_train[obj_col]
        X_test, y_test = df_test[exp_metrics], df_test[obj_col]
        # foldに分割する準備
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=1234)

        # 学習前の設定
        # グリッドサーチで試すパラメータの集合は，モデルの種類によって固定する．repair/side-effで変えない
        parameters_dict = {
            "lr": {"C": [10**i for i in range(-2, 4)], "random_state": [1234]},
            "lgb": {
                "max_depth": [10, 25, 50, 75],
                "learning_rate": [0.001, 0.01, 0.05, 0.1],
                "n_estimators": [10, 50, 100, 200, 500],
            },
            "rf": {"n_estimators": [70, 80, 90, 100], "max_depth": [5, 10, 50]},
        }
        # verboseを事前に設定
        verbose_level = 1
        # 結果保存用のcsvファイル名
        if task_name == "correctness":
            res_filename = f"{dataset}-{task_name}-{rb}.csv"
        elif task_name == "fairness":
            res_filename = f"{dataset}-{task_name}-{sens_name}-{rb}.csv"
        res_save_path = os.path.join(save_dir, res_filename)
        res_arr = []

        # 3種類の分類器をそれぞれcross-validation+grid searchで訓練して結果を表示
        for cname in ["lr", "lgb", "rf"]:
            logger.info(f"{rb} model: {cname}")
            # 分類器の選択
            if cname == "lr":
                clf = LogisticRegression()
            elif cname == "lgb":
                clf = lgb.LGBMClassifier(random_state=1234)
            elif cname == "rf":
                clf = RandomForestClassifier(criterion="entropy", random_state=1234)
            grid = GridSearchCV(
                clf, param_grid=parameters_dict[cname], cv=kf, scoring="accuracy", verbose=verbose_level
            )
            # 訓練の実行
            grid.fit(X_train, y_train)
            # 精度などをログ出力
            logger.info(f"performance on training set")
            train_res_list = print_perf(grid, X_train, y_train)
            logger.info(f"performance on test set")
            test_res_list = print_perf(grid, X_test, y_test)
            res_arr.append(test_res_list)

            # 対象のpklファイル名
            if task_name == "correctness":
                model_filename = f"{dataset}-{task_name}-{rb}-{cname}.pkl"
            elif task_name == "fairness":
                model_filename = f"{dataset}-{task_name}-{sens_name}-{rb}-{cname}.pkl"
            model_save_path = os.path.join(save_dir, model_filename)

            # 学習済みモデルをpklで保存
            pickle.dump(grid.best_estimator_, open(model_save_path, "wb"))
            logger.info(f"saved to {model_save_path}")
        # テストデータ精度をcsvで保存
        np.savetxt(res_save_path, np.array(res_arr), fmt="%.3f", delimiter=",")
