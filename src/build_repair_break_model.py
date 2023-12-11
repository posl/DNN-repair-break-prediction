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


def print_perf(clf, X, y, output_log=True):
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
    if output_log:
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


def under_sampling(df, target_column, minority=True, mode=None, sample_size_ratio=None, reduce_ratio=None, whole_ratio=None):
    """dfのある列target_columnにおいて, 多数派の属性を持つ行を間引いて少数派minorityの行数にmajorityの行数を合わせたdfを返す

    Args:
        df (pd.DataFrame) :対象となるdf
        target_column (string) : 対象となる列名
        minority (boolean) : 少数派のラベル（True, Falseのみ）
        sample_size_ratio (float) : 多数派のラベルのデータ数を，少数派のラベルのデータ数の何倍にするか.
        reduce_ratio (float) : 多数派のラベルのデータ数からどれくらいの割合でサンプリングするか. sample_size_ratioと同時に指定することはできなず, その場合はエラー終了.
        whole_ratio (float) : 両方のラベルから均等にサンプリングする場合の割合.

    Returns:
        pd.DataFrame : 間引いた後のdf
    """
    # modeが指定の3つ以外ならエラー
    assert mode in ["sample_size_ratio", "reduce_ratio", "whole_ratio"]
    # 少ない方の行数を取得
    df_only_minority = df[df[target_column] == minority]
    df_only_majority = df[df[target_column] != minority]
    minority_cnt = df_only_minority.shape[0]
    majority_cnt = df_only_majority.shape[0]
    # 両方ラベルをアンダーサンプリング
    if mode == "whole_ratio":
        assert whole_ratio is not None
        mino_sample_size, majo_sample_size = minority_cnt, majority_cnt
        # 100件より小さい場合はかわいそうだからサンプリングしない
        if mino_sample_size >= 100:
            mino_sample_size = int(whole_ratio * minority_cnt)
        if majo_sample_size >= 100:
            majo_sample_size = int(whole_ratio * majority_cnt)
        # 両方のラベルからwhole_ratioの割合でサンプリング
        df_sampled_minority = df_only_minority.sample(n=mino_sample_size)
        df_sampled_majority = df_only_majority.sample(n=majo_sample_size)
        df_concat = pd.concat([df_sampled_minority, df_sampled_majority])
        df_concat = df_concat.sample(frac=1, random_state=726)
        return df_concat
    # 少ない方の行数のX倍まで多い方の行をサンプリングする
    if mode == "sample_size_ratio":
        assert sample_size_ratio is not None
        # 多い方の行から，少ない方の数 * sample_size_ratioだけサンプリング
        sample_size = int(sample_size_ratio * minority_cnt)
        df_sampled = df_only_majority.sample(n=sample_size, random_state=629)
        df_concat = pd.concat([df_only_minority, df_sampled])
        df_concat = df_concat.sample(frac=1, random_state=726)
        return df_concat
    # 多い方の行数からX%の割合でサンプリングする
    if mode == "reduce_ratio":
        assert reduce_ratio is not None
        # 多い方からreduce_ratioの割合でサンプリング
        sample_size = int(reduce_ratio * majority_cnt)
        df_sampled = df_only_majority.sample(n=sample_size, random_state=629)
        df_concat = pd.concat([df_only_minority, df_sampled])
        df_concat = df_concat.sample(frac=1, random_state=726)
        return df_concat


# TODO: これらの定数を外部化
# =====================================================
# 対象とするrepair手法のリスト
# methods = ["care", "apricot"]
methods = ["arachne"]
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
    method = sys.argv[1]
    dataset = sys.argv[2]
    # NNが学習したfold数
    nn_folds = 10 if dataset in ["census", "bank"] else 5
    inv_nn_folds = 1 / nn_folds
    # ログファイルの生成
    logger = set_exp_logging(f"/src/experiments/{method}", f'{dataset}-{file_name.replace("_", "-")}')
    # 実行時間
    time_df = pd.DataFrame(columns=["rb", "fit_or_inf", "lr", "lgb", "rf"])

    # 実験のディレクトリと実験名を取得
    exp_dir = f"/src/experiments/{method}"
    save_dir = os.path.join(exp_dir, "repair_break_model")
    os.makedirs(save_dir, exist_ok=True)

    for rb in ["repair", "break"]:
        logger.info(f"method = {method}, repair or break = {rb}")
        # 目的変数の列名
        obj_col = "repaired" if rb == "repair" else "broken"
        # モデル作成時の目的関数
        scoring = "recall" if rb == "break" else "f1"
        # 対象のcsvファイル名
        if method == "care":
            train_csv_name = f"{dataset}-fairness-setting1-{rb}-trainval.csv"
            test_csv_name = f"{dataset}-fairness-setting1-{rb}-test.csv"
        elif method == "apricot" or method == "arachne":
            train_csv_name = f"{dataset}-training-setting1-{rb}-trainval.csv"
            test_csv_name = f"{dataset}-training-setting1-{rb}-test.csv"
        # 必要なデータセットをロードする
        train_ds_dir = os.path.join(exp_dir, "repair_break_dataset/preprocessed_data", train_csv_name)
        test_ds_dir = os.path.join(exp_dir, "repair_break_dataset/preprocessed_data", test_csv_name)
        df_train = pd.read_csv(train_ds_dir)
        df_test = pd.read_csv(test_ds_dir)
        # 形状の確認
        logger.info(f"df_train.shape={df_train.shape}, df_test.shape={df_test.shape}")
        logger.info(f"col {obj_col} distribution of trainval:\n{df_train[obj_col].value_counts()}")
        logger.info(f"col {obj_col} distribution of test:\n{df_test[obj_col].value_counts()}")

        # obj_colの値の多数派 (in trainval) がTrueかFalseかを判断する
        obj_col_cnts = df_train[obj_col].value_counts(ascending=False)
        majo = obj_col_cnts.index.values[0]  # 多数派
        mino = not majo  # 少数派
        majo_cnt = obj_col_cnts[majo]  # 多数派のサンプル数
        mino_cnt = obj_col_cnts[mino]  # 少数派のサンプル数
        balance = majo_cnt / mino_cnt  # 多数派と少数派のサンプル数の比
        logger.info(f"majo_cnt: {majo_cnt}, mino_cnt: {mino_cnt}, balance: {balance}")
        # df_trainに対するresamplingを実行する
        # resamplingすべきかどうかの判断
        if mino_cnt >= 10000: # df_trainの少数派行数サイズによって, under samplingの方法を変える
            df_train = under_sampling(df_train, obj_col, minority=mino, mode="whole_ratio", whole_ratio=2*inv_nn_folds)
        elif balance >= 5: # 多数派と少数派のサンプル数の比が5以上なら, 多数派からのunder samplingを実行
            # resamplingの実行
            df_train = under_sampling(df_train, obj_col, minority=mino, mode="reduce_ratio", reduce_ratio=inv_nn_folds)
        X_train, y_train = df_train[exp_metrics], df_train[obj_col]
        # テストデータをX, yに分割
        X_test, y_test = df_test[exp_metrics], df_test[obj_col]
        logger.info(f"(AFTER RESAMP.) X_train.shape={X_train.shape}, X_test.shape={X_test.shape}")
        # foldに分割する準備
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=1234)

        # 学習前の設定
        # グリッドサーチで試すパラメータの集合は，モデルの種類によって固定する．repair/side-effで変えない
        parameters_dict = {
            "lr": {"C": [10**i for i in range(-2, 4)], "random_state": [1234]},
            "lgb": {
                "max_depth": [10, 25, 50],
                "learning_rate": [0.001, 0.01, 0.05, 0.1],
                "n_estimators": [10, 50, 100, 200],
            },
            "rf": {"n_estimators": [70, 80, 90], "max_depth": [5, 10, 50]},
        }
        # verboseを事前に設定
        verbose_level = 1
        # 結果保存用のcsvファイル名
        train_res_filename = f"{dataset}-{rb}-train.csv"
        test_res_filename = f"{dataset}-{rb}-test.csv"
        train_res_save_path = os.path.join(save_dir, train_res_filename)
        test_res_save_path = os.path.join(save_dir, test_res_filename)
        train_res_arr, test_res_arr = [], []
        fit_time_list, inf_time_list = [], []

        # 3種類の分類器をそれぞれcross-validation+grid searchで訓練して結果を表示
        for cname in ["lr", "lgb", "rf"]:
            logger.info(f"{rb} model: {cname}")
            # 分類器の選択
            if cname == "lr":
                clf = LogisticRegression(class_weight="balanced")
            elif cname == "lgb":
                clf = lgb.LGBMClassifier(random_state=1234, class_weight="balanced", objective="binary")
            elif cname == "rf":
                clf = RandomForestClassifier(criterion="entropy", random_state=1234, class_weight="balanced")
            grid = GridSearchCV(
                clf, param_grid=parameters_dict[cname], cv=kf, scoring=scoring, verbose=verbose_level
            )
            # 訓練の実行
            s_fit = time.time()
            grid.fit(X_train, y_train)
            time_for_fit = time.time() - s_fit
            fit_time_list.append(time_for_fit)
            logger.info(f"FIT TIME: {time_for_fit} sec.")

            # 精度などをログ出力
            logger.info(f"performance on training set")
            s_inf1 = time.time()
            train_res_list = print_perf(grid, X_train, y_train)
            time_for_inf1 = time.time() - s_inf1
            train_res_arr.append(train_res_list)

            logger.info(f"performance on test set")
            s_inf2 = time.time()
            test_res_list = print_perf(grid, X_test, y_test)
            time_for_inf2 = time.time() - s_inf2
            time_for_inf = time_for_inf1 + time_for_inf2
            inf_time_list.append(time_for_inf)
            logger.info(f"INF TIME: {time_for_inf} sec.")
            test_res_arr.append(test_res_list)

            # 対象のpklファイル名
            model_filename = f"{dataset}-{rb}-{cname}.pkl"
            model_save_path = os.path.join(save_dir, model_filename)

            # 学習済みモデルをpklで保存
            pickle.dump(grid.best_estimator_, open(model_save_path, "wb"))
            logger.info(f"saved to {model_save_path}")
        # テストデータ精度をcsvで保存
        train_res_df = pd.DataFrame(
            data=train_res_arr,
            columns=["acc", "precision", "recall", "f1", "roc_auc", "pr_auc"],
            index=["lr", "lgb", "rf"],
        )
        test_res_df = pd.DataFrame(
            data=test_res_arr,
            columns=["acc", "precision", "recall", "f1", "roc_auc", "pr_auc"],
            index=["lr", "lgb", "rf"],
        )
        train_res_df.to_csv(train_res_save_path, float_format="%.3f", index=False)
        test_res_df.to_csv(test_res_save_path, float_format="%.3f", index=False)
        # 実行時間df用の行
        fit_time = [rb, "fit", *fit_time_list]
        inf_time = [rb, "inf", *inf_time_list]
        fit_time_row = pd.Series(fit_time, index=time_df.columns)
        inf_time_row = pd.Series(inf_time, index=time_df.columns)
        # dfに行追加
        time_df = time_df.append(fit_time_row, ignore_index=True)
        time_df = time_df.append(inf_time_row, ignore_index=True)
    # 実行時間を保存
    time_df.to_csv(f"/src/experiments/time_for_repair_break_model-{dataset}-{method}.csv", index=False)
