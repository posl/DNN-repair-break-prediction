"""
TODO
修正前モデルのtrain, repair, testデータに対する予測結果は experiments/care/sample_metrics/{dataset}-repair-check-{sens_name}-setting{sid}/{div}_fold{1..K}.csv に保存されている
具体的には, 上記csvの sm_corr_bef 列に格納されているので, そこを取り出せばいい
修正後のモデルの予測結果は experiments/arachne/check_repair_results/{dataset}/rep{rep}/is_corr_fold{0..K-1}.npz に保存されている
具体的には読み込んだnpzのオブジェクトis_corrに対してis_corr["train"]などのようにしてアクセスすればよい.
これらの情報から, 各データに対して以下の情報を取得する:
(1) 修正前に正解だったか否か
(2) 5回の修正後に, 何回予測成功したか
とりあえずそこまでやって (1)と(2)の回数の分布を出したい
"""

import os, sys, re
from collections import defaultdict

import numpy as np
import pandas as pd

from lib.util import json2dict, dataset_type
from lib.log import set_exp_logging
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# plot setting
sns.set_style("white")

# ignore warnings
warnings.filterwarnings("ignore")

# 実験の繰り返し数
num_reps = 5
topn = 0

if __name__ == "__main__":
    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(sys.argv[1])
    arachne_dir = exp_dir.replace("care", "arachne")
    exp_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]

    # log setting
    log_file_name = exp_name.replace("training", "arachne-check")
    logger = set_exp_logging(exp_dir.replace("care", "arachne"), exp_name, log_file_name)

    # 設定用のjsonファイルをdictとしてロード
    # HACK: 共通しているので関数にまとめて自動化したい
    setting_dict = json2dict(sys.argv[1])
    logger.info(f"Settings: {setting_dict}")
    task_name = setting_dict["TASK_NAME"]
    num_fold = setting_dict["NUM_FOLD"]

    # モデルとデータの読み込み先のディレクトリ
    data_dir = f"/src/data/{task_name}/{exp_name}"
    model_dir = f"/src/models/{task_name}/{exp_name}"

    # sample metrics (arachne) の保存用ディレクトリ
    sm_save_dir = os.path.join(arachne_dir, "sample_metrics", exp_name)
    os.makedirs(sm_save_dir, exist_ok=True)
    # repair_break dataset (arachne, raw_data) の保存用ディレクトリ
    rb_ds_save_dir = os.path.join(arachne_dir, "repair_break_dataset", "raw_data")
    os.makedirs(rb_ds_save_dir, exist_ok=True)

    # 対象とする誤分類の発生頻度の順位 (0は1位, 1は2位, ...)
    topn = 0

    df_repair_list, df_break_list = [], []
    # 各foldにおいてArachne dirに保存してあるデータを読み込む
    for k in range(num_fold):
        for div in ["train", "repair", "test"]:
            logger.info(f"processing fold {k} {div} set...")

            # 修正前に正解だったかどうかの情報をcareのsample metricsのファイルから得る
            care_dir = arachne_dir.replace("arachne", "care")
            sample_metrics_path = os.path.join(
                care_dir,
                "sample_metrics",
                exp_name.replace("training", f"repair-check"),
                f"{div}_fold{k+1}.csv",
            )
            df = pd.read_csv(sample_metrics_path)
            is_corr_bef = df["sm_corr_bef"].values

            # 修正後の予測結果を格納するための配列
            is_corr_aft = np.zeros((len(is_corr_bef), num_reps))

            # 修正のnum_reps回の適用結果に関してそれぞれ見ていく
            for rep in range(num_reps):
                # 修正後の予測結果をロード
                is_corr_save_path = os.path.join(
                    arachne_dir, "check_repair_results", task_name, f"rep{rep}", f"is_corr_fold-{k}.npz"
                )
                is_corr_aft_rep = np.load(is_corr_save_path)[div]
                is_corr_aft[:, rep] = is_corr_aft_rep
            is_corr_aft_sum = np.sum(is_corr_aft, axis=1, dtype=np.int32)

            # 修正前にあってたかどうかと，5回の修正それぞれの後で正しく予測できた回数の合計をまとめたDataFrameを作成
            df = pd.DataFrame({"sm_corr_bef": is_corr_bef, "sm_corr_aft_sum": is_corr_aft_sum})
            # repaired, brokenの真偽を決定
            # df["repaired"] = (df["sm_corr_bef"] == 0) & (df["sm_corr_aft_sum"] == 5)  # 厳し目の決定方法
            df["repaired"] = (df["sm_corr_bef"] == 0) & (df["sm_corr_aft_sum"] >= 1)  # ゆる目の決定方法
            df["broken"] = (df["sm_corr_bef"] == 1) & (df["sm_corr_aft_sum"] != 5)  # 厳し目の決定方法
            logger.info(f"df_sm.shape: {df.shape}")
            df.to_csv(os.path.join(sm_save_dir, f"{div}_fold{k+1}.csv"), index=False)
            logger.info(f'saved to {os.path.join(sm_save_dir, f"{div}_fold{k+1}.csv")}')

            # exp. metricsもロードしてきて，repaired, brokenなどの列と結合する
            exp_metrics_path = os.path.join(care_dir, "explanatory_metrics", exp_name, f"{div}_fold{k+1}.csv")
            df_expmet = pd.read_csv(exp_metrics_path)
            df_all = pd.concat([df_expmet, df], axis=1)
            logger.info(f"df_all.shape: {df_all.shape}")

            # repair, breakのデータセットを作成
            df_repair, df_break = df_all[df_all["sm_corr_bef"] == 0], df_all[df_all["sm_corr_bef"] == 1]
            # それぞれのdfからいらない列を削除
            df_repair = df_repair.drop(["sm_corr_bef", "sm_corr_aft_sum", "broken"], axis=1)
            df_break = df_break.drop(["sm_corr_bef", "sm_corr_aft_sum", "repaired"], axis=1)
            logger.info(f"df_repair.shape: {df_repair.shape}, df_break.shape: {df_break.shape}")
            df_repair_list.append(df_repair)
            df_break_list.append(df_break)

    logger.info("Concatenating all folds and divisions...")
    # 全fold, divにおけるdf_repair, df_breakを結合して全体のrepair dataset, break datasetを作る
    df_repair = pd.concat(df_repair_list, axis=0)
    df_break = pd.concat(df_break_list, axis=0)
    logger.info(f"df_repair.shape: {df_repair.shape}, df_break.shape: {df_break.shape}")
    logger.info(f"#repaired is True: {len(df_repair[df_repair['repaired']==True])} / {len(df_repair)}")
    logger.info(f"#broken is True: {len(df_break[df_break['broken']==True])} / {len(df_break)}")
    # それぞれcsvとして保存
    df_repair.to_csv(os.path.join(rb_ds_save_dir, f"{exp_name}-repair.csv"), index=False)
    df_break.to_csv(os.path.join(rb_ds_save_dir, f"{exp_name}-break.csv"), index=False)
