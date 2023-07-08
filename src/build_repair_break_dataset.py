import os, sys
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

# TODO: これらの定数を外部化
# =====================================================
# 対象とするsetting_idのリスト
# setting_ids = list(range(1, 9, 1))
setting_ids = [1]
# 対象とするdivision (train, repair, testのいずれか)
divisions = ["train", "repair", "test"]
# 対象とするdataset
# dataset = "credit"
# dataset = "census"
dataset = "bank"
# 対象とする観点
task_name = "fairness"
# fairnessの場合のsensitive feature
# sens_name = "gender"
sens_name = "age"
# fold数
# num_folds = list(range(1, 6, 1))
num_folds = list(range(1, 11, 1))
# 説明変数の列名のリスト
exp_metrics = ["pcs", "lps", "loss", "entropy"]
# =====================================================


if __name__ == "__main__":
    # このプログラムのファイル名を取得
    file_name = os.path.splitext(sys.argv[0])[0]
    # 実験のディレクトリと実験名を取得
    exp_dir = "/src/experiments"
    # ログファイルの生成
    logger = set_exp_logging(exp_dir, f'{dataset}-{file_name.replace("_", "-")}')

    # setting_idのループ
    for sid in setting_ids:
        logger.info(f"dataset={dataset}, task_name={task_name}, setting_id={sid}")
        judge_dir = os.path.join(exp_dir, "judge_repair_outcome", f"{dataset}-{task_name}-{sens_name}-setting{sid}")
        df_list = []

        # divisionとfold数のループ
        for div, k in product(divisions, num_folds):
            logger.info(f"division={div}, k={k}")
            # 対応するcsvをロード
            judge_path = os.path.join(judge_dir, f"{div}_fold{k}.csv")
            df = pd.read_csv(judge_path)
            logger.info(df.shape)
            df_list.append(df)

        # 同じsettingにおけるfoldやdivisionごとの票をまとめてひとつにする
        df_setting = pd.concat(df_list, axis=0)
        print(df_setting[f"judge_{task_name}"].value_counts())

        # 出力ようのcsvファイル作成して保存
        div_name = "-".join(divisions)
        max_fold = max(num_folds)
        save_path = os.path.join(judge_dir, f"{div_name}_{max_fold}folds.csv")
        df_setting.to_csv(save_path, index=False)
        logger.info(f"saved to {save_path}")

        # repair set, side-effect setに分けて保存したい
        # fairnessの処理をする場合は, correctnessも一緒にやっちゃう
        task_names = [task_name, "correctness"] if task_name == "fairness" else [task_name]
        for tn in task_names:
            # ここで必要な列のみに絞る
            used_cols = exp_metrics + [f"judge_{tn}"]
            df = df_setting[used_cols]
            # dfをrepair set, side-effect setの2つに分割
            df_repair = df[df[f"judge_{tn}"].isin(["repaired", "non-repaired"])]
            df_break = df[df[f"judge_{tn}"].isin(["broken", "non-broken"])]
            logger.info(f"#df_repair: {len(df_repair)}, #df_break: {len(df_break)}")
            # judgeの列をリネームしてbool値に変更
            df_repair = df_repair.rename(columns={f"judge_{tn}": "repaired"})
            df_break = df_break.rename(columns={f"judge_{tn}": "broken"})
            df_repair["repaired"] = df_repair["repaired"] == "repaired"
            df_break["broken"] = df_break["broken"] == "broken"
            # df_repair, df_breakをそれぞれ保存
            df_repair_save_path = os.path.join(judge_dir, f"{div_name}_{max_fold}folds_repair_{tn}.csv")
            df_break_save_path = os.path.join(judge_dir, f"{div_name}_{max_fold}folds_break_{tn}.csv")
            df_repair.to_csv(df_repair_save_path, index=False)
            logger.info(f"saved to {df_repair_save_path}")
            df_break.to_csv(df_break_save_path, index=False)
            logger.info(f"saved to {df_break_save_path}")

    # =======================================
    # 全てのsettingをまとめて保存するフェーズ
    # =======================================
    df_corr_dict = defaultdict(list)
    df_fair_dict = defaultdict(list)

    # settingごとのループ
    for rb in ["repair", "break"]:
        for sid in setting_ids:
            logger.info(f"sid: {sid}, repair/break: {rb}")
            judge_dir = os.path.join(exp_dir, "judge_repair_outcome", f"{dataset}-{task_name}-{sens_name}-setting{sid}")
            corr_file_name = os.path.join(judge_dir, f"train-repair-test_{len(num_folds)}folds_{rb}_correctness.csv")
            fair_file_name = os.path.join(judge_dir, f"train-repair-test_{len(num_folds)}folds_{rb}_fairness.csv")
            df_corr = pd.read_csv(corr_file_name)
            df_fair = pd.read_csv(fair_file_name)
            logger.info(f"df_corr.shape: {df_corr.shape}, df_fair.shape: {df_fair.shape}")
            df_corr_dict[rb].append(df_corr)
            df_fair_dict[rb].append(df_fair)

    df_corr_long_dict = defaultdict(pd.DataFrame)
    df_fair_long_dict = defaultdict(pd.DataFrame)

    for rb in ["repair", "break"]:
        logger.info(f"repair/break: {rb}")
        obj_col = "repaired" if rb == "repair" else "broken"
        # 全設定を一つにまとめる
        df_corr_long_dict[rb] = pd.concat(df_corr_dict[rb], axis=0)
        df_fair_long_dict[rb] = pd.concat(df_fair_dict[rb], axis=0)

        # 目的変数の内訳を確認
        logger.info(f"correctness, {rb}\n{df_corr_long_dict[rb].shape}")
        logger.info(
            f"(correctness) #{obj_col} is True: {len(df_corr_long_dict[rb][df_corr_long_dict[rb][obj_col]==True])} / {len(df_corr_long_dict[rb])}"
        )
        logger.info(f"fairness, {rb}\n{df_fair_long_dict[rb].shape}")
        logger.info(
            f"(fairness) #{obj_col} is True: {len(df_fair_long_dict[rb][df_fair_long_dict[rb][obj_col]==True])} / {len(df_fair_long_dict[rb])}"
        )

        # 全設定まとめたデータフレームを保存
        corr_rb_dataset_path = os.path.join(
            exp_dir, "repair_break_dataset", "raw_data", f"{dataset}-correctness-{rb}.csv"
        )
        fair_rb_dataset_path = os.path.join(
            exp_dir, "repair_break_dataset", "raw_data", f"{dataset}-fairness-{sens_name}-{rb}.csv"
        )
        df_corr_long_dict[rb].to_csv(corr_rb_dataset_path, index=False)
        logger.info(f"saved to {corr_rb_dataset_path}")
        df_fair_long_dict[rb].to_csv(fair_rb_dataset_path, index=False)
        logger.info(f"saved to {fair_rb_dataset_path}")
