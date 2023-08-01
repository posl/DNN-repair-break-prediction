import os, sys
from collections import defaultdict
from itertools import product
import pandas as pd
import numpy as np
from lib.util import json2dict
from lib.log import set_exp_logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split

# plot setting
sns.set()


def outlier_iqr(df, obj_col):
    tmp_df = df.copy()
    for col in tmp_df.columns:
        if col == obj_col:
            continue
        # 四分位数
        q1 = tmp_df[col].quantile(0.25)
        q3 = tmp_df[col].quantile(0.75)
        iqr = q3 - q1  # 四分位範囲
        # 外れ値の基準点
        outlier_min = q1 - (iqr) * 1.5
        outlier_max = q3 + (iqr) * 1.5
        # 範囲から外れている値を除く
        tmp_df = tmp_df[(tmp_df[col] > outlier_min) & (tmp_df[col] < outlier_max)]
        logger.info(f"after {col} filtered: {len(tmp_df)} rows")
    return tmp_df


# TODO: これらの定数を外部化
# =====================================================
# 対象とするrepair手法のリスト
methods = ["care", "apricot"]
# 対象とするdivision (train, repair, testのいずれか)
divisions = ["train", "repair", "test"]

# データセット名などのセッティングの辞書
conf_dic = {
    "credit": {"sens_name": "gender", "num_folds": 5},
    "census": {"sens_name": "gender", "num_folds": 10},
    "bank": {"sens_name": "age", "num_folds": 10},
}
# 説明変数の列名のリスト
exp_metrics = ["pcs", "lps", "loss", "entropy"]
# testの割合
test_ratio = 0.2
# 分割時のシード
random_state = 42
# 対象のsetting id
sid = 1
# =====================================================


if __name__ == "__main__":
    # このプログラムのファイル名を取得
    file_name = os.path.splitext(sys.argv[0])[0]

    df_corr_ld = defaultdict(pd.DataFrame)
    df_fair_ld = defaultdict(pd.DataFrame)
    # dic = {"correctness": df_corr_ld, "fairness": df_fair_ld}
    dic = {"correctness": df_corr_ld}

    for _, df_ld in dic.items():
        for dataset, method in product(conf_dic.keys(), methods):
            # 実験のディレクトリと実験名を取得
            exp_dir = f"/src/experiments/{method}"
            # sensitive feature名
            sens_name = conf_dic[dataset]["sens_name"]
            # ログファイルの生成
            logger = set_exp_logging(exp_dir, f'{dataset}-{file_name.replace("_", "-")}')
            # 最終的な出力物の格納ディレクトリの生成
            save_dir = os.path.join(exp_dir, "repair_break_dataset", "preprocessed_data")
            os.makedirs(save_dir, exist_ok=True)

            for rb in ["repair", "break"]:
                logger.info(f"method={method}, dataset={dataset}, rb={rb}")
                # repair/break datasetのcsvをロード
                rb_ds_filename = (
                    f"{dataset}-fairness-{sens_name}-setting{sid}-{rb}.csv"
                    if method == "care"
                    else f"{dataset}-training-setting{sid}-{rb}.csv"
                )
                rb_ds_path = os.path.join(exp_dir, "repair_break_dataset", "raw_data", rb_ds_filename)
                df = pd.read_csv(rb_ds_path)
                logger.info(f"original df shape: {df.shape}")

                # 目的変数と説明変数を取得
                obj_col = "repaired" if rb == "repair" else "broken"
                feats = [feat for feat in list(df.columns) if feat != obj_col]

                # 異常値の除去を行う (breakだけ)
                # if rb == "break":
                #     logger.info("Elimination of anomaly values...")
                #     df = outlier_iqr(df, obj_col)
                #     logger.info(f"After elimination df shape: {df.shape}")

                # trainval / test に分割
                logger.info("Dividing into trainval / test...")
                df_train, df_test = train_test_split(df, test_size=test_ratio, random_state=random_state)
                logger.info(f"df_train.shape: {df_train.shape}, df_test.shape: {df_test.shape}")

                # trainvalの方に標準化を行う
                logger.info("Standardization...")
                std = StandardScaler()
                std.fit(df_train[feats])
                df_train[feats] = std.transform(df_train[feats])
                # trainvalと同様の変換をtestに適用
                df_test[feats] = std.transform(df_test[feats])

                # trainvalの方にyeo-johnson変換（と，その後に標準化）を行う
                logger.info("Yeo-Johnson transformation and Standardization...")
                pt = PowerTransformer(method="yeo-johnson", standardize=True)
                pt.fit(df_train[feats])
                df_train[feats] = pt.transform(df_train[feats])
                # trainvalと同様の変換をtestに適用
                df_test[feats] = pt.transform(df_test[feats])

                # 作成したdfをコピーして別の変数に保存
                df_ld[rb] = df_train.copy()

                # trainvalの方をcsvで保存
                rb_ds_trainval_filename = (
                    f"{dataset}-fairness-{sens_name}-setting{sid}-{rb}-trainval.csv"
                    if method == "care"
                    else f"{dataset}-training-setting{sid}-{rb}-trainval.csv"
                )
                rb_ds_trainval_path = os.path.join(
                    exp_dir, "repair_break_dataset", "preprocessed_data", rb_ds_trainval_filename
                )
                df_train.to_csv(rb_ds_trainval_path, index=False)

                # testの方をcsvで保存
                rb_ds_test_filename = (
                    f"{dataset}-fairness-{sens_name}-setting{sid}-{rb}-test.csv"
                    if method == "care"
                    else f"{dataset}-training-setting{sid}-{rb}-test.csv"
                )
                rb_ds_test_path = os.path.join(
                    exp_dir, "repair_break_dataset", "preprocessed_data", rb_ds_test_filename
                )
                df_test.to_csv(rb_ds_test_path, index=False)

            # violinplotにまとめる
            for rb in ["repair", "break"]:
                print(f"Making violinplot of trainval set of method={method}, dataset={dataset}, rb={rb}...")
                obj_col = "repaired" if rb == "repair" else "broken"
                ltrue = df_ld[rb][obj_col].sum()
                print(
                    f"df_corr_ld[rb].shape: {df_corr_ld[rb].shape}, #{obj_col}: {ltrue}, #non-{obj_col}: {len(df_ld[rb]) - ltrue}"
                )
                fig, axes = plt.subplots(nrows=1, ncols=4, sharex=False, figsize=(12, 3))
                plt.subplots_adjust(wspace=0.2, hspace=0.2)
                for task_name, df_ld in dic.items():
                    for i, feat in enumerate(feats):
                        df = df_ld[rb]
                        sns.violinplot(data=df, x=obj_col, y=feat, palette="Greys", ax=axes[i], split=True)
                fig.tight_layout()
                fig_path = os.path.join(save_dir, f"{dataset}-trainval-{rb}.pdf")
                plt.savefig(fig_path, bbox_inches="tight")
