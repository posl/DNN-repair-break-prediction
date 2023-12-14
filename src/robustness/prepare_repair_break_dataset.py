import os, sys, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
import pandas as pd
import numpy as np
from lib.util import json2dict, dataset_type
from lib.log import set_exp_logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split

# plot setting
sns.set()

# NOTE: 定数（外部化しなくてもいっかな）
# =====================================================
# testの割合
test_ratio = 0.2
# 分割時のシード
random_state = 42
# =====================================================


def preprocess_df(df, rb):
    # 目的変数と説明変数を取得
    obj_col = "repaired" if rb == "repair" else "broken"
    feats = [feat for feat in list(df.columns) if feat != obj_col]

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

    return df_train, df_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["fmc", "c10c"], help="dataset name ('fmc' or 'c10c')")
    parser.add_argument("method", choices=["care", "apricot", "arachne"], help="dataset name ('care', 'apricot' or 'arachne')")
    args = parser.parse_args()
    ds = args.dataset
    method = args.method
    # log setting
    exp_dir = f"/src/experiments/{method}/"
    care_dir = "/src/experiments/care/"
    this_file_name = os.path.basename(__file__).replace(".py", "").replace("_", "-")
    log_file_name = f"{ds}-{this_file_name}"
    logger = set_exp_logging(exp_dir, log_file_name, log_file_name)
    logger.info(f"target dataset: {ds}, method: {method}")
    
    # original dataset
    if ds in ["fmc", "c10c"]:
        ori_ds = ds.rstrip("c")
        num_fold = 5
        ds_type = dataset_type(ori_ds)
        is_binary = False
    else:
        raise ValueError(f"Invalid dataset name: {ds}")
    
    # rb_datasetのdir
    raw_rb_dir = f"/src/src/robustness/rb_datasets/{ds}/{method}/"
    preprocessed_rb_dir = f"/src/src/robustness/preprocessed_rb_datasets/{ds}/{method}/"
    os.makedirs(preprocessed_rb_dir, exist_ok=True)

    # raw_rb_dirのファイルを取得
    rb_files = [f for f in os.listdir(raw_rb_dir) if f.endswith(".csv")]
    # 上記ファイル名の最後の_の前の部分を取得しsetにする
    file_names = [f.rsplit("_", 1)[0] for f in rb_files]
    file_names_set = set(file_names)

    # 全てのfoldをconcatenateしてdfにする
    # c10cの場合はfile_namesごとにやるが，fmcに関してはまとめる
    dic_df = defaultdict(defaultdict)
    if ds == "fmc":
        dic_df_rb = defaultdict(pd.DataFrame)
        for rb in ["repair", "break"]:
            df = pd.DataFrame()
            for fname in rb_files:
                if not rb in fname:
                    continue
                tmp_df = pd.read_csv(os.path.join(raw_rb_dir, fname))
                df = pd.concat([df, tmp_df], axis=0)
            dic_df_rb[rb] = df
        dic_df["fmc"] = dic_df_rb
    else:
        for division in file_names_set:
            dic_df_rb = defaultdict(pd.DataFrame)
            for rb in ["repair", "break"]:
                df = pd.DataFrame()
                for fname in rb_files:
                    if (rb in fname) and (division in fname):
                        tmp_df = pd.read_csv(os.path.join(raw_rb_dir, fname))
                        df = pd.concat([df, tmp_df], axis=0)
                dic_df_rb[rb] = df
            dic_df[division] = dic_df_rb
    
    # preprocessを適用して保存を繰り返す
    for division, dic_df_rb in dic_df.items():
        for rb, df in dic_df_rb.items():
            logger.info(f"processing {division}, {rb}...")
            # preprocess
            df_train, df_test = preprocess_df(df, rb)
            # save
            df_train.to_csv(os.path.join(preprocessed_rb_dir, f"{division}-{rb}-trainval.csv"), index=False)
            df_test.to_csv(os.path.join(preprocessed_rb_dir, f"{division}-{rb}-test.csv"), index=False)
            # logging
            obj_col = "repaired" if rb == "repair" else "broken"
            logger.info(f"df_{rb}.shape: {df.shape}")
            logger.info(f"#{obj_col} is True: {len(df[df[obj_col]==True])} / {len(df)}")
