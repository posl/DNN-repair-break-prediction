import os, sys, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
import pandas as pd
import numpy as np
from lib.model import select_model, eval_model
from lib.util import json2dict, dataset_type
from lib.log import set_exp_logging
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# plot setting
sns.set()


def update_metrics_dict(metrics_dict, *metrics):
    """モデルの予測メトリクスの辞書を更新するだけの関数.

    Args:
        metrics_dict (defaultdict(list)): モデルの予測のメトリクスの辞書
        metrics (*float): メトリクスの値の位置引数. メトリクス数の変化に対応するために位置引数に.

    Returns:
        defaultdict(list): 更新後の辞書
    """
    # NOTE: eval_modelの返り値の順番の都合上, ここの順番に注意
    metrics_names = ["acc", "f1", "pre", "rec", "mcc"]
    assert len(metrics_names) == len(metrics), "Error: Num of Metrics Inconsistency."

    for mn, m in zip(metrics_names, metrics):
        metrics_dict[mn].append(m)
    return metrics_dict


def make_df_metrics(all_metrics_dict, num_fold):
    """モデルの各foldごとの予測メトリクスをdfにまとめる.

    (Extended Summary)
    | division | metrics | fold_0 | ... | fold_{num_fold} |
    |----------|---------|--------| ... |-----------------|

    Args:
        all_metrics_dict (dict): 各division (train, repair, testなど) の各foldの各メトリクスを保存したdict.
        num_fold (int): fold数.

    Returns:
        pd.DataFrame: 上記のフォーマットの表
    """
    # foldを表す列名を作成
    columns = ["division", "metrics"]
    fold_cols = [f"fold_{k}" for k in range(num_fold)]
    columns.extend(fold_cols)
    # dfの定義
    df = pd.DataFrame(columns=columns)

    # dfに行を追加していく
    for div, div_dic in all_metrics_dict.items():
        for metric, met_list in div_dic.items():
            row = [div, metric]
            row.extend(met_list)
            row_df = pd.DataFrame(data=[row], columns=columns)
            df = pd.concat([df, row_df], axis=0)
    logger.info(f"df.columns={df.columns}. df.shape={df.shape}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["fmc", "c10c"], help="dataset name ('fmc' or 'c10c')")
    args = parser.parse_args()
    ds = args.dataset
    # log setting
    exp_dir = "/src/experiments/care/"
    this_file_name = os.path.basename(__file__).replace(".py", "").replace("_", "-")
    log_file_name = f"{ds}-{this_file_name}"
    logger = set_exp_logging(exp_dir, log_file_name, log_file_name)
    logger.info(f"target dataset: {ds}")
    
    # original dataset
    if ds in ["fmc", "c10c"]:
        ori_ds = ds.rstrip("c")
        num_fold = 5
        ds_type = dataset_type(ori_ds)
    else:
        raise ValueError(f"Invalid dataset name: {ds}")

    # モデルとデータの読み込み先のディレクトリ
    data_dir = f"/src/data/{ds}/"
    data_files = os.listdir(data_dir)
    model_dir = f"/src/models/{ori_ds}/{ori_ds}-training-setting1"

    # データを読み込んでnpyの辞書を作る
    npy_dic = {}
    for file_name in data_files:
        file_path = os.path.join(data_dir, file_name)
        # file_nameからnpyを除いた部分だけ取得
        key = file_name.replace(".npy", "")
        # npyをロード
        npy_dic[key] = torch.from_numpy(np.load(file_path)).clone()
        logger.info(f"{key}, {npy_dic[key].shape}")

    # 読み込んだnpyからdataloaderを作成
    # データセットにより異なる処理をしないといけない
    dl_dic = {}
    if ds == "fmc":
        # train, testに対するTensorDatasetを作成
        train_x = npy_dic["fmnist-c-train"]
        test_x = npy_dic["fmnist-c-test"]
        train_y = npy_dic["fmnist-c-train-labels"]
        test_y = npy_dic["fmnist-c-test-labels"]
        train_ds = TensorDataset(train_x, train_y)
        test_ds = TensorDataset(test_x, test_y)
        # DataLoaderにして辞書に入れる
        dl_dic["train"] = DataLoader(train_ds, batch_size=32, shuffle=False)
        dl_dic["test"] = DataLoader(test_ds, batch_size=32, shuffle=False)
    elif ds == "c10c":
        labels = npy_dic["labels"]
        print(labels.shape)
        dl_dic = {}
        # Corruptionの種類ごとにTensorDatasetを作成
        for k, v in npy_dic.items():
            if "labels" in k:
                continue
            # まずはtensor datasetをつくる
            corruption_ds = TensorDataset(v, labels)
            dl_dic[k] = DataLoader(corruption_ds, batch_size=32, shuffle=False)

    # dl_dicに含まれる各dlに対して，各メトリクスを保存するdictを保存するためのdefaultdict
    metrics_dic = defaultdict(defaultdict)

    # 各foldのtrain/repairをロードして予測
    for k in range(num_fold):
        logger.info(f"processing fold {k}...")

        # 学習済みモデルをロード
        model = select_model(task_name=ori_ds)
        model_path = os.path.join(model_dir, f"trained_model_fold-{k}.pt")
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # dl_dicにあるdlのそれぞれに対して, evaluation criteriaの計算
        for key, dl in dl_dic.items():
            metrics_dic[key] = eval_model(model, dl, ds_type)["metrics"]
        logger.info(metrics_dic)

        exit()

        # メトリクスの記録
        train_metrics_dict = update_metrics_dict(train_metrics_dict, *train_metrics)
        repair_metrics_dict = update_metrics_dict(repair_metrics_dict, *repair_metrics)
        test_metrics_dict = update_metrics_dict(test_metrics_dict, *test_metrics)

        # メトリクスを表示
        logger.info(
            f"train acc: {train_metrics[0]:.4f}, train f1: {train_metrics[1]:.4f}, train pre: {train_metrics[2]:.4f}, train rec: {train_metrics[3]:.4f}, train mcc: {train_metrics[4]:.4f}"
        )
        logger.info(
            f"repair acc: {repair_metrics[0]:.4f}, repair f1: {repair_metrics[1]:.4f}, repair pre: {repair_metrics[2]:.4f}, repair rec: {repair_metrics[3]:.4f}, repair mcc: {repair_metrics[4]:.4f}"
        )
        logger.info(
            f"test acc: {test_metrics[0]:.4f}, test f1: {test_metrics[1]:.4f}, test pre: {test_metrics[2]:.4f}, test rec: {test_metrics[3]:.4f}, test mcc: {test_metrics[4]:.4f}"
        )

    # メトリクスの辞書をcsvにして保存しておく
    all_metrics_dict = {
        "train": dict(train_metrics_dict),
        "repair": dict(repair_metrics_dict),
        "test": dict(test_metrics_dict),
    }
    csv_file_name = log_file_name.replace("setting", "result-setting")
    save_path = os.path.join(exp_dir, f"{csv_file_name}.csv")
    df = make_df_metrics(all_metrics_dict, num_fold)
    df.to_csv(save_path, index=False)
    logger.info(f"saved to {save_path}")
