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
    # gpuが使用可能かをdeviceにセット
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        is_binary = False
    else:
        raise ValueError(f"Invalid dataset name: {ds}")

    # モデルとデータの読み込み先のディレクトリ
    data_dir = f"/src/data/{ds}/"
    # dataloaderのファイル名だけ取り出す
    dl_files = [f for f in os.listdir(data_dir) if f.endswith("loader.pt")]
    model_dir = f"/src/models/{ori_ds}/{ori_ds}-training-setting1"

    # 正解不正解の配列を保存するためのdir
    correctness_dir = os.path.join(f"./correctness_before", ds)
    os.makedirs(correctness_dir, exist_ok=True)

    # dataloaderを読み込む
    dl_dic = {}
    for file_name in dl_files:
        file_path = os.path.join(data_dir, file_name)
        # file_nameから'_loader.pt'を除いた部分だけ取得
        key = file_name.replace("_loader.pt", "")
        dl_dic[key] = torch.load(file_path)
        logger.info(f"loaded {key} dataloader.")

    # dl_dicに含まれる各dlに対して，各メトリクスを保存するdictを保存するためのdefaultdict
    metrics_dic = defaultdict(defaultdict)
    for key, dl in dl_dic.items():
        metrics_dic[key] = defaultdict(list)

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
            logger.info(f"processing dl {key}...")
            eval_res_dic = eval_model(model=model, dataloader=dl, dataset_type=ds_type, is_binary=is_binary, device=device)
            metrics = eval_res_dic["metrics"]
            metrics_dic[key] = update_metrics_dict(metrics_dic[key], *metrics)
            # fold k のモデルの各keyのdlに対する正解/不正解のリストを保存
            correctness_arr = eval_res_dic["correctness_arr"]
            np.save(os.path.join(correctness_dir, f"{key}-fold-{k+1}.npy"), correctness_arr)
        

    # メトリクスの辞書をcsvにして保存しておく
    all_metrics_dict = {}
    for key, m in metrics_dic.items():
        all_metrics_dict[key] = dict(m)
    csv_file_name = log_file_name.replace("setting", "result-setting")
    save_path = os.path.join(exp_dir, f"{csv_file_name}.csv")
    df = make_df_metrics(all_metrics_dict, num_fold)
    df.to_csv(save_path, index=False)
    logger.info(f"saved to {save_path}")