import os, sys
from collections import defaultdict
import pandas as pd
from lib.model import select_model, eval_model
from lib.util import json2dict
from lib.log import set_exp_logging
import torch
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
    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(sys.argv[1])
    exp_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    # log setting
    # prepare_dataset_modelとは別のファイルにログを出すようにする
    # HACK: exp_nameにtrainingが含まれてないといけない
    log_file_name = exp_name.replace("training", "checking")
    logger = set_exp_logging(exp_dir, exp_name, log_file_name)

    # 設定用のjsonファイルをdictとしてロード
    # HACK: 共通しているので関数にまとめて自動化したい
    setting_dict = json2dict(sys.argv[1])
    logger.info(f"Settings: {setting_dict}")

    task_name = setting_dict["TASK_NAME"]
    target_column = setting_dict["TARGET_COLUMN"]
    num_epochs = setting_dict["NUM_EPOCHS"]
    batch_size = setting_dict["BATCH_SIZE"]
    num_fold = setting_dict["NUM_FOLD"]

    # モデルとデータの読み込み先のディレクトリ
    data_dir = f"/src/data/{task_name}/{exp_name}"
    model_dir = f"/src/models/{task_name}/{exp_name}"

    # test dataloaderをロード (foldに関係ないので先にロードする)
    test_data_path = os.path.join(data_dir, f"test_loader.pt")
    test_loader = torch.load(test_data_path)

    # train/repairの各foldの各メトリクスを保存するdefaultdict
    train_metrics_dict = defaultdict(list)
    repair_metrics_dict = defaultdict(list)
    test_metrics_dict = defaultdict(list)

    # 各foldのtrain/repairをロードして予測
    for k in range(num_fold):
        logger.info(f"processing fold {k}...")

        # 学習済みモデルをロード
        model = select_model(task_name=task_name)
        model_path = os.path.join(model_dir, f"trained_model_fold-{k}.pt")
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # foldに対するdataloaderをロード
        train_data_path = os.path.join(data_dir, f"train_loader_fold-{k}.pt")
        repair_data_path = os.path.join(data_dir, f"repair_loader_fold-{k}.pt")
        train_loader = torch.load(train_data_path)
        repair_loader = torch.load(repair_data_path)

        # evaluation criteriaの計算
        train_metrics = eval_model(model, train_loader)["metrics"]
        repair_metrics = eval_model(model, repair_loader)["metrics"]
        test_metrics = eval_model(model, test_loader)["metrics"]

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
