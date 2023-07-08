import os, sys
from collections import defaultdict
import pandas as pd
from lib.model import select_model
from lib.fairness import eval_independence_fairness
from lib.util import json2dict
from lib.log import set_exp_logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# plot setting
sns.set()


def make_df_metrics(fairness_dict, num_fold):
    """モデルの各foldごとのfairnessをdfにまとめる.

    (Extended Summary)
    | division | fold_0 | ... | fold_{num_fold} |
    |----------|--------| ... |-----------------|

    Args:
        fairness_dict (dict): 各division (train, repair, testなど) の各foldのfairnessを保存したdict.
        num_fold (int): fold数.

    Returns:
        pd.DataFrame: 上記のフォーマットの表
    """
    # foldを表す列名を作成
    columns = ["division"]
    fold_cols = [f"fold_{k}" for k in range(num_fold)]
    columns.extend(fold_cols)
    # dfの定義
    df = pd.DataFrame(columns=columns)

    # dfに行を追加していく
    for div, f_list in fairness_dict.items():
        row = [div]
        row.extend(f_list)
        row_df = pd.DataFrame(data=[row], columns=columns)
        df = pd.concat([df, row_df], axis=0)
    logger.info(f"df.columns={df.columns}. df.shape={df.shape}")
    return df


if __name__ == "__main__":
    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(sys.argv[1])
    exp_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    # log setting
    # {dataset}-fairness-{feature}-setting{NO}.logというファイルにログを出す
    log_file_name = exp_name
    logger = set_exp_logging(exp_dir, exp_name)

    # 設定用のjsonファイルをdictとしてロード
    # HACK: 共通しているので関数にまとめて自動化したい
    setting_dict = json2dict(sys.argv[1])
    logger.info(f"Settings: {setting_dict}")

    train_setting_path = setting_dict["TRAIN_SETTING_PATH"]
    # 訓練時の設定名を取得
    train_setting_name = os.path.splitext(train_setting_path)[0]
    # fairnessの計算のための情報をパース
    sens_name = setting_dict["SENS_NAME"]
    sens_idx = setting_dict["SENS_IDX"]
    sens_vals = eval(setting_dict["SENS_VALS"])  # ない場合はNone, ある場合は直接listで定義したりrangeで定義できるようにする. しのためにevalを使う
    target_cls = setting_dict["TARGET_CLS"]

    # 訓練時の設定も読み込む
    train_setting_dict = json2dict(os.path.join(exp_dir, train_setting_path))
    logger.info(f"TRAIN Settings: {train_setting_dict}")
    num_fold = train_setting_dict["NUM_FOLD"]
    task_name = train_setting_dict["TASK_NAME"]

    # モデルとデータの読み込み先のディレクトリ
    data_dir = f"/src/data/{task_name}/{train_setting_name}"
    model_dir = f"/src/models/{task_name}/{train_setting_name}"

    # test dataloaderをロード (foldに関係ないので先にロードする)
    test_data_path = os.path.join(data_dir, f"test_loader.pt")
    test_loader = torch.load(test_data_path)

    # train/repair/testの各foldのfairnessを保存するdefaultdict
    fairness_dict = defaultdict(list)

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

        # fairnessの計算
        logger.info("training set")
        train_fairness = eval_independence_fairness(
            model=model, dataloader=train_loader, sens_idx=sens_idx, sens_vals=sens_vals, target_cls=target_cls
        )[0]
        logger.info("repair set")
        repair_fairness = eval_independence_fairness(
            model=model, dataloader=repair_loader, sens_idx=sens_idx, sens_vals=sens_vals, target_cls=target_cls
        )[0]
        logger.info("test set")
        test_fairness = eval_independence_fairness(
            model=model, dataloader=test_loader, sens_idx=sens_idx, sens_vals=sens_vals, target_cls=target_cls
        )[0]

        # fairnessの記録
        fairness_dict["train"].append(train_fairness)
        fairness_dict["repair"].append(repair_fairness)
        fairness_dict["test"].append(test_fairness)

        # fairnessを表示
        logger.info(
            f"train fairness: {train_fairness:.4f}, repair fairness: {repair_fairness:.4f}, test fairness: {test_fairness:.4f}"
        )
    # 最終的なdictを表示
    logger.info(f"fairness_dict={fairness_dict}")

    # fairnessの辞書をcsvにして保存しておく
    csv_file_name = log_file_name.replace("setting", "result-setting")
    save_path = os.path.join(exp_dir, f"{csv_file_name}.csv")
    df = make_df_metrics(fairness_dict, num_fold)
    df.to_csv(save_path, index=False)
    logger.info(f"saved to {save_path}")
