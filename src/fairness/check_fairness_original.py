import os, sys, time, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
import pandas as pd
from lib.model import select_model
from lib.fairness import eval_independence_fairness
from lib.util import json2dict
from lib.log import set_exp_logging
import torch
import numpy as np

DS_LIST = ["census", "credit", "bank"]

if __name__ == "__main__":
    # gpuが使用可能かをdeviceにセット
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DS_LIST, help=f"dataset name should be in {DS_LIST}")
    args = parser.parse_args()
    ds = args.dataset
    # log setting
    care_dir = "/src/experiments/care/"
    this_file_name = os.path.basename(__file__).replace(".py", "").replace("_", "-")
    log_file_name = f"{ds}-{this_file_name}"
    logger = set_exp_logging(care_dir, log_file_name, log_file_name)
    logger.info(f"target dataset: {ds}")

    # set sensitive features name
    if ds == "census":
        sensitive_feat_name = "gender"
    elif ds == "credit":
        sensitive_feat_name = "gender"
    elif ds == "bank":
        sensitive_feat_name = "age"
    # path to json files for sensitive related information
    json_file_name = f"{ds}-fairness-{sensitive_feat_name}-setting1.json"
    json_path = os.path.join(care_dir, json_file_name)
    setting_dict = json2dict(json_path)
    logger.info(f"Settings: {setting_dict}")

    # 訓練時の設定名を取得
    train_setting_path = setting_dict["TRAIN_SETTING_PATH"]
    train_setting_name = os.path.splitext(train_setting_path)[0]
    # fairnessの計算のための情報をパース
    sens_idx = setting_dict["SENS_IDX"]
    # sens_vals = eval(setting_dict["SENS_VALS"])  # ない場合はNone, ある場合は直接listで定義したりrangeで定義できるようにする. しのためにevalを使う
    # NOTE: sens_valsを使わず実際のデータから値の候補を取得することにした (カテゴリカルデータも標準化されてしまってるので)
    target_cls = setting_dict["TARGET_CLS"]

    # 訓練時の設定も読み込む
    train_setting_dict = json2dict(os.path.join(care_dir, train_setting_path))
    logger.info(f"TRAIN Settings: {train_setting_dict}")
    num_fold = train_setting_dict["NUM_FOLD"]
    task_name = train_setting_dict["TASK_NAME"]

    # モデルとデータの読み込み先のディレクトリ
    data_dir = f"/src/data/{task_name}/{train_setting_name}"
    model_dir = f"/src/models/{task_name}/{train_setting_name}"

    # test dataloaderをロード (foldに関係ないので先にロードする)
    test_data_path = os.path.join(data_dir, f"test_loader.pt")
    test_loader = torch.load(test_data_path)

    # fairnessに関するsample_metricsの保存ディレクトリ
    # json_file_nameの.jsonを省いた部分を取得
    json_file_base_name = os.path.splitext(json_file_name)[0]
    fairness_dir = os.path.join("/src/src/fairness/fairness_before", ds)
    os.makedirs(fairness_dir, exist_ok=True)

    # 各foldのtrain/repairをロードして予測
    for k in range(num_fold):
        logger.info(f"processing fold {k}...")
        # train/repair/testの各foldのfairnessを保存するdefaultdict
        fairness_dict = defaultdict(list)

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

        # sensitive featuresが一列だけ対象の場合
        if isinstance(sens_idx, int):
            tgt_vals = train_loader.dataset.tensors[0][:, sens_idx]
            # tgt_valsをnumpyにしてunique valuesを表示
            sens_vals = np.unique(tgt_vals.numpy())
            logger.info(f"list of sens_vals: {sens_vals} (len={len(sens_vals)})")
            # sens_valsをtorch tensorにしておく
            sens_vals = torch.tensor(sens_vals)
        else:
            sens_vals = None

        # fairnessの計算
        # *_fairness_arrayはデータセットの各サンプルに対するfairnessを格納した配列
        logger.info("training set")
        fairness_dict["train"] = eval_independence_fairness(
            model=model, dataloader=train_loader, sens_idx=sens_idx, sens_vals=sens_vals, target_cls=target_cls
        )[1]
        logger.info("repair set")
        fairness_dict["repair"] = eval_independence_fairness(
            model=model, dataloader=repair_loader, sens_idx=sens_idx, sens_vals=sens_vals, target_cls=target_cls
        )[1]
        logger.info("test set")
        fairness_dict["test"] = eval_independence_fairness(
            model=model, dataloader=test_loader, sens_idx=sens_idx, sens_vals=sens_vals, target_cls=target_cls
        )[1]

        # csvで保存
        for div in ["train", "repair", 'test']:
            save_path = os.path.join(fairness_dir, f"{div}_fold{k+1}.npy")
            np.save(save_path, fairness_dict[div])
            logger.info(f"saved to {save_path}")
            # fairness_dict[div]の平均, 最小, 最大値をログに残す」
            logger.info(f"len: {len(fairness_dict[div])}, mean: {fairness_dict[div].mean()}, min: {fairness_dict[div].min()}, max: {fairness_dict[div].max()}")
