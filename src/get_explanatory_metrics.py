import os, sys
from collections import defaultdict
import pandas as pd
from lib.model import select_model
from lib.explanatory_metrics import get_pcs, get_entropy, get_lps, get_loss
from lib.util import json2dict
from lib.log import set_exp_logging
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# plot setting
sns.set()

if __name__ == "__main__":
    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(sys.argv[1])
    exp_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    # log setting
    # prepare_dataset_modelとは別のファイルにログを出すようにする
    # HACK: exp_nameにtrainingが含まれてないといけない
    log_file_name = exp_name.replace("training", "get-metrics")
    logger = set_exp_logging(exp_dir, exp_name, log_file_name)

    # 設定用のjsonファイルをdictとしてロード
    # HACK: 共通しているので関数にまとめて自動化したい
    setting_dict = json2dict(sys.argv[1])
    logger.info(f"Settings: {setting_dict}")

    task_name = setting_dict["TASK_NAME"]
    # target_column = setting_dict["TARGET_COLUMN"]
    # num_epochs = setting_dict["NUM_EPOCHS"]
    # batch_size = setting_dict["BATCH_SIZE"]
    num_fold = setting_dict["NUM_FOLD"]

    # exp. metrics保存用のディレクトリを作成
    expmet_dir = os.path.join(exp_dir, "explanatory_metrics", exp_name)
    os.makedirs(expmet_dir, exist_ok=True)

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

        # 各サンプルをモデルに入力して予測確率の配列 (サンプル数, クラス数) を得る
        for div_name, dataloader in zip(["train", "repair", "test"], [train_loader, repair_loader, test_loader]):
            # fold, division が決まるのでここでdataframe作る
            df = pd.DataFrame(columns=["pcs", "lps", "entropy", "loss"])
            for x, y in dataloader.dataset:
                row_dict = {}
                out = model.predict(torch.unsqueeze(x, 0))
                prob = out["prob"][0]
                # probを使ったexplanatory metricsの計算
                row_dict["pcs"] = get_pcs(prob)
                row_dict["entropy"] = get_entropy(prob)
                row_dict["lps"] = get_lps(prob, torch.tensor(y))
                row_dict["loss"] = get_loss(prob, torch.tensor(y))
                df = df.append(row_dict, ignore_index=True)
            # 保存先を指定してsaveする
            save_path = os.path.join(expmet_dir, f"{div_name}_fold{k+1}.csv")
            df.to_csv(save_path, index=False)
