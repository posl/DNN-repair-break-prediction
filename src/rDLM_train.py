import os, sys
import warnings
warnings.filterwarnings('ignore')

from lib.log import set_exp_logging
from lib.util import json2dict
from lib.model import train_model, select_model
from lib.dataset import BalancedSubsetDataLoader

from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image

# Hyper-parameters
num_restrict = 1000 # rDLMを作るための縮小データセットを作る際に，各ラベルからどれだけサンプルするか
rDLM_num = 20
batch_size = 64


if __name__=="__main__":
    """
    {dataset名}-training-setting{sid}を入力として必要な情報の準備
    """

    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(sys.argv[1])
    exp_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]

    # log setting
    # {dataset}-repair-fairness-{feature}-setting{NO}.logというファイルにログを出す
    log_file_name = exp_name.replace("training", "apply-apricot")
    logger = set_exp_logging(exp_dir.replace("care", "apricot"), exp_name, log_file_name)

    # 設定用のjsonファイルをdictとしてロード
    # HACK: 共通しているので関数にまとめて自動化したい
    setting_dict = json2dict(sys.argv[1])
    logger.info(f"Settings: {setting_dict}")
    task_name = setting_dict["TASK_NAME"]
    train_repair_data_path = setting_dict["TRAIN-REPAIR_DATA_PATH"]
    test_data_path = setting_dict["TEST_DATA_PATH"]
    target_column = setting_dict["TARGET_COLUMN"]
    num_epochs = setting_dict["NUM_EPOCHS"]
    num_epochs_rdlm = num_epochs // 4 # rDLM訓練用のエポック数
    batch_size = setting_dict["BATCH_SIZE"]
    num_fold = setting_dict["NUM_FOLD"]

    # モデルとデータの読み込み先のディレクトリ
    data_dir = f"/src/data/{task_name}/{exp_name}"
    model_dir = f"/src/models/{task_name}/{exp_name}"

    # rDLMの保存先として, models/以下の各設定のディレクトリにrDLM用のディレクトリを作る
    rdlm_dir = os.path.join(model_dir, "rDLM")
    os.makedirs(rdlm_dir, exist_ok=True)

    # 各foldのtrain setを使ってrDLMを作成する
    for k in range(num_fold):
        logger.info(f"processing fold {k}...")

        # foldに対するtrain dataloaderをロード
        train_data_path = os.path.join(data_dir, f"train_loader_fold-{k}.pt")
        train_loader = torch.load(train_data_path)

        # ラベルの頻度を集計
        labels = np.array(train_loader.dataset)[:, 1] # ここでwarning出るけど動くからOK
        _, cnts = np.unique(labels, return_counts=True)
        logger.info(f"cnts={cnts}")
        # 最も頻度が少ないラベルのサンプル数の半分を，reduced Datasetを作る際の各ラベルからのサンプル数とする
        num_samples_per_class = np.min(cnts) // 2
        logger.info(f"num_samples_per_class={num_samples_per_class}")
        # NOTE: あんまりサンプル数少なすぎるとまずいのでassertion入れる (TODO: 不均衡データセットへの対応)
        assert num_samples_per_class >= 50, "num_samples_per_class must be greater than or equal to 50"

        # rDLMの数だけ繰り返し
        for rdlm_idx in range(rDLM_num):
            logger.info(f"processing rdlm_idx {rdlm_idx}...")
            
            # モデルの雛形をインスタンス化
            model = select_model(task_name=task_name)
            rdlm_path = os.path.join(rdlm_dir, f"trained_model_fold-{k}_rDLM-{rdlm_idx}.pt")

            # オリジナルのデータセットから各ラベル num_samples_per_class だけサンプリングして新たなデータセットを作成
            # NOTE: 各ラベルのサンプリングにランダム性があるためrDLMごとに異なるデータセットが作られる
            balanced_subset_dataloader = BalancedSubsetDataLoader(train_loader, num_samples_per_class, batch_size=batch_size, shuffle=True)
            logger.info(f"reduced dataset size={len(balanced_subset_dataloader.dataset)}")

            # 各rDLMを訓練して保存する
            trained_rdlm, epoch_loss_list = train_model(model=model, dataloader=balanced_subset_dataloader, num_epochs=num_epochs_rdlm)
            logger.info(f"final loss={epoch_loss_list[-1]}")
            torch.save(trained_rdlm.state_dict(), rdlm_path)
            logger.info(f"saved model in {rdlm_path}")
