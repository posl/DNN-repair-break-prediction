import os, sys, pickle
from lib.dataset import divide_train_repair, TableDataset
from torch.utils.data import DataLoader, TensorDataset
from lib.model import train_model, select_model
from lib.util import json2dict, dataset_type
from lib.log import set_exp_logging
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns

# plot setting
sns.set()


def visualize_train_loss_fold(num_epochs, epoch_loss_list_list, save_path):
    """各foldのエポックごとのロスを可視化してpdfに保存する

    Args:
        num_epochs (int): エポック数
        epoch_loss_list_list (list): 各foldのモデルの各エポックのロスのリスト
        save_path (str): 図を保存するパス
    """

    for k, epoch_loss_list in enumerate(epoch_loss_list_list):
        plt.plot(range(num_epochs), epoch_loss_list, label=f"fold {k+1}")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(save_path)
    logger.info(f"saved in {save_path}")


if __name__ == "__main__":
    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(sys.argv[1])
    exp_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    # log setting
    logger = set_exp_logging(exp_dir, exp_name)

    # 設定用のjsonファイルをdictとしてロード
    # HACK: 共通しているので関数にまとめて自動化したい
    setting_dict = json2dict(sys.argv[1])
    logger.info(f"Settings: {setting_dict}")

    task_name = setting_dict["TASK_NAME"]
    num_epochs = setting_dict["NUM_EPOCHS"]
    batch_size = setting_dict["BATCH_SIZE"]
    num_fold = setting_dict["NUM_FOLD"]
    data_dir = f"/src/data/{task_name}"

    # tabular datasetの場合
    if dataset_type(task_name) == "tabular":
        # tabular dataset専用の項目
        train_repair_data_path = setting_dict["TRAIN-REPAIR_DATA_PATH"]
        test_data_path = setting_dict["TEST_DATA_PATH"]
        target_column = setting_dict["TARGET_COLUMN"]
        # csvからdataset型にする
        train_repair_dataset = TableDataset(train_repair_data_path, target_column)
        test_dataset = TableDataset(test_data_path, target_column)

    # image datasetの場合
    elif dataset_type(task_name) == "image":
        # ディレクトリ作成
        raw_data_dir = os.path.join(data_dir, "raw_data")
        os.makedirs(raw_data_dir, exist_ok=True)

        # 各データセットをロード
        if task_name == "fm":
            train_repair_dataset = datasets.FashionMNIST(
                root=raw_data_dir, train=True, download=True, transform=transforms.ToTensor()
            )
            test_dataset = datasets.FashionMNIST(
                root=raw_data_dir, train=False, download=True, transform=transforms.ToTensor()
            )

        elif task_name == "c10":
            train_repair_dataset = datasets.CIFAR10(
                root=raw_data_dir, train=True, download=True, transform=transforms.ToTensor()
            )
            test_dataset = datasets.CIFAR10(
                root=raw_data_dir, train=False, download=True, transform=transforms.ToTensor()
            )

        elif task_name == "gtsrb":
            # NOTE: gtsrbに関してはraw_dataはArachneで利用されてたものをコピーしてきたのでここでダウンロードしなくてよい. 代わりに, PyTorchのDataset型にする
            # train, repair set
            with open(os.path.join(raw_data_dir, "train_data.pkl"), "rb") as f:
                train_repair_dict = pickle.load(f)
            train_repair_dict["data"] = torch.from_numpy(train_repair_dict["data"].astype(np.float32)).clone()
            train_repair_dict["label"] = (
                torch.from_numpy(train_repair_dict["label"].astype(np.int32)).clone().type(torch.LongTensor)
            )
            train_repair_dataset = TensorDataset(train_repair_dict["data"], train_repair_dict["label"])
            # test set
            with open(os.path.join(raw_data_dir, "test_data.pkl"), "rb") as f:
                test_dict = pickle.load(f)
            test_dict["data"] = torch.from_numpy(test_dict["data"].astype(np.float32)).clone()
            test_dict["label"] = torch.from_numpy(test_dict["label"].astype(np.int32)).clone().type(torch.LongTensor)
            test_dataset = TensorDataset(test_dict["data"], test_dict["label"])
        else:
            raise ValueError(f"dataset {task_name} is not supported.")

    # train_repairをK-foldでtrainとrepairに分けて，dataloader型にする
    train_loader_list, repair_loader_list = divide_train_repair(
        train_repair_dataset, num_fold=num_fold, batch_size=batch_size, dataset_type=dataset_type(task_name)
    )
    # testもdataloader型にする
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info(
        f"train : repair : test = {len(train_loader_list[0].dataset)} : {len(repair_loader_list[0].dataset)} : {len(test_loader.dataset)}"
    )

    # 分割後のデータを保存するディレクトリの作成
    data_save_dir = os.path.join(data_dir, exp_name)
    os.makedirs(data_save_dir, exist_ok=True)

    # train/repair loaderの保存
    for k, (train_loader, repair_loader) in enumerate(zip(train_loader_list, repair_loader_list)):
        # train loaderの保存
        train_loader_save_path = os.path.join(data_save_dir, f"train_loader_fold-{k}.pt")
        torch.save(train_loader, train_loader_save_path)
        logger.info(f"saved train_loader in {train_loader_save_path}")

        # repair loaderの保存
        repair_loader_save_path = os.path.join(data_save_dir, f"repair_loader_fold-{k}.pt")
        torch.save(repair_loader, repair_loader_save_path)
        logger.info(f"saved repair_loader in {repair_loader_save_path}")

    # test loaderの保存
    test_loader_save_path = os.path.join(data_save_dir, "test_loader.pt")
    torch.save(test_loader, test_loader_save_path)
    logger.info(f"saved test_loader in {test_loader_save_path}")

    # 各fold (train) に対してモデルを訓練し，各モデルを保存
    model_list = []
    epoch_loss_list_list = []

    # 保存用のディレクトリ
    model_save_dir = f"/src/models/{task_name}/{exp_name}"
    os.makedirs(model_save_dir, exist_ok=True)

    # 訓練->保存を各foldについて繰り返す
    for k, train_loader in enumerate(train_loader_list):
        logger.info(f"training with fold {k}...")
        model_save_path = os.path.join(model_save_dir, f"trained_model_fold-{k}.pt")
        model = select_model(task_name)
        # 訓練
        trained_model, epoch_loss_list = train_model(model=model, dataloader=train_loader, num_epochs=num_epochs)
        # 保存
        torch.save(trained_model.state_dict(), model_save_path)
        logger.info(f"saved model in {model_save_path}")
        # リストに追加
        model_list.append(train_model)
        epoch_loss_list_list.append(epoch_loss_list)

    # 各foldのモデルのepochごとのlossを可視化して保存
    visualize_train_loss_fold(
        num_epochs=num_epochs,
        epoch_loss_list_list=epoch_loss_list_list,
        save_path=os.path.join(exp_dir, f"{exp_name}.pdf"),
    )
