import json
import torch

# for logging
from logging import getLogger

logger = getLogger("base_logger")


def json2dict(json_path):
    """jsonを読み込んでdictにして返す.

    Args:
        json_path (str): jsonファイルのパス（拡張子込みで）.

    Returns:
        dict: jsonをパースしたdict.
    """
    with open(json_path) as f:
        json_dict = json.load(f)
        return json_dict


def dict2json(dic, save_path):
    """dictをjsonとして保存する

    Args:
        dic (dict): 保存したい辞書.
        save_path (str): 保存するパス（拡張子込みで）.
    """
    with open(save_path, "w") as f:
        json.dump(dic, f, indent=4, separators=(",", ":"))
    logger.info(f"saved to {save_path}")


def dataset_type(dataset):
    """datasetの名前から, そのdatasetドメイン(tabular, image, text)を返す.

    Args:
        dataset (str): dataset名.

    Returns:
        str: tabular, image, textのいずれか.
    """
    # datasetがcensus, credit, bankのいずれかであればtabularと返す
    if dataset in ["census", "credit", "bank"]:
        return "tabular"
    elif dataset in ["fm", "c10", "gtsrb"]:
        return "image"
    else:
        raise ValueError(f"dataset {dataset} is not supported.")


def fix_dataloader(dataloader):
    """入力のdataloaderに対し, バッチのランダム性を排除したdataloaderを作成する.

    Args:
        dataloader (torch.DataLoader): 入力のdataloaderオブジェクト.

    Returns:
        torch.DataLoader: 入力のdataloaderのバッチの順番を固定したdataloader (実行のたびに同じデータを返す).
    """
    return torch.utils.data.DataLoader(
        dataset=dataloader.dataset, batch_size=dataloader.batch_size, shuffle=False, num_workers=2
    )

def keras_lid_to_torch_layers(task_name, model):
    """
    kerasモデルのレイヤのインデックスとtorchのレイヤとの対応を取るための辞書
    """
    if task_name == "fm":
        return {
            1: model.conv1, 3: model.conv2, 7: model.dense1, 9: model.dense2
        }
    elif task_name == "gtsrb":
        return {
            1: model.conv1, 4: model.conv2, 7: model.conv3, 12: model.dense1, 14: model.dense2
        }
    elif task_name == "c10":
        return {
            1: model.conv1, 2: model.conv2, 4: model.conv3, 5: model.conv4, 9: model.dense1, 10: model.dense2, 11: model.dense3
        }
    else:
        assert False