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