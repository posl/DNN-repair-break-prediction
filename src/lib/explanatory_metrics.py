import math
import numpy as np
from torch.nn import CrossEntropyLoss

"""
explanatory metricsを得るための関数群.
予測確率の配列 (クラス数,) の形状を入力として, サンプルに対するexplanatory metricsの値を返す関数たち.
"""


def _log2(x):
    try:
        return math.log2(x)
    except ValueError:
        return -math.inf


def get_pcs(prob):
    """PCSを計算. 予測確率の最大値から2番目の値を引いたもの.

    Args:
        prob (list of float): 予測確率の配列.

    Returns:
        float: PCSの値
    """
    top2_score, top1_score = np.partition(prob.detach().numpy(), -2)[-2:]
    return top1_score - top2_score


def get_entropy(prob):
    """Shannon entropyを計算.

    Args:
        prob (list of float): 予測確率の配列.

    Returns:
        float: entropyの値
    """
    entropy = -sum([p * _log2(p) if p != 0 else 0 for p in prob]).item()
    return entropy


def get_lps(prob, label):
    """Label Prediction Score (LPS; 正解ラベルへの予測確率を返す)

    Args:
        prob (list of float): 予測確率の配列
        label (int): 正解ラベル

    Returns:
        float: LPSの値
    """
    return prob[label].item()


def get_loss(prob, label):
    loss_fn = CrossEntropyLoss()
    return loss_fn(prob, label).item()
