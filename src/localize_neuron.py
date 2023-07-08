import os, sys
from collections import defaultdict

import numpy as np
from lib.model import select_model
from lib.fairness import calc_average_causal_effect
from lib.util import json2dict
from lib.log import set_exp_logging
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# plot setting
sns.set_style("white")


def plot_fl_score(fl_score, exp_name, k):
    """各レイヤ, ニューロンにおけるFLスコアをプロットして保存する.
    プロットはPDF形式で保存される.

    Args:
        fl_score (dict): (レイヤ番号, ニューロン番号)=>FLスコア(ACE)への対応辞書.
        exp_name (str): 実験ファイルの名前 ({dataset}-{task}-{sens_feature}-setting{NO})
        k (int): num of folds
    """
    save_path = f"/src/experiments/repair_results/flscore_{exp_name}_fold{k+1}.pdf"
    val_arr = np.array(list(fl_score.values()))

    # 各レイヤの最終ニューロンの通し番号のリスト
    # FIXME: 色んなNNに対応できない
    nids = [63, 63 + 32, 63 + 32 + 16, 63 + 32 + 16 + 8, 63 + 32 + 16 + 8 + 4]

    plt.figure()
    plt.xlabel("(lid, nid)")
    plt.ylabel("ACE")
    plt.xticks(
        nids,
        ["(0, 63)", "(1, 31)", "(2, 15)", "(3, 7)", "(4, 3)"],
        rotation=90,
    )
    plt.scatter(list(fl_score.keys()), val_arr, c="black")
    for nid in nids:
        plt.axvline(x=nid, c="gray", linestyle="dashed")
    plt.savefig(save_path, bbox_inches="tight")
    logger.info(f"saved in {save_path}")


if __name__ == "__main__":
    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(sys.argv[1])
    exp_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    # log setting
    # {dataset}-localize-fairness-{feature}-setting{NO}.logというファイルにログを出す
    log_file_name = exp_name.replace("fairness", "localize-fairness")
    # ロガーの取得
    logger = set_exp_logging(exp_dir, exp_name, log_file_name)

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
    sens_vals = eval(setting_dict["SENS_VALS"])  # ない場合はNone, ある場合は直接listで定義したりrangeで定義できるようにする. そのためにevalを使う
    target_cls = setting_dict["TARGET_CLS"]

    # localizationのための情報をパース
    num_steps = setting_dict["NUM_STEPS"]
    # これらの値は数値かリストで指定可能にするためにevalする
    target_lids = setting_dict["TARGET_LAYER"]
    target_nids = setting_dict["TARGET_NEURON"]
    # 数値の場合は要素数1のリストにする
    if isinstance(target_lids, int):
        target_lids = [target_lids]
    if isinstance(target_nids, int):
        target_nids = [target_nids]
    assert len(target_lids) == len(
        target_nids
    ), f"Error: len(target_lid)({len(target_lids)}) != len(target_nid)({len(target_nids)})"

    # 訓練時の設定も読み込む
    train_setting_dict = json2dict(os.path.join(exp_dir, train_setting_path))
    logger.info(f"TRAIN Settings: {train_setting_dict}")
    num_fold = train_setting_dict["NUM_FOLD"]
    task_name = train_setting_dict["TASK_NAME"]

    # モデルとデータの読み込み先のディレクトリ
    data_dir = f"/src/data/{task_name}/{train_setting_name}"
    model_dir = f"/src/models/{task_name}/{train_setting_name}"

    # 各foldのtrain/repairをロードして予測
    for k in range(num_fold):
        logger.info(f"processing fold {k}...")

        # 学習済みモデルをロード
        model = select_model(task_name=task_name)
        model_path = os.path.join(model_dir, f"trained_model_fold-{k}.pt")
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # foldに対するdataloaderをロード
        # NOTE: repairだけ使うことにしてみる
        repair_data_path = os.path.join(data_dir, f"repair_loader_fold-{k}.pt")
        repair_loader = torch.load(repair_data_path)
        repair_ds = repair_loader.dataset

        fl_score = defaultdict(float)
        # 計算したい各レイヤ/ニューロンに対してFLのスコア（Average Causal Effects）を算出
        for target_lid, target_nids_for_layer in zip(target_lids, target_nids):
            # 対象のレイヤに対する各ニューロンの順伝搬の値を計算
            layer_dist = model.get_layer_distribution(repair_ds, target_lid=target_lid)

            for target_nid in target_nids_for_layer:
                logger.info(f"target layer={target_lid}, target neuron={target_nid}")

                hdist = layer_dist[:, target_nid]
                hmin, hmax = min(hdist), max(hdist)
                hvals = np.linspace(hmin, hmax, num_steps)
                logger.info(f"hmin={hmin}, hmax={hmax},\nhvals={hvals}")

                repair_fairness_list = calc_average_causal_effect(
                    model, repair_loader, sens_idx, target_lid, target_nid, hvals, sens_vals, target_cls
                )
                fl_score[f"({target_lid},{target_nid})"] = np.mean(repair_fairness_list)
        # ニューロンごとのfl_scoreのプロットを保存
        plot_fl_score(fl_score, exp_name, k)
        # fl_scoreを降順に並び替えて保存する
        fl_score_sorted = np.array(sorted(fl_score.items(), key=lambda v: v[1], reverse=True))
        logger.info(f"SORTED FL SCORE\n{fl_score_sorted}")
        fl_score_save_path = f"/src/experiments/repair_results/flscore_{exp_name}_fold{k+1}.npy"
        np.save(fl_score_save_path, fl_score_sorted)
        logger.info(f"saved to {fl_score_save_path}")
