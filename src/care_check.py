import os, sys

sys.path.append(os.pardir)
from src.lib.model import select_model, eval_model
from src.lib.util import json2dict
from src.lib.log import set_exp_logging
from src.lib.fairness import eval_independence_fairness
import numpy as np

np.random.seed(0)  # 乱数固定
from collections import defaultdict
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns

# plot setting
sns.set_style("white")


def make_diff_df(diff_dict, num_fold=5):
    """モデルの各foldごとのaccやfairnessのdiffをdfにまとめる.

    (Extended Summary)
    | division | fold | diff_acc | diff_fairness |
    |----------|------|----------|---------------|

    Args:
        diff_dict (dict): 各division (train, repair, testなど) / metrics (acc, fairness) の各foldのdiffを保存したdict.
        num_fold (int): fold数.

    Returns:
        pd.DataFrame: 上記のフォーマットの表
    """
    # 列名を作成し, dfを定義
    columns = ["division", "fold", "diff_acc", "diff_fairness"]
    df = pd.DataFrame(columns=columns)

    # dfに行を追加していく
    for (met, div), d_list in diff_dict.items():
        row = [met, div]
        row.extend(d_list)
        row_df = pd.DataFrame(data=[row], columns=columns)
        df = pd.concat([df, row_df], axis=0)
    logger.info(f"df.columns={df.columns}. df.shape={df.shape}")
    return df


def make_sm_df(sm_dict):
    """モデルの各division/foldごとに修正前後のsample metricsをサンプルごとに記した表を作成.

    (Extended Summary)
    File Name = {division}_fold{fold_id}.csv
    | sample_id (surrogate) | sm_corr_bef | sm_corr_aft | sm_fair_bef | sm_fair_aft |
    |-----------------------|-------------|-------------|-------------|-------------|

    Args:
        sm_dict (dict): 各division (train, repair, testなど) / 各foldにおける, sampleごとのsample metricsを保存したdict.

    Returns:
        pd.DataFrame: 上記のフォーマットの表
    """
    # 列名を作成し, dfを定義
    columns = ["sm_corr_bef", "sm_corr_aft", "sm_fair_bef", "sm_fair_aft"]
    df = pd.DataFrame(columns=columns)

    # dfに行を追加していく
    for col in columns:
        df[col] = sm_dict[col]
    logger.info(f"df.columns={df.columns}. df.shape={df.shape}")
    return df


if __name__ == "__main__":
    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(sys.argv[1])
    exp_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    # log setting
    # {dataset}-repair-check-{feature}-setting{NO}.logというファイルにログを出す
    log_file_name = exp_name.replace("fairness", "repair-check")
    logger = set_exp_logging(exp_dir, exp_name, log_file_name)

    # sample metrics保存用のディレクトリを作成
    sm_dir = os.path.join(exp_dir, "sample_metrics", log_file_name)
    os.makedirs(sm_dir, exist_ok=True)

    # 設定用のjsonファイルをdictとしてロード
    # HACK: 共通しているので関数にまとめて自動化したい
    org_setting_dict = json2dict(sys.argv[1])
    logger.info(f"Settings: {org_setting_dict}")

    # INHERIT_SETTINGが設定されている場合はまず継承先の設定をロード
    is_inherit = False
    try:
        inherit_setting_path = org_setting_dict["INHERIT_SETTING"]
    # INHERIT_SETTINGなしの場合
    except KeyError as e:
        setting_path = sys.argv[1]
        # 実際に読み込むべき設定のロード
        setting_dict = json2dict(setting_path)
    # INHERIT_SETTINGありの場合
    else:
        is_inherit = True
        setting_path = os.path.join(exp_dir, inherit_setting_path)
        setting_dict = json2dict(setting_path)
        inherit_exp_name = os.path.splitext(os.path.basename(setting_path))[0]
        logger.info(f"Inherit from {inherit_exp_name}")

    train_setting_path = setting_dict["TRAIN_SETTING_PATH"]
    # 訓練時の設定名を取得
    train_setting_name = os.path.splitext(train_setting_path)[0]

    # fairnessの計算のための情報をパース
    sens_name = setting_dict["SENS_NAME"]
    sens_idx = setting_dict["SENS_IDX"]
    sens_vals = eval(setting_dict["SENS_VALS"])  # ない場合はNone, ある場合は直接listで定義したりrangeで定義できるようにする. そのためにevalを使う
    target_cls = setting_dict["TARGET_CLS"]

    # 訓練時の設定も読み込む
    train_setting_dict = json2dict(os.path.join(exp_dir, train_setting_path))
    logger.info(f"TRAIN Settings: {train_setting_dict}")
    num_fold = train_setting_dict["NUM_FOLD"]
    task_name = train_setting_dict["TASK_NAME"]

    # リペアする割合 (疑惑値の上位何％をrepair対象にするか)
    # repair_ratioの型がintの場合はそれをrepairするニューロン数として（上位何件）解釈し,
    # floatの場合はそれをrepairするニューロンの割合として（上位何％）解釈する
    repair_ratio = setting_dict["REPAIR_RATIO"]

    # モデルとデータの読み込み先のディレクトリ
    data_dir = f"/src/data/{task_name}/{train_setting_name}"
    model_dir = f"/src/models/{task_name}/{train_setting_name}"

    # test dataloaderをロード (foldに関係ないので先にロードする)
    test_data_path = os.path.join(data_dir, f"test_loader.pt")
    test_loader = torch.load(test_data_path)

    # division(train/repair/test), metrics(acc/fairness) => modelの各foldのmetricsのdiffを保存するdefaultdict
    diff_dict = defaultdict(list)

    for k in range(num_fold):
        logger.info(f"processing fold {k}...")
        # 学習済みモデルをロード
        model = select_model(task_name=task_name)
        model_path = os.path.join(model_dir, f"trained_model_fold-{k}.pt")
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # train setをロード
        train_data_path = os.path.join(data_dir, f"train_loader_fold-{k}.pt")
        train_loader = torch.load(train_data_path)
        train_ds = train_loader.dataset

        # repair setをロード
        repair_data_path = os.path.join(data_dir, f"repair_loader_fold-{k}.pt")
        repair_loader = torch.load(repair_data_path)
        repair_ds = repair_loader.dataset

        # fl_scoreをロード
        # flscoreに関してはpso関係ないので継承するのであれば継承先のものを使う
        flscore_path = (
            f"/src/experiments/repair_results/flscore_{exp_name}_fold{k+1}.npy"
            if not is_inherit
            else f"/src/experiments/repair_results/flscore_{inherit_exp_name}_fold{k+1}.npy"
        )
        flscore = np.load(flscore_path)
        repair_num = repair_ratio if isinstance(repair_ratio, int) else np.ceil(len(flscore) * repair_ratio).astype(int)
        flscore_repaired = flscore[:repair_num]
        repaired_positions = np.array(list(map(eval, flscore_repaired[:, 0])))
        # patchをロード
        # patchはpso_settingが変わると変わるので，常にexp_nameを参照
        patch_path = f"/src/experiments/repair_results/patch_{exp_name}_fold{k+1}.npy"
        patch = np.load(patch_path)

        acc_bef, acc_aft = {}, {}
        fair_bef, fair_aft = {}, {}
        sm_corr_bef, sm_corr_aft = {}, {}
        sm_fair_bef, sm_fair_aft = {}, {}

        # repair前/後のモデルでそれぞれ予測
        for is_repair in [False, True]:
            # ログ表示
            if not is_repair:
                logger.info(f"BEFORE REPAIRED...")
            else:
                logger.info(f"AFTER REPAIRED...")
            (hvals, neuron_location) = (patch, repaired_positions) if is_repair else (None, None)

            # train/repair/test setそれぞれに対する予測
            for div_name, dataloader in zip(["train", "repair", "test"], [train_loader, repair_loader, test_loader]):
                # 予測を実行して合ってるかどうか記録
                ret_dict = eval_model(
                    model=model,
                    dataloader=dataloader,
                    is_repair=is_repair,
                    hvals=hvals,
                    neuron_location=neuron_location,
                )
                # accのみ取り出す
                div_acc = ret_dict["metrics"][0]
                # correctnessのsample metrics
                sm_corr_list = ret_dict["correctness_arr"]
                if not is_repair:
                    acc_bef[div_name] = div_acc
                    sm_corr_bef[div_name] = sm_corr_list
                else:
                    acc_aft[div_name] = div_acc
                    sm_corr_aft[div_name] = sm_corr_list

                # fairnessのsample metrics
                div_fair, sm_fair_list = eval_independence_fairness(
                    model=model,
                    dataloader=dataloader,
                    sens_idx=sens_idx,
                    sens_vals=sens_vals,
                    target_cls=target_cls,
                    is_repair=is_repair,
                    hvals=hvals,
                    neuron_location=neuron_location,
                )
                if not is_repair:
                    fair_bef[div_name] = div_fair
                    sm_fair_bef[div_name] = sm_fair_list
                else:
                    fair_aft[div_name] = div_fair
                    sm_fair_aft[div_name] = sm_fair_list
                # divisionごとのacc, fairを表示
                logger.info(f"{div_name}_acc = {div_acc}, {div_name}_fairness = {div_fair}")

        # repair適用前後の精度と公平性の差分を取る
        # divisionごとに実行
        for div_name in ["train", "repair", "test"]:
            # 精度の差分
            div_acc_diff = acc_aft[div_name] - acc_bef[div_name]
            # 公平性の差分 (値が低い方がいいので前から後を引く)
            div_fair_diff = fair_bef[div_name] - fair_aft[div_name]
            logger.info(f"{div_name}_acc_diff = {div_acc_diff}, {div_name}_fair_diff = {div_fair_diff}")
            # diffの記録
            diff_dict[(div_name, k)].extend([div_acc_diff, div_fair_diff])
            # sample metricsをcsvで保存
            sm_path = os.path.join(sm_dir, f"{div_name}_fold{k+1}.csv")
            sm_dict = {
                "sm_corr_bef": sm_corr_bef[div_name],
                "sm_corr_aft": sm_corr_aft[div_name],
                "sm_fair_bef": sm_fair_bef[div_name],
                "sm_fair_aft": sm_fair_aft[div_name],
            }
            sm_df = make_sm_df(sm_dict)
            sm_df.to_csv(sm_path, index=False)

    # 最終的なdictを表示
    logger.info(f"diff_dict={diff_dict}")

    # diffの辞書をcsvにして保存
    csv_file_name = log_file_name.replace("setting", "result-setting")
    save_path = os.path.join(exp_dir, f"{csv_file_name}.csv")
    diff_df = make_diff_df(diff_dict, num_fold)
    diff_df.to_csv(save_path, index=False)
    logger.info(f"saved to {save_path}")