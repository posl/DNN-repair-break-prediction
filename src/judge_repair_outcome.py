import os, sys
from collections import defaultdict
import pandas as pd
import numpy as np
from lib.util import json2dict
from lib.log import set_exp_logging
import matplotlib.pyplot as plt
import seaborn as sns

# plot setting
sns.set()

"""
sm_befとsm_aftの列を入力として, (non-)repaired/brokenという4種類の判定の列に変える関数の実装.
定義ごとに3種類実装する.
"""


def th_based_judge(sm_bef, sm_aft, th=0.5):
    """閾値ベースの判断. sm_{}<th -> 0, sm_{}>=th -> 1 に射影して, correctnessと同様にrepaired/brokenを決めるイメージ.

    Args:
        sm_bef (1d array-like): 修正前の各サンプルに対するサンプルメトリクス.
        sm_aft (1d array-like): 修正前の各サンプルに対するサンプルメトリクス.
        th (float, optional): smの良し悪しの判断の閾値. Defaults to 0.5.

    Returns:
        list of str: 各サンプルに対する, repaired, non-repaired, broken, non-brokenの判断結果.
    """

    res = np.where(
        (sm_bef < th) & (sm_aft >= th),
        "repaired",
        np.where(
            (sm_bef < th) & (sm_aft < th),
            "non-repaired",
            np.where((sm_bef >= th) & (sm_aft >= th), "non-broken", "broken"),
        ),
    )
    return res


def diff_based_judge(sm_bef, sm_aft):
    """前後変化ベースの判断. sm_aft>sm_befならrepaired, そうでなければbrokenと判断する.

    Args:
        sm_bef (1d array-like): 修正前の各サンプルに対するサンプルメトリクス.
        sm_aft (1d array-like): 修正前の各サンプルに対するサンプルメトリクス.

    Returns:
        list of str: 各サンプルに対する, repaired, brokenの判断結果.
    """

    res = np.where(sm_aft > sm_bef, "repaired", "broken")
    return res


def hybrid_judge(sm_bef, sm_aft, th=0.5):
    """閾値と前後変化のハイブリッドの判断. sm_bef<th -> 0, sm_bef>=th -> 1 に射影してから, 前後変化の判断によってrepaired/brokenを決めるイメージ.

    Args:
        sm_bef (1d array-like): 修正前の各サンプルに対するサンプルメトリクス.
        sm_aft (1d array-like): 修正前の各サンプルに対するサンプルメトリクス.
        th (float, optional): smの良し悪しの判断の閾値. Defaults to 0.5.

    Returns:
        list of str: 各サンプルに対する, repaired, non-repaired, broken, non-brokenの判断結果.
    """

    res = np.where(
        (sm_bef < th) & (sm_aft > sm_bef),
        "repaired",
        np.where(
            (sm_bef < th) & (sm_aft <= sm_bef),
            "non-repaired",
            np.where((sm_bef >= th) & (sm_aft > sm_bef), "non-broken", "broken"),
        ),
    )
    return res


if __name__ == "__main__":
    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(sys.argv[1])
    exp_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    # log setting
    # prepare_dataset_modelとは別のファイルにログを出すようにする
    # HACK: exp_nameにtrainingが含まれてないといけない
    log_file_name = exp_name.replace("fairness", "judge-repair-outcome")
    logger = set_exp_logging(exp_dir, exp_name, log_file_name)

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

    # 訓練時の設定名を取得
    train_setting_path = setting_dict["TRAIN_SETTING_PATH"]
    train_setting_name = os.path.splitext(train_setting_path)[0]

    # 訓練時の設定も読み込む
    train_setting_dict = json2dict(os.path.join(exp_dir, train_setting_path))
    logger.info(f"TRAIN Settings: {train_setting_dict}")
    num_fold = train_setting_dict["NUM_FOLD"]
    task_name = train_setting_dict["TASK_NAME"]

    # このプログラムで作成するdfの保存用のディレクトリを作成
    judge_dir = os.path.join(exp_dir, "judge_repair_outcome", exp_name)
    os.makedirs(judge_dir, exist_ok=True)

    ################################################
    # sample metricsのcsvをロードしてrepaired,     #
    # non-repaired, broken, non-brokenに振り分ける #
    ################################################
    logger.info(f"judging repaired, non-repaired, broken, non-broken...")
    df_sm_dict = defaultdict(defaultdict)
    # 各foldのループ
    for k in range(1, 1 + num_fold, 1):
        logger.info(f"processing fold {k}...")
        df_sm_dict[k] = defaultdict(pd.DataFrame)

        # data divisionごとのループ
        for div_name in ["train", "repair", "test"]:
            sm_dir_name = os.path.join("sample_metrics", exp_name.replace("fairness", "repair-check"))
            sm_path = os.path.join(
                exp_dir,
                sm_dir_name,
                f"{div_name}_fold{k}.csv",
            )
            # サンプルごとのsample metricsの記録されたdfをロードする
            df = pd.read_csv(sm_path)
            # fairnessの列は1-(該当列)して大きい方が良いようにする
            df["sm_fair_bef"] = 1 - df["sm_fair_bef"]
            df["sm_fair_aft"] = 1 - df["sm_fair_aft"]
            df_sm_dict[k][div_name] = df

    # 全体のsm_befの平均をsmの良し悪しの閾値とする
    df_sm_list = []
    for k in range(1, 1 + num_fold, 1):
        for div_name in ["train", "repair", "test"]:
            df_sm_list.append(df_sm_dict[k][div_name])
    df_sm_setting = pd.concat(df_sm_list, axis=0)
    fair_th = df_sm_setting["sm_fair_bef"].mean()

    # 辞書の方に判断結果の列を追加
    for k in range(1, 1 + num_fold, 1):
        for div_name in ["train", "repair", "test"]:
            # fairnessの方のjudgement
            df_sm_dict[k][div_name]["judge_fairness"] = hybrid_judge(
                df_sm_dict[k][div_name]["sm_fair_bef"], df_sm_dict[k][div_name]["sm_fair_aft"], fair_th
            )
            # correctnessの方のjudgement
            df_sm_dict[k][div_name]["judge_correctness"] = th_based_judge(
                df_sm_dict[k][div_name]["sm_corr_bef"], df_sm_dict[k][div_name]["sm_corr_aft"], th=0.5
            )

    ########################################
    # explanatory metricsのcsvをロードして #
    # 前ブロックのjudgeの列と結合する      #
    ########################################
    logger.info(f"combining judgement results and explanatory metrics...")
    expmet_dir_name = os.path.join("explanatory_metrics", train_setting_name)
    df_expmet_dict = defaultdict(defaultdict)
    # 各foldのループ
    for k in range(1, 1 + num_fold, 1):
        logger.info(f"processing fold {k}...")
        df_expmet_dict[k] = defaultdict(pd.DataFrame)

        # data divisionごとのループ
        for div_name in ["train", "repair", "test"]:
            expmet_path = os.path.join(
                exp_dir,
                expmet_dir_name,
                f"{div_name}_fold{k}.csv",
            )
            df = pd.read_csv(expmet_path)
            # expmetと判断結果の列を追加
            df["judge_fairness"] = df_sm_dict[k][div_name]["judge_fairness"]
            df["judge_correctness"] = df_sm_dict[k][div_name]["judge_correctness"]
            df_expmet_dict[k][div_name] = df
            logger.info(f'\n{df["judge_fairness"].value_counts()}')
            logger.info(f'\n{df["judge_correctness"].value_counts()}')

            # 指定したディレクトリに保存
            save_path = os.path.join(judge_dir, f"{div_name}_fold{k}.csv")
            df_expmet_dict[k][div_name].to_csv(save_path, index=False)
            logger.info(f"saved to {save_path}")
