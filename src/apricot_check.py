import os, sys, time
import warnings

warnings.filterwarnings("ignore")

from lib.log import set_exp_logging
from lib.util import json2dict
from lib.model import train_model, eval_model, select_model
from src_apricot.apricot_lib import setWeights

import torch
import numpy as np
import pandas as pd

# Hyper-parameters
num_restrict = 1000  # rDLMを作るための縮小データセットを作る際に，各ラベルからどれだけサンプルするか
rDLM_num = 20
batch_size = 64
num_reps = 5


if __name__ == "__main__":
    """
    {dataset名}-training-setting{sid}を入力として必要な情報の準備
    """

    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(sys.argv[1])
    apricot_dir = exp_dir.replace("care", "apricot")
    exp_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]

    # log setting
    # {dataset}-repair-fairness-{feature}-setting{NO}.logというファイルにログを出す
    log_file_name = exp_name.replace("training", "check-apricot")
    logger = set_exp_logging(exp_dir.replace("care", "apricot"), exp_name, log_file_name)

    # 設定用のjsonファイルをdictとしてロード
    # HACK: 共通しているので関数にまとめて自動化したい
    setting_dict = json2dict(sys.argv[1])
    logger.info(f"Settings: {setting_dict}")
    task_name = setting_dict["TASK_NAME"]
    sens_name = "age" if task_name == "bank" else "gender"
    train_repair_data_path = setting_dict["TRAIN-REPAIR_DATA_PATH"]
    test_data_path = setting_dict["TEST_DATA_PATH"]
    target_column = setting_dict["TARGET_COLUMN"]
    num_epochs = setting_dict["NUM_EPOCHS"]
    batch_size = setting_dict["BATCH_SIZE"]
    num_fold = setting_dict["NUM_FOLD"]

    # モデルとデータの読み込み先のディレクトリ
    data_dir = f"/src/data/{task_name}/{exp_name}"
    model_dir = f"/src/models/{task_name}/{exp_name}"

    sm_save_dir = os.path.join(apricot_dir, "sample_metrics", exp_name)
    os.makedirs(sm_save_dir, exist_ok=True)
    rb_ds_save_dir = os.path.join(apricot_dir, "repair_break_dataset", "raw_data")
    os.makedirs(rb_ds_save_dir, exist_ok=True)

    # test loaderのロード
    test_data_path = os.path.join(data_dir, f"test_loader.pt")
    test_loader = torch.load(test_data_path)

    df_repair_list, df_break_list = [], []
    # iDLMやdataloaderのロード
    for k in range(num_fold):
        # foldに対するdataloaderをロード
        train_data_path = os.path.join(data_dir, f"train_loader_fold-{k}.pt")
        train_loader = torch.load(train_data_path)
        repair_data_path = os.path.join(data_dir, f"repair_loader_fold-{k}.pt")
        repair_loader = torch.load(repair_data_path)

        for div, dataloader in zip(["train", "repair", "test"], [train_loader, repair_loader, test_loader]):
            logger.info(f"processing fold {k} {div} set...")
            # 修正前に正解だったかどうかの情報をcareのsample metricsのファイルから得る
            care_dir = exp_dir.replace("apricot", "care")
            sample_metrics_path = os.path.join(
                care_dir,
                "sample_metrics",
                exp_name.replace("training", f"repair-check-{sens_name}"),
                f"{div}_fold{k+1}.csv",
            )
            df = pd.read_csv(sample_metrics_path)
            is_corr_bef = df["sm_corr_bef"].values

            # 修正後の予測結果を格納するための配列
            is_corr_aft = np.zeros((len(is_corr_bef), num_reps))

            # 修正のnum_reps回の適用結果に関してそれぞれ見ていく
            for rep in range(num_reps):
                weight_save_dir = os.path.join(model_dir, "apricot-weight", f"rep{rep}")
                weight_save_path = os.path.join(weight_save_dir, f"adjusted_weights_fold-{k}.pt")
                # モデルの初期化
                model = select_model(task_name=task_name)
                # apricot適用後の重みをロード
                model.load_state_dict(torch.load(weight_save_path))
                model.eval()

                # 予測を実行して結果を取得
                ret_dict = eval_model(
                    model=model,
                    dataloader=dataloader,
                )
                is_corr_aft_rep = ret_dict["correctness_arr"]
                is_corr_aft[:, rep] = is_corr_aft_rep
            is_corr_aft_sum = np.sum(is_corr_aft, axis=1, dtype=np.int32)

            # 修正前にあってたかどうかと，5回の修正それぞれの後で正しく予測できた回数の合計をまとめたDataFrameを作成
            df = pd.DataFrame({"sm_corr_bef": is_corr_bef, "sm_corr_aft_sum": is_corr_aft_sum})
            # repaired, brokenの真偽を決定
            df["repaired"] = (df["sm_corr_bef"] == 0) & (df["sm_corr_aft_sum"] == 5)
            df["broken"] = (df["sm_corr_bef"] == 1) & (df["sm_corr_aft_sum"] != 5)
            logger.info(f"df_sm.shape: {df.shape}")
            df.to_csv(os.path.join(sm_save_dir, f"{div}_fold{k+1}.csv"), index=False)
            logger.info(f'saved to {os.path.join(sm_save_dir, f"{div}_fold{k+1}.csv")}')

            # exp. metricsもロードしてきて，repaied, brokenなどの列と結合する
            exp_metrics_path = os.path.join(care_dir, "explanatory_metrics", exp_name, f"{div}_fold{k+1}.csv")
            df_expmet = pd.read_csv(exp_metrics_path)
            df_all = pd.concat([df_expmet, df], axis=1)
            logger.info(f"df_all.shape: {df_all.shape}")

            # repair, breakのデータセットを作成
            df_repair, df_break = df_all[df_all["sm_corr_bef"] == 0], df_all[df_all["sm_corr_bef"] == 1]
            # それぞれのdfからいらない列を削除
            df_repair = df_repair.drop(["sm_corr_bef", "sm_corr_aft_sum", "broken"], axis=1)
            df_break = df_break.drop(["sm_corr_bef", "sm_corr_aft_sum", "repaired"], axis=1)
            logger.info(f"df_repair.shape: {df_repair.shape}, df_break.shape: {df_break.shape}")
            df_repair_list.append(df_repair)
            df_break_list.append(df_break)

    logger.info("Concatenating all folds and divisions...")
    # 全fold, divにおけるdf_repair, df_breakを結合して全体のrepair dataset, break datasetを作る
    df_repair = pd.concat(df_repair_list, axis=0)
    df_break = pd.concat(df_break_list, axis=0)
    logger.info(f"df_repair.shape: {df_repair.shape}, df_break.shape: {df_break.shape}")
    logger.info(f"#repaired is True: {len(df_repair[df_repair['repaired']==True])} / {len(df_repair)}")
    logger.info(f"#broken is True: {len(df_break[df_break['broken']==True])} / {len(df_break)}")
    # それぞれcsvとして保存
    df_repair.to_csv(os.path.join(rb_ds_save_dir, f"{exp_name}-repair.csv"), index=False)
    df_break.to_csv(os.path.join(rb_ds_save_dir, f"{exp_name}-break.csv"), index=False)
