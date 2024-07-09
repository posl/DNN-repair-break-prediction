"""
ran the original and repaired model (obtained by replication script in another repository) for each dataloader (five repair loaders and one test loader) and save the repairs and breaks datasets.
"""
import os, sys, re, math
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from lib.util import json2dict, dataset_type, fix_dataloader
from lib.log import set_exp_logging
from lib.model import eval_model, ACASXuModel
import torch
import pandas as pd
import numpy as np
from collections import defaultdict

NUM_FOLD_ACAS = 5
APRNN_SEED_LIST = [0, 1, 42, 777, 2024]

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
    # columns = ["sm_corr_bef", "sm_corr_aft", "sm_fair_bef", "sm_fair_aft"]
    columns = ["sm_corr_bef", "sm_corr_aft"]
    df = pd.DataFrame(columns=columns)

    # dfに行を追加していく
    for col in columns:
        df[col] = sm_dict[col]
    logger.info(f"df.columns={df.columns}. df.shape={df.shape}")
    return df

if __name__ == "__main__":
    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(sys.argv[1]).replace("care", "aprnn")
    care_dir = os.path.dirname(sys.argv[1])
    exp_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    # log setting
    # {dataset}-repair-check-{feature}-setting{NO}.logというファイルにログを出す
    log_file_name = exp_name.replace("fairness", "aprnn-check")
    logger = set_exp_logging(exp_dir, exp_name, log_file_name)
    # GPUが使えるか確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    # sample metrics保存用のディレクトリを作成
    sm_dir = os.path.join(exp_dir, "sample_metrics", log_file_name)
    os.makedirs(sm_dir, exist_ok=True)
    # n{i}_{j}_prop{p}の部分を正規表現で取り出し
    pattern = r"acasxu_n(\d+)_(\d+)_prop(\d+)-fairness-setting\d+"
    match = re.search(pattern, exp_name)
    if match:
        nnid = int(match.group(1)), int(match.group(2))
        pid = int(match.group(3))
        acas_setting = f"n{nnid[0]}_{nnid[1]}_prop{pid}"
        logger.info(f"acas_setting: {acas_setting}")
    else:
        raise ValueError(f"exp_name {exp_name} is invalid.")
    # 定数の設定
    num_fold = NUM_FOLD_ACAS
    task_name = "acasxu"
    ds_type = dataset_type(task_name)
    data_dir = f"/src/data/{task_name}/{acas_setting}"
    model_dir = f"/src/models/{task_name}/{acas_setting}"
    perspective = "safety"
    collate_fn = None

    # 学習済みオリジナルモデルのロード
    ori_model = ACASXuModel(nnid)
    ori_model.to(device)
    ori_model.eval()

    # 5つのrepaired modelをロードしてリストで保持
    repaired_models = []
    for seed in APRNN_SEED_LIST:
        model_path = f"/src/models/acasxu/n{nnid[0]}_{nnid[1]}_prop{pid}/aprnn/aprnn_n{nnid[0]}{nnid[1]}_seed{seed}.pth"
        logger.info(f"loading repaired model from {model_path}")
        repaired_models.append(ACASXuModel(nnid, "aprnn", model_path))

    # test dataloaderをロード (foldに関係ないので先にロードする)
    test_data_path = os.path.join(data_dir, f"test_loader.pt")
    test_loader = torch.load(test_data_path)

    df_repair_list, df_break_list = [], []
    # kはrepair setの分割方法のバリエーションを示す
    for k in range(num_fold):
        logger.info(f"processing fold {k}...")

        # repair setをロード
        repair_data_path = os.path.join(data_dir, f"repair_loader_fold-{k}.pt")
        repair_loader = fix_dataloader(torch.load(repair_data_path), collate_fn=collate_fn)
        div_names = ["repair", 'test']
        div_dls = [repair_loader, test_loader]

        sm_corr_bef, sm_corr_aft = defaultdict(list), defaultdict(list)
        # original modelの予測を得る
        for div_name, dataloader in zip(div_names, div_dls):
            # 修正前モデル
            ret_dict_bef = eval_model(model=ori_model, dataloader=dataloader, dataset_type=ds_type, device=device, perspective=perspective, pid=pid)
            sm_corr_bef[div_name] = ret_dict_bef["correctness_arr"]
        # repaired modelの予測を得る
        for div_name, dataloader in zip(div_names, div_dls):
            # aprnnの各修正パッチを適用していくloop
            for repaired_model in repaired_models:
                repaired_model.to(device)
                repaired_model.eval()
                # 修正後モデル
                ret_dict_aft = eval_model(model=repaired_model, dataloader=dataloader, dataset_type=ds_type, device=device, perspective=perspective, pid=pid)
                sm_corr_aft[div_name].append(ret_dict_aft["correctness_arr"])
        for div_name in div_names:
                sm_path = os.path.join(sm_dir, f"{div_name}_fold{k+1}.csv")
                sm_dict = {
                    "sm_corr_bef": sm_corr_bef[div_name],
                    "sm_corr_aft": list(np.sum(sm_corr_aft[div_name], axis=0)),
                }
                sm_df = make_sm_df(sm_dict)
                sm_df.to_csv(sm_path, index=False)
    
        # rb datasets保存用
        rb_ds_save_dir = os.path.join(exp_dir, "repair_break_dataset", "raw_data")
        os.makedirs(rb_ds_save_dir, exist_ok=True)

        #####################################
        # repair, break datasetsを作るループ#
        #####################################
        for div in div_names:
            csv_file_name = "test.csv" if div == "test" else f"{div}_fold{k+1}.csv"
            exp_metrics_path = os.path.join(care_dir, "explanatory_metrics", f"acasxu_{acas_setting}", csv_file_name)
            logger.info(f"processing fold {k} {div} set...")

            # 修正前に正解だったかどうかの情報をcareのsample metricsのファイルから得る
            sm_path = os.path.join(sm_dir, f"{div}_fold{k+1}.csv")
            df = pd.read_csv(sm_path)
            # 修正前の予測結果(0が不正解, 1が正解)
            is_corr_bef = df["sm_corr_bef"].values
            # 修正後の予測成功回数を格納するための配列
            is_corr_aft = df["sm_corr_aft"].values

            # 修正前にあってたかどうかと，5回の修正それぞれの後で正しく予測できた回数の合計をまとめたDataFrameを作成
            df = pd.DataFrame({"sm_corr_bef": is_corr_bef, "sm_corr_aft_sum": is_corr_aft})
            # print(df[df["sm_corr_bef"] == 0]["sm_corr_aft_sum"].value_counts())
            # print(df[df["sm_corr_bef"] == 1]["sm_corr_aft_sum"].value_counts())
            # repaired, brokenの真偽を決定
            # df["repaired"] = (df["sm_corr_bef"] == 0) & (df["sm_corr_aft_sum"] == 5)  # 厳し目の決定方法
            df["repaired"] = (df["sm_corr_bef"] == 0) & (df["sm_corr_aft_sum"] >= 1)  # ゆる目の決定方法
            df["broken"] = (df["sm_corr_bef"] == 1) & (df["sm_corr_aft_sum"] != 5)  # 厳し目の決定方法

            # exp. metricsもロードしてきて，repaied, brokenなどの列と結合する
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
