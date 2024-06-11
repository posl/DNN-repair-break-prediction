
import os, sys, argparse, re
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
import pandas as pd
import numpy as np
from lib.fairness import eval_independence_fairness
from lib.model import select_model, eval_model
from lib.util import json2dict, dataset_type
from lib.log import set_exp_logging
import torch
import matplotlib.pyplot as plt
import seaborn as sns

DS_LIST = ["census", "credit", "bank"]

if __name__ == "__main__":
    # gpuが使用可能かをdeviceにセット
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DS_LIST, help=f"dataset name should be in {DS_LIST}")
    args = parser.parse_args()
    ds = args.dataset
    method = "care"
    # log setting
    care_dir = "/src/experiments/care/"
    this_file_name = os.path.basename(__file__).replace(".py", "").replace("_", "-")
    log_file_name = f"{ds}-{this_file_name}"
    logger = set_exp_logging(care_dir, log_file_name, log_file_name)
    logger.info(f"target dataset: {ds}, method={method}")

    # set sensitive features name
    if ds == "census":
        sensitive_feat_name = "gender"
    elif ds == "credit":
        sensitive_feat_name = "gender"
    elif ds == "bank":
        sensitive_feat_name = "age"
    # path to json files for sensitive related information
    json_file_name = f"{ds}-fairness-{sensitive_feat_name}-setting1.json"
    json_path = os.path.join(care_dir, json_file_name)
    setting_dict = json2dict(json_path)
    logger.info(f"Settings: {setting_dict}")

    # 訓練時の設定名を取得
    train_setting_path = setting_dict["TRAIN_SETTING_PATH"]
    train_setting_name = os.path.splitext(train_setting_path)[0]
    # fairnessの計算のための情報をパース
    sens_idx = setting_dict["SENS_IDX"]
    # sens_vals = eval(setting_dict["SENS_VALS"])  # ない場合はNone, ある場合は直接listで定義したりrangeで定義できるようにする. しのためにevalを使う
    # NOTE: sens_valsを使わず実際のデータから値の候補を取得することにした (カテゴリカルデータも標準化されてしまってるので)
    target_cls = setting_dict["TARGET_CLS"]
    repair_ratio = setting_dict["REPAIR_RATIO"]

    # 訓練時の設定も読み込む
    train_setting_dict = json2dict(os.path.join(care_dir, train_setting_path))
    logger.info(f"TRAIN Settings: {train_setting_dict}")
    num_fold = train_setting_dict["NUM_FOLD"]
    task_name = train_setting_dict["TASK_NAME"]

    # モデルとデータの読み込み先のディレクトリ
    data_dir = f"/src/data/{task_name}/{train_setting_name}"
    model_dir = f"/src/models/{task_name}/{train_setting_name}"

    # test dataloaderをロード (foldに関係ないので先にロードする)
    test_data_path = os.path.join(data_dir, f"test_loader.pt")
    test_loader = torch.load(test_data_path)

    # 正解不正解の配列を保存するためのdir
    fairness_dir = os.path.join("/src/src/fairness/fairness_before", ds)
    # repair break datasetsのfoldごとのcsvを保存するdir
    rb_apricot_save_dir = os.path.join("/src/src/fairness/rb_datasets", ds, method)
    os.makedirs(rb_apricot_save_dir, exist_ok=True)

    # 各foldのtrain/repairをロードして予測
    for k in range(num_fold):
        logger.info(f"processing fold {k}...")
        # train/repair/testの各foldのfairnessを保存するdefaultdict
        fairness_dict = defaultdict(list)

        # 学習済みモデルをロード
        model = select_model(task_name=task_name)
        model_path = os.path.join(model_dir, f"trained_model_fold-{k}.pt")
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # foldに対するdataloaderをロード
        train_data_path = os.path.join(data_dir, f"train_loader_fold-{k}.pt")
        repair_data_path = os.path.join(data_dir, f"repair_loader_fold-{k}.pt")
        train_loader = torch.load(train_data_path)
        repair_loader = torch.load(repair_data_path)

        # sensitive featuresが一列だけ対象の場合
        if isinstance(sens_idx, int):
            tgt_vals = train_loader.dataset.tensors[0][:, sens_idx]
            # tgt_valsをnumpyにしてunique valuesを表示
            sens_vals = np.unique(tgt_vals.numpy())
            logger.info(f"list of sens_vals: {sens_vals} (len={len(sens_vals)})")
            # sens_valsをtorch tensorにしておく
            sens_vals = torch.tensor(sens_vals)
        else:
            sens_vals = None

        # fl_scoreをロード
        flscore_path = f"/src/experiments/care/repair_results/flscore_{ds}-fairness-setting1_fold{k+1}.npy"
        fl_scores = np.load(flscore_path)
        repair_num = (
            repair_ratio if isinstance(repair_ratio, int) else np.ceil(len(fl_scores) * repair_ratio).astype(int)
        )
        # 修正対象のニューロンの位置(何層目の,何番目のニューロンか)
        repaired_positions = fl_scores[:repair_num][:, 0]
        # target_lidが固定かどうかチェック
        if "fixed" in repaired_positions[0]:
            # target_neuronを示す1次元の配列にする
            repaired_positions = np.array([int(re.search(r"\d+", entry).group(0)) for entry in repaired_positions])
        # target_lidの情報も必要な場合
        else:
            repaired_positions = np.array(list(map(eval, repaired_positions)))
        logger.info(f"repaired_positions: {repaired_positions}")

        # 各dataloaderに対するfairness
        for div_name, div_dl in zip(["train", "repair", "test"], [train_loader, repair_loader, test_loader]):
            logger.info(f"processing {div_name} set...")
            # fold k のモデルのkeyに対するexp metricsをロード
            exp_metrics_path = os.path.join(care_dir, "explanatory_metrics", f"{ds}-training-setting1", f"{div_name}_fold{k+1}.csv")
            df_expmet = pd.read_csv(exp_metrics_path)
            # 修正前にあってたかどうかの配列
            is_corr_bef = np.load(os.path.join(fairness_dir, f"{div_name}_fold{k+1}.npy"))
            is_corr_bef = 1 - is_corr_bef # NOTE: 高い方が良い, にするため
            # 修正後の予測結果を格納するための配列
            is_corr_aft = np.zeros((len(is_corr_bef), 5))
            # 修正のnum_reps回の適用結果に関してそれぞれ見ていく
            for rep in range(5):
                logger.info(f"processing rep {rep}...")

                # patchをロード
                # patchはpso_settingが変わると変わるので，常にexp_nameを参照
                patch_path = os.path.join(model_dir, "care-result", f"rep{rep}", f"patch_{ds}-fairness-setting1_fold{k}.npy")
                patch = np.load(patch_path)
                hvals, neuron_location = patch, repaired_positions

                # fairnessの結果を取得
                fairness_dict[div_name] = eval_independence_fairness(
                model=model, dataloader=div_dl, sens_idx=sens_idx, sens_vals=sens_vals, target_cls=target_cls, is_repair=True, hvals=hvals, neuron_location=repaired_positions
                )[1]
                is_corr_aft[:, rep] = 1 - fairness_dict[div_name] # NOTE: 高い方が良い, にするため
            
            # repair/breakの対象を決定するための閾値を設定
            sm_fair_th = 0.95 # 1-fairnessがこの閾値以上ならOK，より小さいならNG
            # is_corr_aftの各行において，閾値以上の値の数を取得
            is_corr_aft_sum = np.sum(is_corr_aft >= sm_fair_th, axis=1)
            # 修正前のfairnessと, 修正後のfairness (5回の試行の平均) を2値化したものをまとめたDataFrameを作成
            df = pd.DataFrame({"sm_fair_bef": (is_corr_bef >= sm_fair_th).astype(int), "sm_fair_aft_sum": is_corr_aft_sum})

            # repaired, brokenの真偽を決定
            df["repaired"] = (df["sm_fair_bef"] == 0) & (df["sm_fair_aft_sum"] == 5)  # 厳し目の決定方法
            df["repaired"] = (df["sm_fair_bef"] == 0) & (df["sm_fair_aft_sum"] >= 1)  # ゆる目の決定方法
            df["broken"] = (df["sm_fair_bef"] == 1) & (df["sm_fair_aft_sum"] != 5)  # 厳し目の決定方法
            logger.info(f"df_sm.shape: {df.shape}")

            # exp. metricsと，repaied, brokenなどの列と結合する
            df_all = pd.concat([df_expmet, df], axis=1)
            logger.info(f"df_all.shape: {df_all.shape}")

            # repair, breakのデータセットを作成
            df_repair, df_break = df_all[df_all["sm_fair_bef"] == 0], df_all[df_all["sm_fair_bef"] == 1]
            # それぞれのdfからいらない列を削除
            df_repair = df_repair.drop(["sm_fair_bef", "sm_fair_aft_sum", "broken"], axis=1)
            df_break = df_break.drop(["sm_fair_bef", "sm_fair_aft_sum", "repaired"], axis=1)
            logger.info(f"df_repair.shape: {df_repair.shape}, df_break.shape: {df_break.shape}")
            # df_repair, df_breakをそれぞれ保存
            df_repair.to_csv(os.path.join(rb_apricot_save_dir, f"{div_name}_fold{k+1}-repair.csv"), index=False)
            df_break.to_csv(os.path.join(rb_apricot_save_dir, f"{div_name}_fold{k+1}-break.csv"), index=False)
            logger.info(f'saved to {os.path.join(rb_apricot_save_dir, f"{div_name}_fold{k+1}-[repair|brek].csv")}')
            # df_repair, df_breakの情報を表示
            logger.info(f"#repaired is True: {len(df_repair[df_repair['repaired']==True])} / {len(df_repair)}")
            logger.info(f"#broken is True: {len(df_break[df_break['broken']==True])} / {len(df_break)}")
