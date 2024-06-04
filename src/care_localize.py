import os, sys, time, re
from collections import defaultdict

import numpy as np
from lib.model import select_model
from lib.fairness import calc_acc_average_causal_effect, calc_safety_average_causal_effect
from lib.util import json2dict, dataset_type
from lib.log import set_exp_logging
from lib.safety import check_safety_prop
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# plot setting
sns.set_style("white")

# tabluerデータセットにおいてfairness repairをするかどうか
TABULAR_FAIRNESS_SW = False  # FIXME: 最悪なので外部化する
NUM_FOLD_ACAS = 5

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

    # GPUが使えるか確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # 設定用のjsonファイルをdictとしてロード
    # HACK: 共通しているので関数にまとめて自動化したい
    setting_dict = json2dict(sys.argv[1])
    logger.info(f"Settings: {setting_dict}")

    train_setting_path = setting_dict["TRAIN_SETTING_PATH"]
    if not exp_name.startswith("acas"):
        # 訓練時の設定名を取得
        train_setting_name = os.path.splitext(train_setting_path)[0]
        # 訓練時の設定も読み込む
        train_setting_dict = json2dict(os.path.join(exp_dir, train_setting_path))
        logger.info(f"TRAIN Settings: {train_setting_dict}")
        num_fold = train_setting_dict["NUM_FOLD"]
        task_name = train_setting_dict["TASK_NAME"]
        # モデルとデータの読み込み先のディレクトリ
        data_dir = f"/src/data/{task_name}/{train_setting_name}"
        model_dir = f"/src/models/{task_name}/{train_setting_name}"
    else:
        num_fold = NUM_FOLD_ACAS
        task_name = "acasxu"
        # n{i}_{j}_prop{p}の部分を正規表現で取り出し
        pattern = r"acasxu_n(\d+)_(\d+)_prop(\d+)-fairness-setting\d+"
        match = re.search(pattern, exp_name)
        if match:
            nnid = int(match.group(1)), int(match.group(2))
            pid = int(match.group(3))
            acas_setting = f"n{nnid[0]}_{nnid[1]}_prop{pid}"
            logger.info(f"acas_setting: {acas_setting}")
        data_dir = f"/src/data/{task_name}/{acas_setting}"
        model_dir = f"/src/model/{task_name}/{acas_setting}"
        # data_dir, model_dirがなかったら作る
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
    ds_type = dataset_type(task_name)

    # fairnessの計算のための情報をパース
    if (ds_type is "tabular") and TABULAR_FAIRNESS_SW:
        sens_name = setting_dict["SENS_NAME"]
        sens_idx = setting_dict["SENS_IDX"]
        sens_vals = eval(setting_dict["SENS_VALS"])  # ない場合はNone, ある場合は直接listで定義したりrangeで定義できるようにする. そのためにevalを使う
        target_cls = setting_dict["TARGET_CLS"]

    # localizationのための情報をパース
    num_steps = setting_dict["NUM_STEPS"]
    # これらの値は数値かリストで指定可能にするためにevalする
    target_lids = setting_dict["TARGET_LAYER"]
    target_nids = setting_dict["TARGET_NEURON"]
    # lids, nidsのどちらも未指定の場合は後で処理
    if not (target_lids or target_nids):
        target_lids = target_nids = None
    # lids,  nidsどちらも指定されている場合
    elif target_lids and target_nids:
        # 数値の場合は要素数1のリストにする
        if isinstance(target_lids, int):
            target_lids = [target_lids]
        if isinstance(target_nids, int):
            target_nids = [target_nids]
        assert len(target_lids) == len(
            target_nids
        ), f"Error: len(target_lid)({len(target_lids)}) != len(target_nid)({len(target_nids)})"

    time_per_fold = []
    # 各foldのtrain/repairをロードして予測
    for k in range(num_fold):
        logger.info(f"processing fold {k}...")

        # 学習済みモデルをロード
        if not exp_name.startswith("acas"):
            model = select_model(task_name=task_name)
            model_path = os.path.join(model_dir, f"trained_model_fold-{k}.pt")
            model.load_state_dict(torch.load(model_path))
        else:
            # NOTE: acasの場合はkによらずモデルは変化しないがmodel定義の場所がばらけるのが嫌なのでここでやる
            model = select_model(task_name=task_name, nnid=nnid)
            target_lids = range(len(model.layers) - 1) # 最終層ニューロンは出力ニューロンなので含めない
            target_nids = range(model.hidden_size) # 全ニューロンを対象にする
        model.to(device)
        model.eval()

        # foldに対するdataloaderをロード
        # NOTE: repairだけ使うことにしてみる
        repair_data_path = os.path.join(data_dir, f"repair_loader_fold-{k}.pt")
        repair_loader = torch.load(repair_data_path)
        repair_ds = repair_loader.dataset

        # 元のmodelのdataloaderに対するaccuracyを計算
        total_corr = 0  # acc計算用
        # repair_loaderからバッチを読み込み
        for batch_idx, batch in enumerate(repair_loader):
            if ds_type == "safety":
                data = batch[0].to(device=device)
                outputs = model.predict(x=data, device=device)
                is_unsafe = check_safety_prop(outputs, pid)
                num_corr = len(is_unsafe) - sum(is_unsafe) # safety propを満たした数
                total_corr += num_corr.cpu()
            else: # correctnessの観点
                if ds_type == "text":
                    data, labels = batch[0].to(device), batch[1].to(device)
                    data_lens = batch[2]
                    org_preds = model.predict(x=data, x_lens=data_lens, device=device)["pred"]
                else:
                    data, labels = batch[0].to(device), batch[1].to(device)
                    org_preds = model.predict(x=data, device=device)["pred"]
                num_corr = sum(org_preds == labels)
                total_corr += num_corr.cpu()
        acc_org = total_corr / len(repair_ds)
        logger.info(f"acc_org = {acc_org} ({total_corr} / {len(repair_ds)})")

        # FL開始時刻
        s = time.clock()
        fl_score = defaultdict(float)
        # 計算したい各レイヤ/ニューロンに対してFLのスコア（Average Causal Effects）を算出
        # ここからsafetyデータ用====================================================
        if ds_type == "safety":
            # target_layer, neuronごとにFLスコアを計算
            for target_lid, target_nids_for_layer in zip(target_lids, [target_nids for _ in range(len(target_lids))]):
                layer_dist = []
                # バッチごとにあるレイヤの出力を取得して最後に全バッチ結合する
                for batch_idx, batch in enumerate(repair_loader):
                    data = batch[0].to(device=device)
                    layer_dist_batch = model.get_layer_distribution(data, target_lid=target_lid, device=device)
                    layer_dist.append(layer_dist_batch)
                layer_dist = np.concatenate(layer_dist, axis=0)
                logger.info(f"layer_dist.shape={layer_dist.shape}")

                # あるターゲットレイヤの各ターゲットニューロンに対しACEを計算
                for target_nid in target_nids_for_layer:
                    logger.info(f"target layer={target_lid}, target neuron={target_nid}")
                    # ターゲットニューロンの取りうる値の範囲を求めてnum_steps個に等分する
                    hdist = layer_dist[:, target_nid]
                    hmin, hmax = min(hdist), max(hdist)
                    hvals = np.linspace(hmin, hmax, num_steps)
                    logger.info(f"hmin={hmin}, hmax={hmax},\nhvals={hvals}")
                    repair_fairness_list = calc_safety_average_causal_effect(
                        model,
                        repair_loader,
                        target_lid=target_lid,
                        target_nid=target_nid,
                        hvals=hvals,
                        acc_org=acc_org,
                        device=device,
                        ds_type=ds_type,
                        pid=pid
                    )
                    # repair_fairness_listの各要素をcpuに移動
                    repair_fairness_list = [v.cpu() for v in repair_fairness_list]
                    fl_score[f"({target_lid},{target_nid})"] = np.mean(repair_fairness_list)
        # ここまでsafetyデータ用====================================================

        # ここからimageデータ用====================================================
        elif ds_type == "image":
            assert (target_lids is None) and (target_nids is None)
            layer_dist = []
            # バッチごとにあるレイヤの出力を取得して最後に全バッチ結合する
            for batch_idx, batch in enumerate(repair_loader):
                # if ds_type == "text":
                #     data, labels = batch[0].to(device), batch[1].to(device)
                #     data_lens = batch[2]
                # else:
                data, labels = batch[0].to(device), batch[1].to(device)
                layer_dist_batch = model.get_layer_distribution(data, device=device)
                layer_dist.append(layer_dist_batch)
            layer_dist = np.concatenate(layer_dist, axis=0)
            logger.info(f"layer_dist.shape={layer_dist.shape}")
            num_neuron = layer_dist.shape[1]

            # 各ニューロンに対する繰り返し
            # NOTE:全ニューロン対象だと時間がかかるので，2個間隔あけて対象ニューロンを決定
            for target_nid in range(0, num_neuron, 3):  # (FMモデルの場合)0,3,6,...,1023番目がtarget_nid
                hdist = layer_dist[:, target_nid]
                hmin, hmax = min(hdist), max(hdist)
                hvals = np.linspace(hmin, hmax, num_steps)
                logger.info(f"nid={target_nid}, hmin={hmin}, hmax={hmax},\nhvals={hvals}")
                repair_accdiff_list = calc_acc_average_causal_effect(
                    model,
                    repair_loader,
                    target_lid=None,
                    target_nid=target_nid,
                    hvals=hvals,
                    acc_org=acc_org,
                    device=device,
                    ds_type=ds_type,
                )
                fl_score[f"(fixed, {target_nid})"] = np.mean(repair_accdiff_list)
        # ここまでimageデータ用====================================================
        elif ds_type == "tabular":
            # (新しい方) ここからtabularデータ用====================================================
            # 古い方からの変更点
            # - Imageの方と同じようにGPU高速化やバッチ処理によるメモリ節約.
            # - calc_average_causal_effectでなくcalc_acc_average_causal_effectを使うようにした (つまりfairness repairではなくacc repair).
            # target_lids, 各ターゲットレイヤに対するtarget_nidsも必ず指定されているという前提
            for target_lid, target_nids_for_layer in zip(target_lids, target_nids):
                layer_dist = []
                # バッチごとにあるレイヤの出力を取得して最後に全バッチ結合する
                for batch_idx, (data, labels) in enumerate(repair_loader):
                    data, labels = data.to(device), labels.to(device)
                    layer_dist_batch = model.get_layer_distribution(data, target_lid=target_lid)
                    layer_dist.append(layer_dist_batch)
                layer_dist = np.concatenate(layer_dist, axis=0)
                logger.info(f"layer_dist.shape={layer_dist.shape}")

                # あるターゲットレイヤの各ターゲットニューロンに対しACEを計算
                for target_nid in target_nids_for_layer:
                    logger.info(f"target layer={target_lid}, target neuron={target_nid}")

                    hdist = layer_dist[:, target_nid]
                    hmin, hmax = min(hdist), max(hdist)
                    hvals = np.linspace(hmin, hmax, num_steps)
                    logger.info(f"hmin={hmin}, hmax={hmax},\nhvals={hvals}")

                    repair_fairness_list = calc_acc_average_causal_effect(
                        model,
                        repair_loader,
                        target_lid=target_lid,
                        target_nid=target_nid,
                        hvals=hvals,
                        acc_org=acc_org,
                        device=device,
                        ds_type=ds_type,
                    )
                    fl_score[f"({target_lid},{target_nid})"] = np.mean(repair_fairness_list)
        elif ds_type == "text":
            assert (target_lids is None) and (target_nids is None)
            # ここからtext dataset用
            layer_dist = []
            # バッチごとにあるレイヤの出力を取得して最後に全バッチ結合する
            for batch_idx, (data, labels, data_lens) in enumerate(repair_loader):
                data, labels = data.to(device), labels.to(device)
                # NOTE: target_layerは最終のFC層で固定
                layer_dist_batch, _ = model.get_layer_distribution(data, data_lens)
                layer_dist.append(layer_dist_batch)
            layer_dist = np.concatenate(layer_dist, axis=0)
            logger.info(f"layer_dist.shape={layer_dist.shape}")
            
            for target_nid in range(layer_dist.shape[1]):
                logger.info(f"fixed target layers, target neuron={target_nid}")
                hdist = layer_dist[:, target_nid]
                hmin, hmax = min(hdist), max(hdist)
                hvals = np.linspace(hmin, hmax, num_steps)
                logger.info(f"hmin={hmin}, hmax={hmax},\nhvals={hvals}")
                repair_fairness_list = calc_acc_average_causal_effect(
                    model,
                    repair_loader,
                    target_lid=None,
                    target_nid=target_nid,
                    hvals=hvals,
                    acc_org=acc_org,
                    device=device,
                    ds_type=ds_type,
                )
                fl_score[f"(fixed, {target_nid})"] = np.mean(repair_fairness_list)
        else:
            raise NotImplementedError("the fairness repair for the tabular dataset is under construction.")
        
            # # (古い方) ここからtabularデータ用====================================================
            #     for target_lid, target_nids_for_layer in zip(target_lids, target_nids):
            #         # 対象のレイヤに対する各ニューロンの順伝搬の値を計算
            #         layer_dist = model.get_layer_distribution(repair_ds, target_lid=target_lid)

            #         for target_nid in target_nids_for_layer:
            #             logger.info(f"target layer={target_lid}, target neuron={target_nid}")

            #             hdist = layer_dist[:, target_nid]
            #             hmin, hmax = min(hdist), max(hdist)
            #             hvals = np.linspace(hmin, hmax, num_steps)
            #             logger.info(f"hmin={hmin}, hmax={hmax},\nhvals={hvals}")

            #             repair_fairness_list = calc_average_causal_effect(
            #                 model, repair_loader, sens_idx, target_lid, target_nid, hvals, sens_vals, target_cls
            #             )
            #             fl_score[f"({target_lid},{target_nid})"] = np.mean(repair_fairness_list)
            #     # ニューロンごとのfl_scoreのプロットを保存
            #     plot_fl_score(fl_score, exp_name, k)
            # # ここまでtabularデータ用====================================================

        # FL修了時刻 (あくまでfoldごとなので実際はこの時間 * fold数かかる)
        e = time.clock()
        # logger.info(f"End time: {e}")
        logger.info(f"Total execution time for FL: {e-s} sec.")
        time_per_fold.append(e-s)
        # fl_scoreを降順に並び替えて保存する
        fl_score_sorted = np.array(sorted(fl_score.items(), key=lambda v: v[1], reverse=True))
        logger.info(f"SORTED FL SCORE: \n{fl_score_sorted}")
        fl_score_save_path = os.path.join(exp_dir, f"repair_results/flscore_{exp_name}_fold{k+1}.npy")
        np.save(fl_score_save_path, fl_score_sorted)
        logger.info(f"saved to {fl_score_save_path}")
    # 全foldのFL時間の平均をログに残す
    logger.info(f"Average execution time for FL: {np.mean(time_per_fold)} sec.")