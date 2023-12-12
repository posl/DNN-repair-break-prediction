import os, sys, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pickle
from collections.abc import Iterable
from collections import defaultdict
import pandas as pd
import numpy as np
from lib.model import select_model, eval_model
from lib.util import json2dict, dataset_type, keras_lid_to_torch_layers
from lib.log import set_exp_logging
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# plot setting
sns.set()

def set_new_weights(model, deltas, dic_keras_lid_to_torch_layers, device):
    """
    修正後の重みをモデルにセットする関数

    Parameters
    ------------------
    model: keras model
        修正前のモデル
    deltas: dict
        修正後の重みを保持する辞書
    dic_keras_lid_to_torch_layers: dict
        kerasモデルでの修正対象レイヤのインデックスとtorchモデルのレイヤの対応辞書
    device: str

    Returns
    ------------------
    model: keras model
        deltasで示される重みをセットした後の, 修正後のモデル
    """
    for idx_to_tl, delta in deltas.items():
        if isinstance(idx_to_tl, Iterable):
            idx_to_tl, idx_to_w = idx_to_tl
        else:
            idx_to_tl = idx_to_tl
        tl = dic_keras_lid_to_torch_layers[idx_to_tl]
        lname = tl.__class__.__name__
        if lname == "Conv2d" or lname == "Linear":
            tl.weight.data = torch.from_numpy(delta).to(device)
        elif lname == "LSTM":
            for i, tp in enumerate(tl.parameters()):
                if i == idx_to_w:
                    tp.data = torch.from_numpy(delta).to(device)
        else:
            print("{} not supported".format(lname))
            assert False
    return model


if __name__ == "__main__":
    # gpuが使用可能かをdeviceにセット
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["fmc", "c10c"], help="dataset name ('fmc' or 'c10c')")
    args = parser.parse_args()
    ds = args.dataset
    method = "arachne"
    # log setting
    exp_dir = f"/src/experiments/{method}/"
    care_dir = "/src/experiments/care/"
    this_file_name = os.path.basename(__file__).replace(".py", "").replace("_", "-")
    log_file_name = f"{ds}-{this_file_name}"
    logger = set_exp_logging(exp_dir, log_file_name, log_file_name)
    logger.info(f"target dataset: {ds}")
    
    # original dataset
    if ds in ["fmc", "c10c"]:
        ori_ds = ds.rstrip("c")
        num_fold = 5
        ds_type = dataset_type(ori_ds)
        is_binary = False
    else:
        raise ValueError(f"Invalid dataset name: {ds}")

    # 正解不正解の配列を保存するためのdir
    correctness_dir = os.path.join(f"./correctness_before", ds)
    # repair break datasetsのfoldごとのcsvを保存するdir
    rb_apricot_save_dir = os.path.join(f"./rb_datasets", ds, method)
    os.makedirs(rb_apricot_save_dir, exist_ok=True)

    # モデルとデータの読み込み先のディレクトリ
    data_dir = f"/src/data/{ds}/"
    # dataloaderのファイル名だけ取り出す
    dl_files = [f for f in os.listdir(data_dir) if f.endswith("loader.pt")]
    model_dir = f"/src/models/{ori_ds}/{ori_ds}-training-setting1"

    # dataloaderを読み込む
    dl_dic = {}
    for file_name in dl_files:
        file_path = os.path.join(data_dir, file_name)
        # file_nameから'_loader.pt'を除いた部分だけ取得
        key = file_name.replace("_loader.pt", "")
        dl_dic[key] = torch.load(file_path)

    # 各foldのtrain/repairをロードして予測
    for k in range(num_fold):
        logger.info(f"processing fold {k}...")

        # 学習済みモデルをロード
        model = select_model(task_name=ori_ds)
        model_path = os.path.join(model_dir, f"trained_model_fold-{k}.pt")
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # idx_to_tlとmodelのレイヤの対応
        dic_keras_lid_to_torch_layers = keras_lid_to_torch_layers(task_name=ori_ds, model=model)

        # dl_dicにあるdlのそれぞれに対して
        for key, dl in dl_dic.items():
            logger.info(f"processing dl {key}...")
            # fold k のモデルのkeyに対するexp metricsをロード
            exp_metrics_path = os.path.join(care_dir, "explanatory_metrics", ds, f"{key}_fold{k+1}.csv")
            df_expmet = pd.read_csv(exp_metrics_path)
            # 修正前にあってたかどうかの配列
            is_corr_bef = np.load(os.path.join(correctness_dir, f"{key}-fold-{k+1}.npy"))
            # 修正後の予測結果を格納するための配列
            is_corr_aft = np.zeros((len(is_corr_bef), 5))

            # 修正のnum_reps回の適用結果に関してそれぞれ見ていく
            for rep in range(5):
                logger.info(f"processing rep {rep}...")
                # arachneの修正済みweightをロード
                weight_save_dir = os.path.join(model_dir, "arachne-weight", f"rep{rep}")
                weight_file_name = [f for f in os.listdir(weight_save_dir) if f.endswith(f"{k}.pkl")][0]
                weight_save_path = os.path.join(weight_save_dir, weight_file_name)
                with open(weight_save_path, "rb") as f:
                    deltas = pickle.load(f)
                # Arachneの結果えられたdeltasをモデルにセット
                logger.info("Set the patches to the model...")
                repaired_model = set_new_weights(model, deltas, dic_keras_lid_to_torch_layers, device)

                # 予測を実行して結果を取得
                ret_dict = eval_model(
                    model=repaired_model,
                    dataloader=dl,
                    dataset_type=ds_type,
                    is_binary=is_binary,
                    device=device,
                )
                is_corr_aft_rep = ret_dict["correctness_arr"]
                is_corr_aft[:, rep] = is_corr_aft_rep
            is_corr_aft_sum = np.sum(is_corr_aft, axis=1, dtype=np.int32)
            
            # 修正前にあってたかどうかと，5回の修正それぞれの後で正しく予測できた回数の合計をまとめたDataFrameを作成
            df = pd.DataFrame({"sm_corr_bef": is_corr_bef, "sm_corr_aft_sum": is_corr_aft_sum})
            # repaired, brokenの真偽を決定
            # df["repaired"] = (df["sm_corr_bef"] == 0) & (df["sm_corr_aft_sum"] == 5)  # 厳し目の決定方法
            df["repaired"] = (df["sm_corr_bef"] == 0) & (df["sm_corr_aft_sum"] >= 1)  # ゆる目の決定方法
            df["broken"] = (df["sm_corr_bef"] == 1) & (df["sm_corr_aft_sum"] != 5)  # 厳し目の決定方法
            logger.info(f"df_sm.shape: {df.shape}")
            # df.to_csv(os.path.join(rb_apricot_save_dir, f"{key}_fold{k+1}.csv"), index=False)
            # logger.info(f'saved to {os.path.join(rb_apricot_save_dir, f"{key}_fold{k+1}.csv")}')

            # exp. metricsと，repaied, brokenなどの列と結合する
            df_all = pd.concat([df_expmet, df], axis=1)
            logger.info(f"df_all.shape: {df_all.shape}")

            # repair, breakのデータセットを作成
            df_repair, df_break = df_all[df_all["sm_corr_bef"] == 0], df_all[df_all["sm_corr_bef"] == 1]
            # それぞれのdfからいらない列を削除
            df_repair = df_repair.drop(["sm_corr_bef", "sm_corr_aft_sum", "broken"], axis=1)
            df_break = df_break.drop(["sm_corr_bef", "sm_corr_aft_sum", "repaired"], axis=1)
            logger.info(f"df_repair.shape: {df_repair.shape}, df_break.shape: {df_break.shape}")
            # df_repair, df_breakをそれぞれ保存
            df_repair.to_csv(os.path.join(rb_apricot_save_dir, f"{key}_fold{k+1}-repair.csv"), index=False)
            df_break.to_csv(os.path.join(rb_apricot_save_dir, f"{key}_fold{k+1}-break.csv"), index=False)