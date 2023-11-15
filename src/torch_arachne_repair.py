import os, sys, re, time, argparse, pickle
from collections import defaultdict
from ast import literal_eval

import numpy as np
import pandas as pd

import torch
from lib.model import eval_model, select_model
from lib.util import json2dict, dataset_type, fix_dataloader, keras_lid_to_torch_layers
from lib.log import set_exp_logging
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Arachne関連のimport
import src_arachne.torch_search.de_vk as de

sys.path.append(os.pardir)
# plot setting
sns.set_style("white")

# ignore warnings
warnings.filterwarnings("ignore")


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
        tl = dic_keras_lid_to_torch_layers[idx_to_tl]
        lname = tl.__class__.__name__
        if lname == "Conv2d" or lname == "Linear":
            tl.weight.data = torch.from_numpy(delta).to(device)
        # TODO: LSTMレイヤへの対応
        elif lname == "LSTM":
            pass
            # if idx_to_w == 0:  # kernel
            #     new_kernel_w = delta  # use the full
            #     new_recurr_kernel_w = self.init_weights[(idx_to_t_mdl_l, 1)]
            # elif idx_to_w == 1:
            #     new_recurr_kernel_w = delta
            #     new_kernel_w = self.init_weights[(idx_to_t_mdl_l, 0)]
            # else:
            #     print("{} not allowed".format(idx_to_w), idx_to_t_mdl_l, idx_to_tl)
            #     assert False
            # set kernel, recurr kernel, bias
            # fn_mdl.layers[idx_to_t_mdl_l].set_weights(
            #     [new_kernel_w, new_recurr_kernel_w, self.init_biases[idx_to_t_mdl_l]]
            # )
        else:
            print("{} not supported".format(lname))
            assert False
    return model


def load_localization_info(arachne_dir, model_dir, task_name, model, k, rep, device):
    """
    localization時の情報を読み込む部分が長いのでまとめただけの関数
    """
    ################################################
    # localize時に保存したrepair用データを読み込み #
    ################################################

    used_data_save_dir = os.path.join(arachne_dir, "used_data", task_name, f"rep{rep}")
    used_data_save_path = os.path.join(used_data_save_dir, f"X-y_for_loc-repair_fold-{k}.npz")
    used_data = np.load(used_data_save_path)
    # just for checking
    for kw in used_data.files:
        logger.info(f"{kw}.shape = {used_data[kw].shape}")
    # checking end

    # repairのためのデータを抽出
    X_for_repair_keras, y_for_repair_keras = used_data["X_for_repair"], used_data["y_for_repair"]
    if task_name in ["c10", "gtsrb", "fm"]:
        X_for_repair_torch = np.transpose(X_for_repair_keras, (0, 3, 1, 2)) # channel firstにもどす
    else:
        X_for_repair_torch = X_for_repair_keras
    X_for_repair, y_for_repair = torch.from_numpy(X_for_repair_torch.astype(np.float32)).clone().to(device), torch.from_numpy(y_for_repair_keras).clone().to(device)
    # 予測の成功or失敗数を確認
    # NOTE: model.predict(X_for_repair, device=device)["pred"] とするとX_for_repairのサイズが大きすぎてGTSRBでメモリエラーになる
    # なのでバッチ化する
    tmp_ds = TensorDataset(X_for_repair, y_for_repair)
    tmp_dl = DataLoader(tmp_ds, batch_size=128, shuffle=False)
    pred_labels = []
    for x, _ in tmp_dl:
        tmp_pred = model.predict(x, device=device)["pred"].to("cpu").detach().numpy().copy()
        pred_labels.append(tmp_pred)
    pred_labels = np.concatenate(pred_labels, axis=0)
    y_for_repair = y_for_repair.to('cpu').detach().numpy().copy()
    print(pred_labels.shape, y_for_repair.shape)
    correct_predictions = pred_labels == y_for_repair
    logger.info(
        f"correct predictions in (X_for_repair, y_for_repair): {np.sum(correct_predictions)} / {len(correct_predictions)}"
    )
    # 正誤データのインデックス確認
    num_wrong = len(correct_predictions) - np.sum(correct_predictions)
    indices_to_wrong = list(range(num_wrong))
    indices_to_correct = list(range(num_wrong, len(correct_predictions)))
    # 不正解データ
    pred_labels_for_neg = pred_labels[indices_to_wrong]
    is_wrong_arr = pred_labels_for_neg != y_for_repair[indices_to_wrong]
    # is_wrong_arrが全てTrue (indices to wrongから抜き出したデータは全て不正解)
    assert np.sum(is_wrong_arr) == len(is_wrong_arr)
    # 誤分類元のラベル, 誤分類先のラベルを取得
    misclf_true, misclf_pred = y_for_repair[indices_to_wrong][0], pred_labels_for_neg[0]
    # 正解データ
    pred_labels_for_corr = pred_labels[indices_to_correct]
    is_correct_arr = pred_labels_for_corr == y_for_repair[indices_to_correct]
    # is_correct_arrが全てTrue (indices to correctから抜き出したデータは全て正解)
    assert np.sum(is_correct_arr) == len(is_correct_arr)

    ##############################
    # localizationの結果をロード #
    ##############################
    loc_dir = os.path.join(model_dir, "arachne-localization")
    loc_path = os.path.join(loc_dir, f"rep{rep}", f"place_to_fix_fold-{k}.csv")
    places_df = pd.read_csv(loc_path)
    places_list = []
    for idx_to_tl, pair in places_df.iterrows():
        # literal_evalを入れるのはweightの位置を示すペアのtupleが何故か文字列で入ってるから
        places_list.append((pair["layer"], literal_eval(pair["weight"])))
    logger.info(f"places_list (keras-styled shape): {places_list}")
    indices_to_ptarget_layers = sorted(list(set([idx_to_tl for idx_to_tl, _ in places_list])))
    logger.info(f"Patch target layers: {indices_to_ptarget_layers}")

    return (
        X_for_repair,
        y_for_repair,
        places_list,
        indices_to_ptarget_layers,
        indices_to_correct,
        indices_to_wrong,
        misclf_true,
        misclf_pred,
    )

def reshape_places_tf_to_torch(places_list):
    """
    localizeした重みの位置を表すタプルの辞書を, tfの形状からtorchの形状にreshapeする.
    """
    reshaped_places_list = []
    for lid, nid in places_list:
        if len(nid) == 4: # Conv2d
            place = (lid, (nid[3], nid[2], nid[0], nid[1]))
        elif len(nid) == 2: # Linear
            place = (lid, (nid[1], nid[0]))
        reshaped_places_list.append(place)
    return reshaped_places_list



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("fold", type=int)
    parser.add_argument("rep", type=int)
    parser.add_argument("--mode", choices=["repair", "predict"], default="repair")
    args = parser.parse_args()
    mode = args.mode
    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(args.path)
    arachne_dir = exp_dir.replace("care", "arachne")
    exp_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]

    # log setting
    log_file_name = exp_name.replace("training", f"arachne-{mode}")
    logger = set_exp_logging(exp_dir.replace("care", "arachne"), exp_name, log_file_name)

    # GPU使えるかチェック
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # コマンドライン引数からfoldとrepの取得
    k = args.fold
    rep = args.rep
    logger.info(f"exp_dir={exp_dir}, fold={k}, rep={rep}, mode={mode}")

    # 設定用のjsonファイルをdictとしてロード
    # HACK: 共通しているので関数にまとめて自動化したい
    setting_dict = json2dict(sys.argv[1])
    logger.info(f"Settings: {setting_dict}")
    task_name = setting_dict["TASK_NAME"]
    num_fold = setting_dict["NUM_FOLD"]
    # binary clf. かどうかのフラグ
    if task_name in ["fm", "c10", "gtsrb"]:
        is_binary = False
    elif task_name in ["credit", "census", "bank"]: 
        is_binary = True

    # モデルとデータの読み込み先のディレクトリ
    data_dir = f"/src/data/{task_name}/{exp_name}"
    model_dir = f"/src/models/{task_name}/{exp_name}"

    # localizationの結果保存用のディレクトリ
    loc_save_dir = os.path.join(model_dir, "arachne-localization")

    # 対象とする誤分類の発生頻度の順位 (0は1位, 1は2位, ...)
    topn = 0

    # 学習済みモデルをロード (torch)
    model = select_model(task_name=task_name)
    model_path = os.path.join(model_dir, f"trained_model_fold-{k}.pt")
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device=device)))
    model.to(device)
    model.eval()
    dic_keras_lid_to_torch_layers = keras_lid_to_torch_layers(task_name, model)

    # localization時の情報をロード
    (
        X_for_repair,
        y_for_repair,
        places_list,
        indices_to_ptarget_layers,
        indices_to_correct,
        indices_to_wrong,
        misclf_true,
        misclf_pred,
    ) = load_localization_info(arachne_dir, model_dir, task_name, model, k, rep, device)
    # 特定したニューロン位置のインデックスをtfからtorchにする
    places_list = reshape_places_tf_to_torch(places_list)
    logger.info(f"Reshaped places list = {places_list}")

    # 修正後の重みのファイル名
    file_name = f"misclf-top{topn}-{misclf_true}to{misclf_pred}_fold-{k}.pkl"
    # 修正後の重みを格納するディレクトリ名
    save_dir = os.path.join(model_dir, "arachne-weight", f"rep{rep}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)

    # idx_to_tlとmodelのレイヤの対応
    dic_keras_lid_to_torch_layers = keras_lid_to_torch_layers(task_name=task_name, model=model)

    if mode == "repair":
        # deltas (修正対象レイヤの重みを保持する辞書) を初期化
        deltas = {}
        for lid in indices_to_ptarget_layers:
            target_layer = dic_keras_lid_to_torch_layers[lid]
            target_param = target_layer.weight
            deltas[lid] = target_param
        # この間でdeltasをArachneの手法で更新していけばいいという話
        # searcherのinitializerに入れる変数をここで定義 HACK: 外部化
        num_label = len(set(y_for_repair))
        max_search_num = 100
        patch_aggr = 10
        batch_size =128
        # searchのためのクラスのインスタンス化
        searcher = de.DE_searcher(
            inputs=X_for_repair.to("cpu").detach().numpy().copy().astype(np.float32),
            labels=y_for_repair,
            indices_to_correct=indices_to_correct,
            indices_to_wrong=indices_to_wrong,
            num_label=num_label,
            indices_to_target_layers=indices_to_ptarget_layers,
            mutation=(0.5, 1),
            recombination=0.7,
            task_name=task_name,
            device=device,
            max_search_num=max_search_num,
            model=model,
            patch_aggr=patch_aggr,
            batch_size=batch_size,
        )

        logger.info(f"Start DE search...")
        s = time.clock()
        searcher.search(places_list, save_path=save_path)
        e = time.clock()
        logger.info(f"Total execution time: {e-s} sec.")

    elif mode == "predict":
        # 修正済みのdeltasをロード
        logger.info(f"Loading patches from {save_path}...")
        with open(save_path, "rb") as f:
            deltas = pickle.load(f)

        # Arachneの結果えられたdeltasをモデルにセット
        logger.info("Set the patches to the model...")
        repaired_model = set_new_weights(model, deltas, dic_keras_lid_to_torch_layers, device)
        
        # TODO: できたらtrain, repair, testの予測結果を出して比較する. keras ver. 同様にis_corr_dictみたいな名前のやつを保存する
        # train, repair, test dataloader
        train_data_path = os.path.join(data_dir, f"train_loader_fold-{k}.pt")
        train_loader = fix_dataloader(torch.load(train_data_path))
        repair_data_path = os.path.join(data_dir, f"repair_loader_fold-{k}.pt")
        repair_loader = fix_dataloader(torch.load(repair_data_path))
        test_data_path = os.path.join(data_dir, f"test_loader.pt")
        test_loader = torch.load(test_data_path) # これはfixしたらバグる(もともとtestなのでシャッフルされてないけどそれとの関係は不明)
        # train, repair, testそれぞれのfix_loaderに対してeval_modelを実行
        logger.info("Eval the repaired model with patches...")
        train_ret_dict = eval_model(repaired_model, train_loader, is_binary=is_binary, device=device)
        repair_ret_dict = eval_model(repaired_model, repair_loader, is_binary=is_binary, device=device)
        test_ret_dict = eval_model(repaired_model, test_loader, is_binary=is_binary, device=device)
        # 各サンプルに対する予測の正解(1)か不正解か(0)のnumpy配列を取得
        is_corr_train = train_ret_dict["correctness_arr"]
        is_corr_repair = repair_ret_dict["correctness_arr"]
        is_corr_test = test_ret_dict["correctness_arr"]
        # is_corr_dicの情報を表示しておく
        logger.info(f"train: {np.sum(is_corr_train)} / {len(is_corr_train)}, repair: {np.sum(is_corr_repair)} / {len(is_corr_repair)}, test: {np.sum(is_corr_test)} / {len(is_corr_test)}")
        # is_corr_dicの保存先
        check_save_dir = os.path.join(arachne_dir, "check_repair_results", task_name, f"rep{rep}")
        os.makedirs(check_save_dir, exist_ok=True)
        check_save_path = os.path.join(check_save_dir, f"is_corr_fold-{k}.npz")
        # npz形式で保存
        np.savez(check_save_path, train=is_corr_train, repair=is_corr_repair, test=is_corr_test)
        logger.info(f"save is_corr_dic to {check_save_path}")
    else:
        raise ValueError(f"Invalid mode: {mode}")
