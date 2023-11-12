import os, sys, re, time, argparse, pickle
from collections import defaultdict
from ast import literal_eval

import numpy as np
import pandas as pd

import keras as K
import torch
# import tensorflow as tf
from keras.models import Model, load_model
from lib.model import get_misclassified_index, sort_keys_by_cnt, select_model
from lib.util import json2dict, dataset_type, fix_dataloader, keras_lid_to_torch_layers
from lib.log import set_exp_logging
import torch
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


def set_new_weights(model, deltas):
    """
    修正後の重みをモデルにセットする関数

    Parameters
    ------------------
    model: keras model
        修正前のモデル
    deltas: dict
        修正後の重みを保持する辞書

    Returns
    ------------------
    model: keras model
        deltasで示される重みをセットした後の, 修正後のモデル
    """

    # 修正後の重みをセット
    for lid, delta in deltas.items():
        # 修正対象のレイヤの重みを取得
        weights = model.layers[lid].get_weights()
        # NOTE: レイヤがdense, conv2dの場合はこれでいけるがそれ以外だとエラーになるかも
        # 修正後の重みをセット
        model.layers[lid].set_weights([delta, weights[1]])
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
    X_for_repair_torch = np.transpose(X_for_repair_keras, (0, 3, 1, 2)) # channel firstにもどす
    X_for_repair, y_for_repair = torch.from_numpy(X_for_repair_torch.astype(np.float32)).clone(), torch.from_numpy(y_for_repair_keras).clone()
    # 予測の成功or失敗数を確認
    pred_labels = model.predict(X_for_repair, device=device)["pred"]
    correct_predictions = pred_labels == y_for_repair
    correct_predictions = correct_predictions.detach().numpy().copy()
    logger.info(
        f"correct predictions in (X_for_repair, y_for_repair): {np.sum(correct_predictions)} / {len(correct_predictions)}"
    )
    # 正誤データのインデックス確認
    num_wrong = len(correct_predictions) - np.sum(correct_predictions)
    indices_to_wrong = list(range(num_wrong))
    indices_to_correct = list(range(num_wrong, len(correct_predictions)))
    # 不正解データ
    pred_labels, y_for_repair = pred_labels.detach().numpy().copy(), y_for_repair.detach().numpy().copy()
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


def load_data(data_dir, k):
    # train set, repair set, test setをロード（最終確認用）
    train_data_path = os.path.join(data_dir, f"train_loader_fold-{k}.pt")
    train_loader = torch.load(train_data_path)
    repair_data_path = os.path.join(data_dir, f"repair_loader_fold-{k}.pt")
    repair_loader = torch.load(repair_data_path)
    test_data_path = os.path.join(data_dir, f"test_loader.pt")
    test_loader = torch.load(test_data_path)
    # X_train, X_repair, X_test, y_train, y_repair, y_testが必要なのでセットする
    X_train, X_repair, X_test, y_train, y_repair, y_test = [], [], [], [], [], []  # 6つの空のリストを作成
    # tabular datasetの場合はデータセット全体をそのままnumpy配列にする
    if dataset_type(task_name) == "tabular":
        train_ds = train_loader.dataset
        repair_ds = repair_loader.dataset
        test_ds = test_loader.dataset
        X_train, y_train = train_ds.tensors[0].detach().numpy().copy(), train_ds.tensors[1].detach().numpy().copy()
        logger.info(f"X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")
        print(f"X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")

        X_repair, y_repair = repair_ds.tensors[0].detach().numpy().copy(), repair_ds.tensors[1].detach().numpy().copy()
        logger.info(f"X_repair.shape = {X_repair.shape}, y_repair.shape = {y_repair.shape}")
        print(f"X_repair.shape = {X_repair.shape}, y_repair.shape = {y_repair.shape}")

        X_test, y_test = test_ds.x.detach().numpy().copy(), test_ds.y.detach().numpy().copy()
        logger.info(f"X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}")
        print(f"X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}")
    # TODO: image datasetの場合は？
    elif dataset_type(task_name) == "image":
        # train, repair, testそれぞれのfix_loaderに対して
        for div, fixed_loader in zip(
            ["train", "repair", "test"],
            [fix_dataloader(train_loader), fix_dataloader(repair_loader), fix_dataloader(test_loader)],
        ):
            # loader内の各バッチに対して
            for data, labels in fixed_loader:
                data, labels = (
                    data.detach().cpu().numpy().copy(),
                    labels.detach().cpu().numpy().copy(),
                )
                data = np.transpose(data, (0, 2, 3, 1))
                if div == "train":
                    X_train.append(data)
                    y_train.append(labels)
                elif div == "repair":
                    X_repair.append(data)
                    y_repair.append(labels)
                elif div == "test":
                    X_test.append(data)
                    y_test.append(labels)
        # X_train, X_repair, X_testをnumpy配列に変換
        X_train, X_repair, X_test = (
            np.concatenate(X_train, axis=0),
            np.concatenate(X_repair, axis=0),
            np.concatenate(X_test, axis=0),
        )
        # y_train, y_repair, y_testをnumpy配列に変換
        y_train, y_repair, y_test = (
            np.concatenate(y_train, axis=0),
            np.concatenate(y_repair, axis=0),
            np.concatenate(y_test, axis=0),
        )
        # 各データの形状を出力
        logger.info(
            f"X_train.shape = {X_train.shape}, X_repair.shape = {X_repair.shape}, X_test.shape = {X_test.shape}"
        )
        y_train, y_repair, y_test = np.array(y_train), np.array(y_repair), np.array(y_test)
        logger.info(
            f"y_train.shape = {y_train.shape}, y_repair.shape = {y_repair.shape}, y_test.shape = {y_test.shape}"
        )
        return X_train, X_repair, X_test, y_train, y_repair, y_test


def reshape_places_tf_to_torch(places_list):
    """
    localizeした重みの位置を表すタプルに辞書を, tfの形状からtorchの形状にreshapeする.
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
    parser.add_argument("--mode", choices=["repair", "check"], default="repair")
    args = parser.parse_args()
    mode = args.mode
    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(args.path)
    arachne_dir = exp_dir.replace("care", "arachne")
    exp_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]

    # log setting
    log_file_name = exp_name.replace("training", "arachne-repair")
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
    
    # # 学習済みモデルをロード (keras)
    # model = load_model(os.path.join(model_dir, f"keras_model_fold-{k}.h5"))
    # if dataset_type(task_name) == "tabular":
    #     loss_fn = "binary_crossentropy"
    # elif dataset_type(task_name) == "image":
    #     loss_fn = "categorical_crossentropy"
    # # TODO: for text dataset
    # # elif dataset_type(task_name) == "text":
    # model.compile(loss=loss_fn, optimizer="adam", metrics=["accuracy"])  # これがないと予測できない(エラーになる)

    # train, repair, test datasetをロード
    X_train, X_repair, X_test, y_train, y_repair, y_test = load_data(data_dir, k)

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
            inputs=np.float32(X_for_repair),
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
        # 修正後の重みのファイル名
        file_name = f"misclf-top{topn}-{misclf_true}to{misclf_pred}_fold-{k}.pkl"
        # 修正後の重みを格納するディレクトリ名
        save_dir = os.path.join(model_dir, "arachne-weight", f"rep{rep}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)

        logger.info(f"Start DE search...")
        s = time.clock()
        searcher.search(places_list, save_path=save_path)
        e = time.clock()
        logger.info(f"Total execution time: {e-s} sec.")

    elif mode == "check":
        # 修正済みのdeltasをロード
        deltas = {}  # TODO: load deltas
        # Arachneの結果えられたdeltasをセット
        new_model = set_new_weights(model, deltas)
    else:
        raise ValueError(f"Invalid mode: {mode}")
