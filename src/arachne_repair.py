import os, sys, re, time, argparse, pickle
from collections import defaultdict
from ast import literal_eval

import numpy as np
import pandas as pd
from lib.model import get_misclassified_index, sort_keys_by_cnt

# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import tensorflow.keras.backend as K

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from sklearn.preprocessing import Normalizer
from lib.util import json2dict, dataset_type, fix_dataloader
from lib.log import set_exp_logging
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Arachne関連のimport
import src_arachne.search.de_vk as de

# plot setting
sns.set_style("white")

# ignore warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("fold", type=int)
    parser.add_argument("rep", type=int)
    parser.add_argument("--mode", choices=["repair", "check"])
    args = parser.parse_args()
    mode = args.mode
    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(args.path)
    arachne_dir = exp_dir.replace("care", "arachne")
    exp_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]

    # log setting
    log_file_name = exp_name.replace("training", "arachne-repair")
    logger = set_exp_logging(exp_dir.replace("care", "arachne"), exp_name, log_file_name)

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

    # 学習済みモデルをロード (keras)
    model = load_model(os.path.join(model_dir, f"keras_model_fold-{k}.h5"))
    if dataset_type(task_name) == "tabular":
        loss_fn = "binary_crossentropy"
    elif dataset_type(task_name) == "image":
        loss_fn = "categorical_crossentropy"
    # TODO: for text dataset
    # elif dataset_type(task_name) == "text":
    model.compile(loss=loss_fn, optimizer="adam", metrics=["accuracy"])  # これがないと予測できない(エラーになる)

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
    X_for_repair, y_for_repair = used_data["X_for_repair"], used_data["y_for_repair"]
    # 予測の成功or失敗数を確認
    pred_scores = model.predict(X_for_repair, verbose=-1)
    pred_labels = np.argmax(pred_scores, axis=1)
    correct_predictions = pred_labels == y_for_repair
    logger.info(
        f"correct predictions in (X_for_repair, y_for_repair): {np.sum(correct_predictions)} / {len(correct_predictions)}"
    )

    # データのインデックス確認
    num_wrong = len(correct_predictions) - np.sum(correct_predictions)
    indices_to_wrong = list(range(num_wrong))
    indices_to_correct = list(range(num_wrong, len(correct_predictions)))

    # インデックスと予測結果の一貫性確認
    # for indices to wrong
    pred_scores = model.predict(X_for_repair[indices_to_wrong], verbose=-1)
    pred_labels = np.argmax(pred_scores, axis=1)
    is_wrong_arr = pred_labels != y_for_repair[indices_to_wrong]
    # is_wrong_arrが全てTrue (indices to wrongから抜き出したデータは全て不正解)
    assert np.sum(is_wrong_arr) == len(is_wrong_arr)
    # 誤分類元のラベル, 誤分類先のラベルを取得
    misclf_true, misclf_pred = y_for_repair[indices_to_wrong][0], pred_labels[0]

    # for indices to correct
    pred_scores = model.predict(X_for_repair[indices_to_correct], verbose=-1)
    pred_labels = np.argmax(pred_scores, axis=1)
    is_correct_arr = pred_labels == y_for_repair[indices_to_correct]
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
    logger.info(f"places_list: {places_list}")
    indices_to_ptarget_layers = sorted(list(set([idx_to_tl for idx_to_tl, _ in places_list])))
    logger.info(f"Patch target layers: {indices_to_ptarget_layers}")

    ################################################
    # differential evolutionによるrepairを適用する #
    ################################################

    # 開始時間計測
    s = time.clock()

    # searcherのinitializerに入れる変数をここで定義 TODO: 外部化
    num_label = len(set(y_for_repair))
    max_search_num = 100
    patch_aggr = 10
    batch_size = 64

    # searchのためのクラスのインスタンス化
    searcher = de.DE_searcher(
        inputs=np.float32(X_for_repair),
        labels=y_for_repair,
        indices_to_correct=indices_to_correct,
        indices_to_wrong=[],
        num_label=num_label,
        indices_to_target_layers=indices_to_ptarget_layers,
        mutation=(0.5, 1),
        recombination=0.7,
        max_search_num=max_search_num,
        initial_predictions=None,
        model=model,
        patch_aggr=patch_aggr,
        batch_size=batch_size,
        act_func=tf.nn.relu,
        X_train=X_train,
        X_repair=X_repair,
        X_test=X_test,
        y_train=y_train,
        y_repair=y_repair,
        y_test=y_test,
    )
    # DEによるrepairを実行
    searcher.set_indices_to_wrong(indices_to_wrong)
    # 修正後の重みのファイル名
    file_name = f"misclf-top{topn}-{misclf_true}to{misclf_pred}_fold-{k}.pkl"
    # 修正後の重みを格納するディレクトリ名
    save_dir = os.path.join(model_dir, "arachne-weight", f"rep{rep}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    # repairをする場合
    if mode == "repair":
        # DEによるrepairを実行
        logger.info(f"Start DE search...")
        searcher.search(places_list, save_path=save_path)
        # 終了時間計測
        e = time.clock()
        logger.info(f"Total execution time: {e-s} sec.")
    # repairの結果を各divisionでチェックする場合
    elif mode == "check":
        with open(save_path, "rb") as f:
            deltas = pickle.load(f)
        print(deltas.keys())
        is_corr_dic = searcher.summarise_results(deltas)
        # is_corr_dicの保存先
        check_save_dir = os.path.join(arachne_dir, "check_repair_results", task_name, f"rep{rep}")
        os.makedirs(check_save_dir, exist_ok=True)
        check_save_path = os.path.join(check_save_dir, f"is_corr_fold-{k}.npz")
        # npz形式で保存
        np.savez(check_save_path, train=is_corr_dic["train"], repair=is_corr_dic["repair"], test=is_corr_dic["test"])
        logger.info(f"save is_corr_dic to {check_save_path}")
