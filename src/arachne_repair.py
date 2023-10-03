import os, sys, re
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
from lib.util import json2dict
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

# 実験の繰り返し数
num_reps = 5
topn = 0

if __name__ == "__main__":
    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(sys.argv[1])
    arachne_dir = exp_dir.replace("care", "arachne")
    exp_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]

    # log setting
    log_file_name = exp_name.replace("training", "arachne-repair")
    logger = set_exp_logging(exp_dir.replace("care", "arachne"), exp_name, log_file_name)

    # 設定用のjsonファイルをdictとしてロード
    # HACK: 共通しているので関数にまとめて自動化したい
    setting_dict = json2dict(sys.argv[1])
    logger.info(f"Settings: {setting_dict}")
    task_name = setting_dict["TASK_NAME"]
    target_column = setting_dict["TARGET_COLUMN"]
    num_fold = setting_dict["NUM_FOLD"]

    # モデルとデータの読み込み先のディレクトリ
    data_dir = f"/src/data/{task_name}/{exp_name}"
    model_dir = f"/src/models/{task_name}/{exp_name}"

    # localizationの結果保存用のディレクトリ
    loc_save_dir = os.path.join(model_dir, "arachne-localization")

    # 対象とする誤分類の発生頻度の順位 (0は1位, 1は2位, ...)
    topn = 0

    # 各foldにおいてArachne dirに保存してあるデータを読み込む
    for k in range(num_fold):
        logger.info(f"processing fold {k}...")

        # 学習済みモデルをロード (keras)
        model = load_model(os.path.join(model_dir, f"keras_model_fold-{k}.h5"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])  # これがないと予測できない(エラーになる)

        # train set, repair set, test setをロード（最終確認用）
        train_data_path = os.path.join(data_dir, f"train_loader_fold-{k}.pt")
        train_loader = torch.load(train_data_path)
        train_ds = train_loader.dataset
        X_train, y_train = train_ds.tensors[0].detach().numpy().copy(), train_ds.tensors[1].detach().numpy().copy()
        logger.info(f"X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")
        print(f"X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")

        repair_data_path = os.path.join(data_dir, f"repair_loader_fold-{k}.pt")
        repair_loader = torch.load(repair_data_path)
        repair_ds = repair_loader.dataset
        X_repair, y_repair = repair_ds.tensors[0].detach().numpy().copy(), repair_ds.tensors[1].detach().numpy().copy()
        logger.info(f"X_repair.shape = {X_repair.shape}, y_repair.shape = {y_repair.shape}")
        print(f"X_repair.shape = {X_repair.shape}, y_repair.shape = {y_repair.shape}")

        test_data_path = os.path.join(data_dir, f"test_loader.pt")
        test_loader = torch.load(test_data_path)
        test_ds = test_loader.dataset
        X_test, y_test = test_ds.x.detach().numpy().copy(), test_ds.y.detach().numpy().copy()
        logger.info(f"X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}")
        print(f"X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}")

        for rep in range(num_reps):
            logger.info(f"starting rep {rep}...")

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
            # DEによるrepairを実行
            is_corr_dic = searcher.search(places_list, save_path=save_path)

            # is_corr_dicの保存先
            check_save_dir = os.path.join(arachne_dir, "check_repair_results", task_name, f"rep{rep}")
            os.makedirs(check_save_dir, exist_ok=True)
            check_save_path = os.path.join(check_save_dir, f"is_corr_fold-{k}.npz")
            # npz形式で保存
            np.savez(
                check_save_path, train=is_corr_dic["train"], repair=is_corr_dic["repair"], test=is_corr_dic["test"]
            )
            logger.info(f"save is_corr_dic to {check_save_path}")
