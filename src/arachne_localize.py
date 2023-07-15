import os, sys
from collections import defaultdict

import numpy as np
from lib.model import get_misclassified_index, sort_keys_by_cnt
from keras.models import load_model
from lib.util import json2dict
from lib.log import set_exp_logging
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# plot setting
sns.set_style("white")

# ignore warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(sys.argv[1])
    exp_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]

    # log setting
    log_file_name = exp_name.replace("training", "arachne-localize")
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

    # 対象とする誤分類の発生頻度の順位 (0は1位, 1は2位, ...)
    topn = 0

    # 各foldにおいてrepair loaderを読み込む
    for k in range(num_fold):
        logger.info(f"processing fold {k}...")

        # repair loaderをロード
        repair_data_path = os.path.join(data_dir, f"repair_loader_fold-{k}.pt")
        repair_loader = torch.load(repair_data_path)
        repair_ds = repair_loader.dataset
        X_repair, y_repair = repair_ds.tensors[0].detach().numpy().copy(), repair_ds.tensors[1].detach().numpy().copy()
        logger.info(f"X_repair.shape = {X_repair.shape}, y_repair.shape = {y_repair.shape}")

        # 学習済みモデルをロード (keras)
        model = load_model(os.path.join(model_dir, f"keras_model_fold-{k}.h5"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])  # これがないと予測できない(エラーになる)
        # 予測のスコアとそこから予測ラベル取得
        pred_scores = model.predict(X_repair, verbose=-1)
        pred_labels = np.argmax(pred_scores, axis=1)
        correct_predictions = pred_labels == y_repair

        # 正解したデータと失敗したデータに分ける
        indices = {"wrong": [], "correct": []}
        (indices_to_wrong,) = np.where(correct_predictions == False)
        (indices_to_correct,) = np.where(correct_predictions == True)
        indices["correct"] = list(indices_to_correct)
        indices["wrong"] = list(indices_to_wrong)
        # 正しい予測の数と間違った予測の数を表示
        logger.info(f"correct predictions: {len(indices['correct'])}, wrong predictions: {len(indices['wrong'])}")

        # 誤分類パターンごとのデータのインデックス取得
        misclf_dic = get_misclassified_index(y_repair, pred_labels)
        # 対象の誤分類パターンを取得
        target_miclf = sort_keys_by_cnt(misclf_dic)[topn]
        # 対象とする誤分類パターンを起こすインデックスを取得
        target_indices_wrong = misclf_dic[target_miclf]
        logger.info(f"target_miclf: {target_miclf}, num of target_misclf: {len(target_indices_wrong)}")

        # just for checking
        s = 0
        for k, v in misclf_dic.items():
            s += len(v)
        # sとlen(indices['wrong'])は一致するはず.
        assert s == len(indices["wrong"])
        # checking end

        # target_indicesと同じ数のデータを，正しい予測のデータからランダムにサンプリングする
        # この際, 正しいデータからのサンプリングは各ラベルから均等になるようにサンプリングする
        uniq_labels = np.unique(y_repair[indices["correct"]])

        # ラベルごとのインデックスを格納する辞書
        grouped_by_label = {uniq_label: [] for uniq_label in uniq_labels}
        for idx in indices["correct"]:
            pred_label = pred_labels[idx]
            assert pred_label == y_repair[idx]  # 正解のはずなので
            grouped_by_label[pred_label].append(idx)

        sampled_indices_correct = []
        # 各ラベルごとにサンプリングする数を計算
        for l, idxs in grouped_by_label.items():
            # 間違いと正解の比率からサンプル数を決定
            num_sample = int(np.round(len(target_indices_wrong) * len(idxs) / len(indices["correct"])))
            # 例外処理的な
            if num_sample <= 0:
                num_sample = 1
            if num_sample > len(idxs):
                num_sample = len(idxs)
            # どのラベルからどれだけサンプリングされるかをログ出力
            logger.info(f"label: {l}, num_sample: {num_sample}")
            # ラベルごとにサンプリング
            sampled_indices_correct.extend(list(np.random.choice(idxs, num_sample, replace=False)))
        # sampled_indices_correctとtarget_indices_wrongの数は一致するはず
        logger.info(
            f"len(sampled_indices_correct): {len(sampled_indices_correct)}, len(target_indices_wrong): {len(target_indices_wrong)}"
        )

        # localizationのために使うデータのインデックス
        indices_for_loc = sampled_indices_correct + target_indices_wrong
        # インデックスからデータを取得
        X_for_loc, y_for_loc = X_repair[indices_for_loc], y_repair[indices_for_loc]
        logger.info(f"X_for_loc.shape = {X_for_loc.shape}, y_for_loc.shape = {y_for_loc.shape}")
