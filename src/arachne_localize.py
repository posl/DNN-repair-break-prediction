import os, sys, re
from collections import defaultdict

import numpy as np
import pandas as pd
from lib.model import get_misclassified_index, sort_keys_by_cnt
import tensorflow as tf
import tensorflow.keras.backend as K

from keras.models import load_model, Model
from keras.layers import Input
from sklearn.preprocessing import Normalizer
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

# 実験の繰り返し数
num_reps = 5


def get_target_weights(model):
    """
    対象となるレイヤの重みを取得する

    Args:
        model (keras.models.Model): 対象となるモデル

    Returns:
        dict: key: layer index (int), value: [weight value, layer name]
    """
    # 対象となるレイヤ名のパターン
    targeting_clsname_pattns = ["Dense*", "Conv*", ".*LSTM*"]
    # 上のパターンのどれかにマッチするかを真偽値で返す関数
    is_target = lambda clsname, targets: (targets is None) or any([bool(re.match(t, clsname)) for t in targets])

    target_weights = {}  # key = layer index (int), value: [weight value, layer name]

    for i, layer in enumerate(model.layers):
        # 各レイヤが対象となるレイヤ名のパターンにマッチするかを判定
        if is_target(layer.__class__.__name__, targeting_clsname_pattns):
            # NOTE: Denseの場合は重みとバイアスのペアで返ってくる
            weights = layer.get_weights()
            if len(weights):  # has weights
                # TODO: Dense層以外にも対応 (Conv2Dの場合は一緒でいいらしい)
                target_weights[i] = [weights[0], layer.__class__.__name__]
    return target_weights


def compute_gradient_to_output(model, idx_to_target_layer, X, on_weight=False, wo_reset=False, by_batch=False):
    """
    compute gradients normalisesd and averaged for a given input X
    on_weight = False -> on output of idx_to_target_layer'th layer
    """
    norm_scaler = Normalizer(norm="l1")

    # since this might cause OOM error, divide them
    num = X.shape[0]
    if by_batch:
        batch_size = 64
        num_split = int(np.round(num / batch_size))
        if num_split == 0:
            num_split = 1
        chunks = np.array_split(np.arange(num), num_split)
    # by_batchがFalseの場合はchunkにわけない
    else:
        chunks = [np.arange(num)]

    if not on_weight:
        # 対象となるレイヤの出力
        target = model.layers[idx_to_target_layer].output
        # 勾配の形状を決定
        grad_shape = tuple([num] + [int(v) for v in target.shape[1:]])
        gradient = np.zeros(grad_shape)
        logger.info(f"gradient_to_output.shape = {gradient.shape}")
        # チャンクごとに処理
        for chunk in chunks:
            # 自動微分のためにチャンクのデータをtf.Variableにする
            x = tf.Variable(X[chunk])
            # tf.GradientTape()を使ってターゲットレイヤに関する出力の勾配を計算
            with tf.GradientTape() as tape:
                # 中間層の出力のリスト
                hidden_list = []
                # 最終層までの出力を1層ごとにトレース
                for l in model.layers:
                    x = l(x)
                    hidden_list.append(x)
                # 最終的な出力
                final_out = x
                # just for checking
                # print(f"final_out.shape: {final_out.shape}")
                # for h in hidden_list:
                # print(f"h.shape: {h.shape}")
                # checking end
            # 最終的な出力の対象となる層の出力に関する勾配を取得
            _gradient = tape.gradient(final_out, hidden_list[idx_to_target_layer])
            # チャンクごとの勾配を格納
            gradient[chunk] = _gradient
        gradient = np.abs(gradient)
        reshaped_gradient = gradient.reshape(gradient.shape[0], -1)  # flatten
        norm_gradient = norm_scaler.fit_transform(reshaped_gradient)  # normalised
        mean_gradient = np.mean(norm_gradient, axis=0)  # compute mean for a given input
        ret_gradient = mean_gradient.reshape(gradient.shape[1:])  # reshape to the orignal shape
        logger.info(f"ret_gradient.shape = {ret_gradient.shape}")
        return ret_gradient

    # TODO: on weights(いつ使うんかしらんけど)の方も対処
    # else:  # on weights
    #     with tf.GradientTape() as tape:
    #         # 対象となるレイヤの重み（バイアス除く）
    #         target = model.layers[idx_to_target_layer].weights[:-1]  # exclude the bias
    #         gradients = []
    #         for chunk in chunks:
    #             _gradients = K.get_session().run(tensor_grad, feed_dict={model.input: X[chunk]})
    #             if len(gradients) == 0:
    #                 gradients = _gradients
    #             else:
    #                 for i in range(len(_gradients)):
    #                     gradients[i] += _gradients[i]
    #         ret_gradients = list(map(np.abs, gradients))
    #         if not wo_reset:
    #             reset_keras([tensor_grad])

    #         if len(ret_gradients) == 0:
    #             return ret_gradients[0]
    #         else:
    #             return ret_gradients


def compute_gradient_to_loss(model, idx_to_target_layer, X, y, loss_func_name, by_batch=False, **kwargs):
    """
    compute gradients for the loss.
    kwargs contains the key-word argumenets required for the loss funation
    """
    targets = model.layers[idx_to_target_layer].weights[:-1]

    if len(model.output.shape) == 3:
        y_tensor = tf.keras.Input(shape=(model.output.shape[-1],), name="labels")
    else:  # is not multi label
        y_tensor = tf.keras.Input(shape=list(model.output.shape)[1:], name="labels")

    # since this might cause OOM error, divide them
    num = X.shape[0]
    if by_batch:
        batch_size = 64
        num_split = int(np.round(num / batch_size))
        if num_split == 0:
            num_split += 1
        chunks = np.array_split(np.arange(num), num_split)
    else:
        chunks = [np.arange(num)]

    # loss_funcの設定
    if loss_func_name == "categorical_cross_entropy":
        loss_func = tf.nn.softmax_cross_entropy_with_logits_v2
    elif loss_func_name == "binary_crossentropy":
        if "name" in kwargs.keys():
            kwargs.pop("name")
        loss_func = tf.keras.losses.binary_crossentropy
        y = y.reshape(-1, 1)
    elif loss_func_name in ["mean_squared_error", "mse"]:
        loss_func = tf.keras.losses.MeanSquaredError
    else:
        print(loss_func)
        print("{} not supported yet".format(loss_func))
        assert False

    gradients = [[] for _ in range(len(targets))]
    for chunk in chunks:
        # 自動微分のためにチャンクのデータをtf.Variableにする
        x = tf.Variable(X[chunk])
        y_true = tf.Variable(y[chunk])
        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = loss_func(y_true, y_pred)
        # モデル出力のロスの対象のレイヤの重みに関する勾配を取得
        _gradients = tape.gradient(loss, targets)
        for i, _gradient in enumerate(_gradients):
            gradients[i].append(_gradient)

    for i, gradients_p_chunk in enumerate(gradients):
        gradients[i] = np.abs(np.sum(np.asarray(gradients_p_chunk), axis=0))  # combine

    # if not wo_reset:
    #     reset_keras(gradients + [loss_tensor, y_tensor])
    return gradients[0] if len(gradients) == 1 else gradients


def reset_keras(delete_list=None, frac=1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=frac)
    config = tf.ConfigProto(gpu_options=gpu_options)

    if delete_list is None:
        K.clear_session()
        s = tf.InteractiveSession(config=config)
        K.set_session(s)
    else:
        import gc

        K.clear_session()
        try:
            for d in delete_list:
                del d
        except:
            pass
        gc.collect()
        K.set_session(tf.Session(config=config))


def compute_FI_and_GL(X, y, indices_to_target, target_weights, model):
    norm_scaler = Normalizer(norm="l1")
    total_cands = {}
    FIs, grad_scndcr = None, None
    target_X = X[indices_to_target]
    target_y = y[indices_to_target]
    loss_func = model.loss  # str

    # 対象となるレイヤの重みを取得
    for idx_to_tl, vs in target_weights.items():
        logger.info(f"idx_to_tl = {idx_to_tl}")
        t_w, lname = vs

        ############ FI begin #########
        # 対象レイヤが0, 1層目の場合
        if idx_to_tl == 0 or idx_to_tl == 1:
            prev_output = target_X
        # 対象レイヤが2層目以降の場合
        else:
            # 対象レイヤまでの中間層を出力するモデルを作成して出力を得る
            t_model = Model(inputs=model.input, outputs=model.layers[idx_to_tl - 1].output)
            prev_output = t_model.predict(target_X, verbose=-1)
        layer_config = model.layers[idx_to_tl].get_config()
        logger.info(f"t_w.shape = {t_w.shape}")
        logger.info(f"prev_output.shape = {prev_output.shape}")

        from_front = []
        for idx in range(t_w.shape[-1]):
            assert int(prev_output.shape[-1]) == t_w.shape[0], "{} vs {}".format(
                int(prev_output.shape[-1]), t_w.shape[0]
            )
            output = np.multiply(prev_output, t_w[:, idx])  # -> shape = prev_output.shape
            output = np.abs(output)
            output = norm_scaler.fit_transform(output)
            output = np.mean(output, axis=0)
            from_front.append(output)

        # from_frontは前のレイヤとの影響度
        from_front = np.asarray(from_front)
        from_front = from_front.T  # ow/sum(ow)の計算
        logger.info(f"from_front.shape = {from_front.shape}")  # t_w (target weight)と同じshapeになる

        # from_behindは後続のレイヤに関する勾配
        from_behind = compute_gradient_to_output(model, idx_to_tl, target_X)  # \partial O / \partial oの計算
        logger.info(f"from_behind.shape = {from_behind.shape}")
        # Forward Impactの計算
        FIs = from_front * from_behind
        logger.info(f"FIs.shape = {FIs.shape}")
        ############ FI end #########

        ############ GL begin #######
        grad_scndcr = compute_gradient_to_loss(model, idx_to_tl, target_X, target_y, loss_func_name=loss_func)
        logger.info(f"grad_scndcr.shape: {grad_scndcr.shape}")
        ############ GL end #########

        # 2種類のコストのペア
        # NOTE: (N, 2), N=target layerの前のレイヤと繋がってる重みの数 (前レイヤのニューロン数 * targetレイヤのニューロン数)
        pairs = np.asarray([grad_scndcr.flatten(), FIs.flatten()]).T
        total_cands[idx_to_tl] = {"shape": FIs.shape, "costs": pairs}

        # TODO: 以下はLSTMの場合の処理？============================================
        # total_cands[idx_to_tl] = {"shape": [], "costs": []}
        # pairs = []
        # for _FIs, _grad_scndcr in zip(FIs, grad_scndcr):
        #     pairs = np.asarray([_grad_scndcr.flatten(), _FIs.flatten()]).T
        #     total_cands[idx_to_tl]["shape"].append(_FIs.shape)
        #     total_cands[idx_to_tl]["costs"].append(pairs)
        # ここまで==================================================================

    return total_cands


def run_arachne_localize(X, y, indices_to_wrong, indices_to_correct, target_weights, model):
    # compute FI and GL with changed inputs
    logger.info("compute FI and GL with wrong inputs")
    total_cands_wrong = compute_FI_and_GL(X, y, indices_to_wrong, target_weights, model)
    # compute FI and GL with unchanged inputs
    logger.info("compute FI and GL with correct inputs")
    total_cands_correct = compute_FI_and_GL(X, y, indices_to_correct, target_weights, model)

    indices_to_tl = list(total_cands_wrong.keys())
    costs_and_keys = []
    indices_to_nodes = []
    shapes = {}

    for idx_to_tl in indices_to_tl:
        logger.info(f"idx_to_tl = {idx_to_tl}")
        cost_from_wrong = total_cands_wrong[idx_to_tl]["costs"]
        cost_from_correct = total_cands_correct[idx_to_tl]["costs"]
        ## key: more influential to changed behaviour and less influential to unchanged behaviour
        costs_combined = cost_from_wrong / (1.0 + cost_from_correct)  # shape = (N,2)
        logger.info(f"costs_combined.shape = {costs_combined.shape}")
        shapes[idx_to_tl] = total_cands_wrong[idx_to_tl]["shape"]

        for i, c in enumerate(costs_combined):
            costs_and_keys.append(([idx_to_tl, i], c))  # ([index of target layer, index of neuron], cost)
            # NOTE: np.unravel_index(i, shapes[idx_to_tl])で前層の何番目のニューロンからtarget層の何番目のニューロンという重みを特定できる
            indices_to_nodes.append([idx_to_tl, np.unravel_index(i, shapes[idx_to_tl])])
        # print(len(costs_combined)) # targetレイヤとその前層の重みの合計数に等しい

    # costだけをリストにする
    costs = np.asarray([vs[1] for vs in costs_and_keys])  # len(costs)はモデルの重みの合計数に等しい（レイヤによらず）
    logger.info(f"costs.shape = {costs.shape}")
    _costs = costs.copy()
    is_efficient = np.arange(costs.shape[0])

    # 多分ここがパレートフロントを求める処理
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(_costs):
        nondominated_point_mask = np.any(_costs > _costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        # maskがTrueのものだけ残し，Falseのものは除外される
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        _costs = _costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    logger.info(f"_costs.shape = {_costs.shape}")
    logger.info(f"is_efficient.shape = {is_efficient}")

    pareto_front = [tuple(v) for v in np.asarray(indices_to_nodes, dtype=object)[is_efficient]]
    return pareto_front, costs_and_keys


if __name__ == "__main__":
    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(sys.argv[1])
    arachne_dir = exp_dir.replace("care", "arachne")
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

    # localizationの結果保存用のディレクトリ
    loc_save_dir = os.path.join(model_dir, "arachne-localization")

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

        # 対象となる重みを取得
        target_weight = get_target_weights(model)
        # just for checking
        for i, v in target_weight.items():
            w, name = v
            logger.info(f"layer: {i}, weight.shape: {w.shape}, name: {name}")
        # checking end

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
        for _, v in misclf_dic.items():
            s += len(v)
        # sとlen(indices['wrong'])は一致するはず.
        assert s == len(indices["wrong"])
        # checking end

        # repairのために使うデータのインデックス
        indices_for_repair = target_indices_wrong + list(indices_to_correct)
        X_for_repair, y_for_repair = X_repair[indices_for_repair], y_repair[indices_for_repair]

        # target_indicesと同じ数のデータを，正しい予測のデータからランダムにサンプリングする
        # この際, 正しいデータからのサンプリングは各ラベルから均等になるようにサンプリングする
        uniq_labels = np.unique(y_repair[indices["correct"]])

        # ラベルごとのインデックスを格納する辞書
        grouped_by_label = {uniq_label: [] for uniq_label in uniq_labels}
        for idx in indices["correct"]:
            pred_label = pred_labels[idx]
            assert pred_label == y_repair[idx]  # 正解のはずなので
            grouped_by_label[pred_label].append(idx)

        # 以降は乱数が入るので繰り返す
        for rep in range(num_reps):
            logger.info(f"starting rep {rep}...")
            # 保存用のディレクトリ作成
            save_dir = os.path.join(loc_save_dir, f"rep{rep}")
            os.makedirs(save_dir, exist_ok=True)
            used_data_save_dir = os.path.join(arachne_dir, "used_data", task_name, f"rep{rep}")
            os.makedirs(used_data_save_dir, exist_ok=True)

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
            indices_for_loc = target_indices_wrong + sampled_indices_correct
            # インデックスからデータを取得
            X_for_loc, y_for_loc = X_repair[indices_for_loc], y_repair[indices_for_loc]
            logger.info(f"X_for_loc.shape = {X_for_loc.shape}, y_for_loc.shape = {y_for_loc.shape}")
            # X_for_locやy_for_locのためにインデックスを振り直し
            num_wrong = len(target_indices_wrong)
            num_loc_target = len(indices_for_loc)
            target_indices_wrong = list(range(0, num_wrong))
            sampled_indices_correct = list(range(num_wrong, num_loc_target))

            # そのexp_setting, fold, repで使われる, localization用とrepair用のデータを保存
            used_data_save_path = os.path.join(used_data_save_dir, f"X-y_for_loc-repair_fold-{k}.npz")
            np.savez(
                used_data_save_path,
                X_for_loc=X_for_loc,
                y_for_loc=y_for_loc,
                X_for_repair=X_for_repair,
                y_for_repair=y_for_repair,
            )
            logger.info(f"saved to {used_data_save_path}")

            # localizationを実行
            indices_to_places_to_fix, front_lst = run_arachne_localize(
                X_for_loc, y_for_loc, target_indices_wrong, sampled_indices_correct, target_weight, model
            )
            logger.info(f"Places to fix {indices_to_places_to_fix}")

            # 修正すべき位置の保存用のdfを作成
            output_df = pd.DataFrame(
                {
                    "layer": [vs[0] for vs in indices_to_places_to_fix],
                    "weight": [vs[1] for vs in indices_to_places_to_fix],
                }
            )
            save_path = os.path.join(save_dir, f"place_to_fix_fold-{k}.csv")
            output_df.to_csv(save_path, index=False)
            logger.info(f"saved to {save_path}")

            # TODO: front_lstの保存. 必要?今のところ感じてないのでコメントアウト
            # import pickle
            # with open(os.path.join(save_dir, f"cost_lst_fold-{k}.pkl"), "wb") as f:
            #     pickle.dump(front_lst, f)
