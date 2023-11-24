import os, sys, re, time
from collections import defaultdict
from collections.abc import Iterable
from tqdm import tqdm

import numpy as np
import pandas as pd
from lib.model import get_misclassified_index, sort_keys_by_cnt

# import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow import keras
import keras.backend as K


from keras.models import load_model, Model
from keras.layers import Input
from sklearn.preprocessing import Normalizer
from lib.util import json2dict, dataset_type, fix_dataloader
from lib.log import set_exp_logging
from lib.dataset import pad_collate
from src_arachne.utils import model_util as model_utils
from src_arachne.utils import lstm_layer
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
        lname = layer.__class__.__name__
        if is_target(lname, targeting_clsname_pattns):
            # NOTE: Denseの場合は重みとバイアスのペアで返ってくる
            weights = layer.get_weights()
            if len(weights):  # has weights
                if model_utils.is_FC(lname) or model_utils.is_C2D(lname):
                    target_weights[i] = [weights[0], lname]
                elif model_utils.is_LSTM(lname):
                    # NOTE: biasがTrueの場合は元の実装通りweights[:-1]にしないといけないけどFalseにしてるので無視
                    target_weights[i] = [weights, lname]
    return target_weights

def compute_output_per_w(x, h, t_w_kernel, t_w_recurr_kernel, const, with_norm = False): 
    """
    A slice for a single neuron (unit or lstm cell)
    x = (batch_size, time_steps, num_features)
    h = (batch_size, time_steps, num_units)
    t_w_kernel = (num_features,)
    t_w_recurr_kernel = (num_units,)
    consts = (batch_size, time_steps) -> the value that is multiplied in the final state computation
    Return the product of the multiplication of weights and input for each unit (i.e., each LSTM cell)
    -> meaning the direct front impact computed per neural weights 
    """
    from sklearn.preprocessing import Normalizer
    norm_scaler = Normalizer(norm = "l1")

    # here, "multiply" instead of dot, b/c we want to get the output of each neural weight (not the final one)
    out_kernel = x * t_w_kernel #np.multiply(x, t_w_kernel) # shape = (batch_size, time_steps, num_features)
    if h is not None: # None -> we will skip the weights for hiddens states
        out_recurr_kernel = h * t_w_recurr_kernel # shape = (batch_size, time_steps, num_units)
        out = np.append(out_kernel, out_recurr_kernel, axis = -1) # shape:(batch_size,time_steps,(num_features+num_units))
    else:
        out = out_kernel # shape = (batch_size, time_steps, num_features)

    # normalise
    out = np.abs(out)
    if with_norm: 
        original_shape = out.shape
        out = norm_scaler.fit_transform(out.flatten().reshape(1,-1)).reshape(-1,)
        out = out.reshape(original_shape)

    # N = num_features or num_features + num_units
    out = np.einsum('ijk,ij->ijk', out, const) # shape = (batch_size, time_steps, N) 
    return out


def get_constants(gate, F, I, C, O, cell_states):
    """
    """
    if gate == 'input':
        return np.multiply(O, np.divide(
            C, cell_states[:,1:,:], 
            out = np.zeros_like(C), where = cell_states[:,1:,:] != 0))
    elif gate == 'forget':
        return np.multiply(O, np.divide(
            cell_states[:,:-1,:], cell_states[:,1:,:], 
            out = np.zeros_like(cell_states[:,:-1,:]), where = cell_states[:,1:,:] != 0))
    elif gate == 'cand':
        return np.multiply(O, np.divide(
            I, cell_states[:,1:,:], 
            out = np.zeros_like(C), where = cell_states[:,1:,:] != 0))
    else: # output
        return np.tanh(cell_states[:,1:,:])

def lstm_local_front_FI_for_target_all(
    x, h, num_units, 
    t_w_kernels, t_w_recurr_kernels, consts, 
    gate_orders = ['input', 'forget', 'cand', 'output']):
    """
    x = previous output
    h = hidden state (-> should be computed using the layer)
    t_w_kernels / t_w_recurr_kernels / consts: 
        a group of neural weights that should be taken into account when measuring the impact.
        arg consts is the corresponding group of constants that will be multiplied 
        to each nueral weight's output, respectively
    """
    from sklearn.preprocessing import Normalizer
    norm_scaler = Normalizer(norm = "l1")
    from_front = []
    for idx_to_unit in tqdm(range(num_units)):
        out_combined = None
        for gate in gate_orders:
            # out's shape, (batch_size, time_steps, (num_features + num_units))	
            # since the weights of each gate are added with the weights of the other gates, normalise later
            out = compute_output_per_w(
                x, h,
                t_w_kernels[gate][:,idx_to_unit],
                t_w_recurr_kernels[gate][:,idx_to_unit],
                consts[gate][...,idx_to_unit],
                with_norm = False)

            if out_combined is None:
                out_combined = out 
            else:
                out_combined = np.append(out_combined, out, axis = -1)

        # the shape of out_combined => 
        # 	(batch_size, time_steps, 4 * (num_features + num_units)) (since this is per unit)
        # here, keep in mind that we have to use a scaler on the current out_combined 
        # (for instance, divide by the final output (the last hidden state won't work here anymore, 
        # as the summation of the current value differs from the original due to 
        # the absence of act and the scaling in the middle, etc.)
        original_shape = out_combined.shape
        # normalised
        scaled_out_combined = norm_scaler.fit_transform(np.abs(out_combined).flatten().reshape(1,-1)) 
        scaled_out_combined = scaled_out_combined.reshape(original_shape) 
        # mean out_combined's shape: ((num_features + num_units) * 4,) 
        # for each neural weight, the average over both time step and the batch 
        avg_scaled_out_combined = np.mean(
            scaled_out_combined.reshape(-1, scaled_out_combined.shape[-1]), axis = 0) 
        from_front.append(avg_scaled_out_combined)

    # from_front's shape = (num_units, (num_features + num_units) * 4)
    from_front = np.asarray(from_front)
    print ("For lstm's front part of FI: {}".format(from_front.shape))
    return from_front, gate_orders



def compute_gradient_to_output(model, idx_to_target_layer, X, on_weight=False, wo_reset=False, by_batch=False, target_lens=None):
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
        lname = model.layers[idx_to_target_layer].__class__.__name__
        prev_lname = model.layers[idx_to_target_layer-1].__class__.__name__
        # 勾配の形状を決定
        # 前のレイヤがLSTMでも, 次のレイヤの形状に影響を与える（系列長の次元が増えてしまう）
        if not (model_utils.is_LSTM(lname) or model_utils.is_LSTM(prev_lname)):
            grad_shape = tuple([num] + [int(v) for v in target.shape[1:]])
        else:
            max_len = X.shape[1] # X内の最大系列長
            grad_shape = (num, max_len, target.shape[-1]) # (batch_size, num_units)
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
        if model_utils.is_LSTM(lname) or model_utils.is_LSTM(prev_lname):
            gradient = gradient[np.arange(gradient.shape[0]), target_lens - 1, :]
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


def compute_gradient_to_loss(model, idx_to_target_layer, X, y, loss_func_name, by_batch=False, target_lens=None, **kwargs):
    """
    compute gradients for the loss.
    kwargs contains the key-word argumenets required for the loss funation
    """
    lname = model.layers[idx_to_target_layer].__class__.__name__
    if not model_utils.is_LSTM(lname):
        targets = model.layers[idx_to_target_layer].weights[:-1]
    else:
        targets = model.layers[idx_to_target_layer].weights

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
    if loss_func_name == "categorical_crossentropy":
        loss_func = tf.nn.softmax_cross_entropy_with_logits_v2
    elif loss_func_name == "binary_crossentropy":
        if "name" in kwargs.keys():
            kwargs.pop("name")
        loss_func = tf.keras.losses.binary_crossentropy
    elif loss_func_name in ["mean_squared_error", "mse"]:
        loss_func = tf.keras.losses.MeanSquaredError
    else:
        print("{} not supported yet".format(loss_func_name))
        assert False
    # lossの計算でこのshapeにしないとエラーになるのでreshape
    y = y.reshape(-1, 1)

    gradients = [[] for _ in range(len(targets))]
    for chunk in chunks:
        # 自動微分のためにチャンクのデータをtf.Variableにする
        x = tf.Variable(X[chunk])
        y_true = tf.Variable(y[chunk])
        with tf.GradientTape() as tape:
            y_pred = model(x)
            if target_lens is None:
                loss = loss_func(y_true, y_pred)
            else:
                # 計算グラフを保持して勾配計算ができるようにしつつ可変長に対応できるようlossを計算
                losses = [loss_func(y_true[i], y_pred[i][target_lens[i]-1]) for i in range(y_pred.shape[0])]
                loss = tf.reduce_sum(losses)
        # モデル出力のロスの対象のレイヤの重みに関する勾配を取得
        _gradients = tape.gradient(loss, targets)
        for i, _gradient in enumerate(_gradients):
            gradients[i].append(_gradient)

    for i, gradients_p_chunk in enumerate(gradients):
        gradients[i] = np.abs(np.sum(np.asarray(gradients_p_chunk), axis=0))  # combine
        logger.info(f"gradients[{i}].shape={gradients[i].shape}")

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


def compute_FI_and_GL(X, y, indices_to_target, target_weights, model, lens=None):
    norm_scaler = Normalizer(norm="l1")
    total_cands = {}
    FIs, grad_scndcr = None, None
    target_X = X[indices_to_target]
    target_y = y[indices_to_target]
    target_lens = lens[indices_to_target] if lens is not None else None
    loss_func = model.loss  # str

    # 対象となるレイヤの重みを取得
    for idx_to_tl, vs in target_weights.items():
        logger.info(f"idx_to_tl = {idx_to_tl}")
        t_w, lname = vs
        layer_config = model.layers[idx_to_tl].get_config()
        if idx_to_tl == 0:
            # meaning the model doesn't specify the input layer explicitly
            prev_output = target_X
        else:
            prev_output = model.layers[idx_to_tl - 1].output

        # FC層の場合
        if model_utils.is_FC(lname):
            ############ FI begin #########
            # 対象レイヤが0, 1層目の場合
            if idx_to_tl == 0 or idx_to_tl == 1:
                prev_output = target_X
            # 対象レイヤが2層目以降の場合
            else:
                # 対象レイヤまでの中間層を出力するモデルを作成して出力を得る
                t_model = Model(inputs=model.input, outputs=model.layers[idx_to_tl - 1].output)
                prev_output = t_model.predict(target_X, verbose=-1)
                prev_lname = model.layers[idx_to_tl - 1].__class__.__name__
                if (model_utils.is_LSTM(prev_lname)) and (target_lens is not None):
                    prev_output = prev_output[np.arange(prev_output.shape[0]), target_lens - 1, :]
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
            from_behind = compute_gradient_to_output(model, idx_to_tl, target_X, target_lens=target_lens)  # \partial O / \partial oの計算
            logger.info(f"from_behind.shape = {from_behind.shape}")
            # Forward Impactの計算
            FIs = from_front * from_behind
            logger.info(f"FIs.shape = {FIs.shape}")
            ############ FI end #########

            ############ GL begin #######
            grad_scndcr = compute_gradient_to_loss(model, idx_to_tl, target_X, target_y, loss_func_name=loss_func, target_lens=target_lens)
            logger.info(f"grad_scndcr.shape: {grad_scndcr.shape}")
            ############ GL end #########

        elif model_utils.is_C2D(lname):
            ############ FI begin #########
            is_channel_first = layer_config["data_format"] == "channels_first"
            # 修正に使うデータを対象レイヤまで順伝搬した出力を得る
            if idx_to_tl == 0 or idx_to_tl - 1 == 0:
                prev_output_v = target_X
            else:
                t_model = Model(inputs=model.input, outputs=model.layers[idx_to_tl - 1].output)
                prev_output_v = t_model.predict(target_X)
            tr_prev_output_v = np.moveaxis(prev_output_v, [1, 2, 3], [3, 1, 2]) if is_channel_first else prev_output_v
            logger.info(f"t_w.shape = {t_w.shape}")
            logger.info(f"prev_output.shape = {prev_output.shape}")

            # Conv2Dのハイパラを取得
            kernel_shape = t_w.shape[:2]
            strides = layer_config["strides"]
            padding_type = layer_config["padding"]
            # paddingの具体的な値を取得
            if padding_type == "valid":
                paddings = [0, 0]
            else:
                if padding_type == "same":
                    # P = ((S-1)*W-S+F)/2
                    true_ws_shape = [
                        t_w.shape[0],
                        t_w.shape[-1],
                    ]  # Channel_in, Channel_out
                    paddings = [
                        int(((strides[i] - 1) * true_ws_shape[i] - strides[i] + kernel_shape[i]) / 2) for i in range(2)
                    ]
                elif not isinstance(padding_type, str) and isinstance(
                    padding_type, Iterable
                ):  # explicit paddings given
                    paddings = list(padding_type)
                    if len(paddings) == 1:
                        paddings = [paddings[0], paddings[0]]
                else:
                    print("padding type: {} not supported".format(padding_type))
                    paddings = [0, 0]
                    assert False

                # add padding
                if is_channel_first:
                    paddings_per_axis = [
                        [0, 0],
                        [0, 0],
                        [paddings[0], paddings[0]],
                        [paddings[1], paddings[1]],
                    ]
                else:
                    paddings_per_axis = [
                        [0, 0],
                        [paddings[0], paddings[0]],
                        [paddings[1], paddings[1]],
                        [0, 0],
                    ]

                tr_prev_output_v = np.pad(
                    tr_prev_output_v,
                    paddings_per_axis,
                    mode="constant",
                    constant_values=0,
                )  # zero-padding

            if is_channel_first:
                num_kernels = int(prev_output.shape[1])  # Channel_in
            else:  # channels_last
                assert layer_config["data_format"] == "channels_last", layer_config["data_format"]
                num_kernels = int(prev_output.shape[-1])  # Channel_in
            assert num_kernels == t_w.shape[2], "{} vs {}".format(num_kernels, t_w.shape[2])

            # H x W
            if is_channel_first:
                # the last two (front two are # of inputs and # of kernels (Channel_in))
                input_shape = [int(v) for v in prev_output.shape[2:]]
            else:
                input_shape = [int(v) for v in prev_output.shape[1:-1]]
            # (W1−F+2P)/S+1, W1 = input volumne , F = kernel, P = padding
            n_mv_0 = int((input_shape[0] - kernel_shape[0] + 2 * paddings[0]) / strides[0] + 1)  # H_out
            n_mv_1 = int((input_shape[1] - kernel_shape[1] + 2 * paddings[1]) / strides[1] + 1)  # W_out

            n_output_channel = t_w.shape[-1]  # Channel_out
            from_front = []
            # move axis for easier computation
            for idx_ol in tqdm(range(n_output_channel)):  # t_w.shape[-1]
                for i in range(n_mv_0):  # H
                    for j in range(n_mv_1):  # W
                        curr_prev_output_slice = tr_prev_output_v[
                            :, i * strides[0] : i * strides[0] + kernel_shape[0], :, :
                        ]
                        curr_prev_output_slice = curr_prev_output_slice[
                            :, :, j * strides[1] : j * strides[1] + kernel_shape[1], :
                        ]
                        output = curr_prev_output_slice * t_w[:, :, :, idx_ol]
                        sum_output = np.sum(np.abs(output))
                        output = output / sum_output
                        sum_output = np.nan_to_num(output, posinf=0.0)
                        output = np.mean(output, axis=0)
                        from_front.append(output)

            from_front = np.asarray(from_front)
            # from_front.shape: [Channel_out * n_mv_0 * n_mv_1, F1, F2, Channel_in]
            if is_channel_first:
                from_front = from_front.reshape(
                    (n_output_channel, n_mv_0, n_mv_1, kernel_shape[0], kernel_shape[1], int(prev_output.shape[1]))
                )
            else:  # channels_last
                from_front = from_front.reshape(
                    (n_mv_0, n_mv_1, n_output_channel, kernel_shape[0], kernel_shape[1], int(prev_output.shape[-1]))
                )

            # [F1,F2,Channel_in, Channel_out, n_mv_0, n_mv_1]
            # 	or [F1,F2,Channel_in, n_mv_0, n_mv_1,Channel_out]
            from_front = np.moveaxis(from_front, [0, 1, 2], [3, 4, 5])
            logger.info(f"from_front.shape = {from_front.shape}")
            # [Channel_out, H_out(n_mv_0), W_out(n_mv_1)]
            from_behind = compute_gradient_to_output(model, idx_to_tl, target_X, by_batch=True)
            logger.info(f"from_behind.shape = {from_behind.shape}")
            # [F1,F2,Channel_in, Channel_out, n_mv_0, n_mv_1] (channels_firs)
            # or [F1,F2,Channel_in,n_mv_0, n_mv_1,Channel_out] (channels_last)
            FIs = from_front * from_behind
            if is_channel_first:
                FIs = np.sum(np.sum(FIs, axis=-1), axis=-1)  # [F1, F2, Channel_in, Channel_out]
            else:
                FIs = np.sum(np.sum(FIs, axis=-2), axis=-2)  # [F1, F2, Channel_in, Channel_out]
            logger.info(f"FIs.shape = {FIs.shape}")
            ############ FI end #########

            ############ GL begin #######
            grad_scndcr = compute_gradient_to_loss(model, idx_to_tl, target_X, target_y, loss_func_name=loss_func)
            logger.info(f"grad_scndcr.shape: {grad_scndcr.shape}")
            ############ GL end #########
        elif model_utils.is_LSTM(lname):
            from scipy.special import expit as sigmoid

            t_w_kernel, t_w_recurr_kernel = t_w
            logger.info(f"t_w_kernel.shape={t_w_kernel.shape}, t_w_recurr_kernel.shape={t_w_recurr_kernel.shape}")
            if model_utils.is_Input(type(model.layers[idx_to_tl - 1]).__name__):
                prev_output = target_X
            else:
                # shape = (batch_size, time_steps, input_feature_size)
                t_model = Model(inputs = model.input, outputs = model.layers[idx_to_tl - 1].output) 
                prev_output = t_model.predict(target_X)
            assert len(prev_output.shape) == 3, prev_output.shape
            num_features = prev_output.shape[-1] # the dimension of features that will be processed by the model
            
            num_units = t_w_recurr_kernel.shape[0] 
            assert t_w_kernel.shape[0] == num_features, "{} (kernel) vs {} (input)".format(t_w_kernel.shape[0], num_features)

            # hidden state and cell state sequences computation
            # generate a temporary model that only contains the target lstm layer 
            # but with the modification to return sequences of hidden and cell states
            temp_lstm_layer_inst = lstm_layer.LSTM_Layer(model.layers[idx_to_tl])
            hstates_sequence, cell_states_sequence = temp_lstm_layer_inst.gen_lstm_layer_from_another(prev_output)
            init_hstates, init_cell_states = lstm_layer.LSTM_Layer.get_initial_state(model.layers[idx_to_tl])
            if init_hstates is None: 
                init_hstates = np.zeros((len(target_X), num_units)) 
            if init_cell_states is None:
                # shape = (batch_size, num_units)
                init_cell_states = np.zeros((len(target_X), num_units)) 
        
            # shape = (batch_size, time_steps + 1, num_units)
            hstates_sequence = np.insert(hstates_sequence, 0, init_hstates, axis = 1)
                # shape = (batch_size, time_steps + 1, num_units)
            cell_states_sequence = np.insert(cell_states_sequence, 0, init_cell_states, axis = 1)
            # bias = model.layers[idx_to_tl].get_weights()[-1] # shape = (4 * num_units,)
            indices_to_each_gates = np.array_split(np.arange(num_units * 4), 4)

            ## prepare all the intermediate outputs and the variables that will be used later
            idx_to_input_gate = 0
            idx_to_forget_gate = 1
            idx_to_cand_gate = 2
            idx_to_output_gate = 3

            # input 
            t_w_kernel_I = t_w_kernel[:, indices_to_each_gates[idx_to_input_gate]] 
            t_w_recurr_kernel_I = t_w_recurr_kernel[:, indices_to_each_gates[idx_to_input_gate]] 
            # bias_I = bias[indices_to_each_gates[idx_to_input_gate]] # NOTE: biasが必要なときは下の行にbiasを足すのを忘れるな
            I = sigmoid(np.dot(prev_output, t_w_kernel_I) + np.dot(hstates_sequence[:,:-1,:], t_w_recurr_kernel_I))

            # forget
            t_w_kernel_F = t_w_kernel[:, indices_to_each_gates[idx_to_forget_gate]]
            t_w_recurr_kernel_F = t_w_recurr_kernel[:, indices_to_each_gates[idx_to_forget_gate]]
            # bias_F = bias[indices_to_each_gates[idx_to_forget_gate]]
            F = sigmoid(np.dot(prev_output, t_w_kernel_F) + np.dot(hstates_sequence[:,:-1,:], t_w_recurr_kernel_F))

            # cand
            t_w_kernel_C = t_w_kernel[:, indices_to_each_gates[idx_to_cand_gate]]
            t_w_recurr_kernel_C = t_w_recurr_kernel[:, indices_to_each_gates[idx_to_cand_gate]]
            # bias_C = bias[indices_to_each_gates[idx_to_cand_gate]]
            C = np.tanh(np.dot(prev_output, t_w_kernel_C) + np.dot(hstates_sequence[:,:-1,:], t_w_recurr_kernel_C))

            # output
            t_w_kernel_O = t_w_kernel[:, indices_to_each_gates[idx_to_output_gate]]
            t_w_recurr_kernel_O = t_w_recurr_kernel[:, indices_to_each_gates[idx_to_output_gate]]
            # bias_O = bias[indices_to_each_gates[idx_to_output_gate]]
            # shape = (batch_size, time_steps, num_units)
            O = sigmoid(np.dot(prev_output, t_w_kernel_O) + np.dot(hstates_sequence[:,:-1,:], t_w_recurr_kernel_O))

            # set arguments to compute forward impact for the neural weights from these four gates
            t_w_kernels = {
                'input':t_w_kernel_I, 'forget':t_w_kernel_F, 
                'cand':t_w_kernel_C, 'output':t_w_kernel_O}
            t_w_recurr_kernels = {
                'input':t_w_recurr_kernel_I, 'forget':t_w_recurr_kernel_F, 
                'cand':t_w_recurr_kernel_C, 'output':t_w_recurr_kernel_O}

            consts = {}
            consts['input'] = get_constants('input', F, I, C, O, cell_states_sequence)
            consts['forget'] = get_constants('forget', F, I, C, O, cell_states_sequence)
            consts['cand'] = get_constants('cand', F, I, C, O, cell_states_sequence)
            consts['output'] = get_constants('output', F, I, C, O, cell_states_sequence)

            # from_front's shape = (num_units, (num_features + num_units) * 4)
            # gate_orders = ['input', 'forget', 'cand', 'output']
            from_front, gate_orders  = lstm_local_front_FI_for_target_all(
                prev_output, hstates_sequence[:,:-1,:], num_units, 
                t_w_kernels, t_w_recurr_kernels, consts)

            from_front = from_front.T # ((num_features + num_units) * 4, num_units)
            N_k_rk_w = int(from_front.shape[0]/4)
            assert N_k_rk_w == num_features + num_units, "{} vs {}".format(N_k_rk_w, num_features + num_units)
            logger.info(f"from_front.shape={from_front.shape}")
            
            ## from behind
            from_behind = compute_gradient_to_output(model, idx_to_tl, target_X, by_batch = True, target_lens=target_lens) # shape = (num_units,)
            logger.info(f"from_behind.shape={from_behind.shape}")

            #t1 = time.time()
            # shape = (N_k_rk_w, num_units) 
            FIs_combined = from_front * from_behind
            #print ("Shape", from_behind.shape, FIs_combined.shape)
            #t2 = time.time()
            #print ('Time for multiplying front and behind results: {}'.format(t2 - t1))
            
            # reshaping
            FIs_kernel = np.zeros(t_w_kernel.shape) # t_w_kernel's shape (num_features, num_units * 4)
            FIs_recurr_kernel = np.zeros(t_w_recurr_kernel.shape) # t_w_recurr_kernel's shape (num_units, num_units * 4)
            # from (4 * N_k_rk_w, num_units) to 4 * (N_k_rk_w, num_units)
            for i, FI_p_gate in enumerate(np.array_split(FIs_combined, 4, axis = 0)): 
                # FI_p_gate's shape = (N_k_rk_w, num_units) 
                # 	-> will divided into (num_features, num_units) & (num_units, num_units)
                # local indices that will split FI_p_gate (shape = (N_k_rk_w, num_units))
                # since we append the weights in order of a kernel weight and a recurrent kernel weight
                indices_to_features = np.arange(num_features)
                indices_to_units = np.arange(num_units) + num_features
                #FIs_kernel[indices_to_features + (i * N_k_rk_w)] 
                # = FI_p_gate[indices_to_features] # shape = (num_features, num_units)
                #FIs_recurr_kernel[indices_to_units + (i * N_k_rk_w)] 
                # = FI_p_gate[indices_to_units] # shape = (num_units, num_units)
                FIs_kernel[:, i * num_units:(i+1) * num_units] = FI_p_gate[indices_to_features] # shape = (num_features, num_units)
                FIs_recurr_kernel[:, i * num_units:(i+1) * num_units] = FI_p_gate[indices_to_units] # shape = (num_units, num_units)

            #t3 =time.time()
            FIs = [FIs_kernel, FIs_recurr_kernel] # [(num_features, num_units*4), (num_units, num_units*4)]
            logger.info(f"FIs_kernel.shape={FIs_kernel.shape}, FIs_recurr_kernel.shape={FIs_recurr_kernel.shape}")
            #print ('Time for formatting: {}'.format(t3 - t2))
            
            ## Gradient
            grad_scndcr = compute_gradient_to_loss(model, idx_to_tl, target_X, target_y, by_batch=False, loss_func_name = loss_func, target_lens=target_lens)

        # 2種類のコストのペア
        if not model_utils.is_LSTM(lname):
            # NOTE: (N, 2), N=target layerの前のレイヤと繋がってる重みの数 (前レイヤのニューロン数 * targetレイヤのニューロン数)
            pairs = np.asarray([grad_scndcr.flatten(), FIs.flatten()]).T
            total_cands[idx_to_tl] = {"shape": FIs.shape, "costs": pairs}
        else:
            total_cands[idx_to_tl] = {"shape": [], "costs": []}
            pairs = []
            for _FIs, _grad_scndcr in zip(FIs, grad_scndcr):
                pairs = np.asarray([_grad_scndcr.flatten(), _FIs.flatten()]).T
                total_cands[idx_to_tl]["shape"].append(_FIs.shape)
                total_cands[idx_to_tl]["costs"].append(pairs)

    return total_cands


def run_arachne_localize(X, y, indices_to_wrong, indices_to_correct, target_weights, model, len_for_loc=None):
    # compute FI and GL with changed inputs
    logger.info("compute FI and GL with wrong inputs")
    total_cands_wrong = compute_FI_and_GL(X, y, indices_to_wrong, target_weights, model, len_for_loc)
    # compute FI and GL with unchanged inputs
    logger.info("compute FI and GL with correct inputs")
    total_cands_correct = compute_FI_and_GL(X, y, indices_to_correct, target_weights, model, len_for_loc)

    indices_to_tl = list(total_cands_wrong.keys())
    costs_and_keys = []
    indices_to_nodes = []
    shapes = {}

    for idx_to_tl in indices_to_tl:
        if not model_utils.is_LSTM(target_weights[idx_to_tl][1]):
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
        else:
            num = len(total_cands_wrong[idx_to_tl]['shape'])
            shapes[idx_to_tl] = []
            for idx_to_pair in range(num):
                cost_from_wrong = total_cands_wrong[idx_to_tl]['costs'][idx_to_pair]
                cost_from_correct = total_cands_correct[idx_to_tl]['costs'][idx_to_pair]
                costs_combined = cost_from_wrong/(1. + cost_from_correct) # shape = (N,2)
                logger.info(f"costs_combined.shape = {costs_combined.shape}")
                shapes[idx_to_tl].append(total_cands_wrong[idx_to_tl]['shape'][idx_to_pair])

                for i,c in enumerate(costs_combined):
                    costs_and_keys.append(([(idx_to_tl, idx_to_pair), i], c))
                    indices_to_nodes.append(
                        [(idx_to_tl, idx_to_pair), np.unravel_index(i, shapes[idx_to_tl][idx_to_pair])])

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
    collate_fn = None if dataset_type(task_name) != "text" else pad_collate
    # target_column = setting_dict["TARGET_COLUMN"]
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
        fixed_repair_loader = fix_dataloader(repair_loader, collate_fn=collate_fn)  # 順番fix ver.
        repair_ds = repair_loader.dataset  # HACK: これはメモリ食うのでなんとかした方がいいかも. 画像データでも問題なければそのままで.
        # 学習済みモデルをロード (keras)
        model = load_model(os.path.join(model_dir, f"keras_model_fold-{k}.h5"))
        # tabular dataset
        if dataset_type(task_name) == "tabular":
            X_repair, y_repair = (
                repair_ds.tensors[0].detach().numpy().copy(),
                repair_ds.tensors[1].detach().numpy().copy(),
            )
            logger.info(f"X_repair.shape = {X_repair.shape}, y_repair.shape = {y_repair.shape}")
            loss_fn = "binary_crossentropy"
            model.compile(loss=loss_fn, optimizer="adam", metrics=["accuracy"])  # これがないと予測できない(エラーになる)
            # 予測のスコアとそこから予測ラベル取得
            pred_scores = model.predict(X_repair, verbose=-1)
            pred_labels = np.argmax(pred_scores, axis=1)
            correct_predictions = pred_labels == y_repair

        # image dataset
        elif dataset_type(task_name) == "image":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            loss_fn = "categorical_crossentropy"
            model.compile(loss=loss_fn, optimizer="adam", metrics=["accuracy"])  # これがないと予測できない(エラーになる)
            # repair_loaderを使ってバッチで予測
            pred_labels = []  # 予測ラベルの配列
            correct_predictions = []  # あってたかどうかの配列
            X_repair, y_repair = [], []
            for batch_idx, (data, labels) in enumerate(fixed_repair_loader):
                data, labels = data.to(device), labels.to(device)
                # data, labelsをnumpyに変換
                data, labels = (
                    data.detach().cpu().numpy().copy(),
                    labels.detach().cpu().numpy().copy(),
                )
                # dataをchannel_lastに変換
                data = np.transpose(data, (0, 2, 3, 1))
                # keras modelの予測を実行
                pred_scores = model.predict(data, verbose=1)  # NOTE: これはkeras.modelのpredictメソッドね
                pred_labels_tmp = np.argmax(pred_scores, axis=1)
                # 全体の予測ラベルの配列に追加
                pred_labels.extend(pred_labels_tmp)
                # repair_loaderの各バッチに対する予測の正解(1), 不正解(0)の配列
                correctness_tmp = pred_labels_tmp == labels
                # 全体の正解配列に追加
                correct_predictions.extend(correctness_tmp)
                # data, labelを追加
                X_repair.extend(data)
                y_repair.extend(labels)
            correct_predictions = np.array(correct_predictions)
            X_repair = np.array(X_repair)
            y_repair = np.array(y_repair)
            # print(sum(correct_predictions), len(correct_predictions))

        elif dataset_type(task_name) == "text":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            loss_fn = "binary_crossentropy"
            model.compile(loss=loss_fn, optimizer="adam", metrics=["accuracy"])  # これがないと予測できない(エラーになる)
            # repair_loaderを使ってバッチで予測
            pred_labels = []  # 予測ラベルの配列
            correct_predictions = []  # あってたかどうかの配列
            X_repair, y_repair, len_repair = [], [], []
            max_len = 0 # X_repair内の最大系列長
            for data, _, _ in fixed_repair_loader:
                max_len = max(max_len, data.shape[1])
            fixed_repair_loader = fix_dataloader(repair_loader, collate_fn=collate_fn)
            for batch_idx, (data, labels, data_lens) in enumerate(fixed_repair_loader):
                data, labels = data.to(device), labels.to(device)
                # data, labelsをnumpyに変換
                data, labels = (
                    data.detach().cpu().numpy().copy(),
                    labels.detach().cpu().numpy().copy(),
                )
                pred_scores_seqs = model.predict(data, verbose=1)
                pred_scores = pred_scores_seqs[np.arange(pred_scores_seqs.shape[0]), np.array(data_lens) - 1, :]
                pred_labels_tmp = np.argmax(pred_scores, axis=1)
                # 全体の予測ラベルの配列に追加
                pred_labels.extend(pred_labels_tmp)
                # repair_loaderの各バッチに対する予測の正解(1), 不正解(0)の配列
                correctness_tmp = pred_labels_tmp == labels
                # 全体の正解配列に追加
                correct_predictions.extend(correctness_tmp)
                # data, labelを追加
                # dataに関してはmax_lenの長さにpaddingし直す
                data = np.pad(data, ((0, 0), (0, max_len - data.shape[1]), (0, 0)), mode="constant", constant_values=0)
                X_repair.extend(data)
                y_repair.extend(labels)
                len_repair.extend(data_lens)
            correct_predictions = np.array(correct_predictions)
            X_repair = np.array(X_repair)
            y_repair = np.array(y_repair)
            len_repair = np.array(len_repair)
            print(f"{sum(correct_predictions)} / {len(correct_predictions)} = {sum(correct_predictions) / len(correct_predictions)}")

        # 対象となる重みを取得
        target_weight = get_target_weights(model)
        # just for checking
        for i, v in target_weight.items():
            w, name = v
            if not isinstance(w, list):
                logger.info(f"layer: {i}, weight.shape: {w.shape}, name: {name}")
            else:
                for _w in w:
                    logger.info(f"layer: {i}, weight.shape: {_w.shape}, name: {name}")
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
        if dataset_type(task_name) != "text":
            X_for_repair, y_for_repair = (
                X_repair[indices_for_repair],
                y_repair[indices_for_repair],
            )
        else:
            X_for_repair, y_for_repair, len_for_repair = (
                X_repair[indices_for_repair],
                y_repair[indices_for_repair],
                len_repair[indices_for_repair],
            )
        logger.info(f"X_for_repair.shape = {X_for_repair.shape}, y_for_repair.shape = {y_for_repair.shape}")

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
            if dataset_type(task_name) != "text":
                X_for_loc, y_for_loc = X_repair[indices_for_loc], y_repair[indices_for_loc]
            else:
                X_for_loc, y_for_loc, len_for_loc = X_repair[indices_for_loc], y_repair[indices_for_loc], len_repair[indices_for_loc]
            logger.info(f"X_for_loc.shape = {X_for_loc.shape}, y_for_loc.shape = {y_for_loc.shape}")
            # X_for_locやy_for_locのためにインデックスを振り直し
            num_wrong = len(target_indices_wrong)
            num_loc_target = len(indices_for_loc)
            target_indices_wrong = list(range(0, num_wrong))
            sampled_indices_correct = list(range(num_wrong, num_loc_target))

            # そのexp_setting, fold, repで使われる, localization用とrepair用のデータを保存
            used_data_save_path = os.path.join(used_data_save_dir, f"X-y_for_loc-repair_fold-{k}.npz")
            if dataset_type(task_name) != "text":
                np.savez(
                    used_data_save_path,
                    X_for_loc=X_for_loc,
                    y_for_loc=y_for_loc,
                    X_for_repair=X_for_repair,
                    y_for_repair=y_for_repair,
                )
            else:
                np.savez(
                    used_data_save_path,
                    X_for_loc=X_for_loc,
                    y_for_loc=y_for_loc,
                    len_for_loc=len_for_loc,
                    X_for_repair=X_for_repair,
                    y_for_repair=y_for_repair,
                    len_for_repair=len_for_repair,
                )
            logger.info(f"saved to {used_data_save_path}")

            # =====================================================================
            # NOTE: ここまでは画像データでもエラーなく行けたっぽいので, localizeの本体の実装に集中.
            # =====================================================================

            # 開始時間計測
            s = time.clock()
            # logger.info(f"Start time: {s}")
            # localizationを実行
            _len_for_loc = None if dataset_type(task_name) != "text" else len_for_loc
            indices_to_places_to_fix, front_lst = run_arachne_localize(
                X_for_loc,
                y_for_loc,
                target_indices_wrong,
                sampled_indices_correct,
                target_weight,
                model,
                _len_for_loc
            )
            logger.info(f"Places to fix {indices_to_places_to_fix}")
            # 終了時間計測
            e = time.clock()
            # logger.info(f"End time: {e}")
            logger.info(f"Total execution time: {e-s} sec.")

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
