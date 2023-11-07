import keras as K
import torch
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    ReLU,
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    Permute,
    BatchNormalization,
)
from lib.model import select_model
from lib.log import set_exp_logging
import warnings

warnings.filterwarnings("ignore")


# tabular datasetのkerasによるモデルクラス
def TabularModelKeras(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation="relu")(inputs)
    x = ReLU()(x)
    x = Dense(32, activation="relu")(x)
    x = ReLU()(x)
    x = Dense(16, activation="relu")(x)
    x = ReLU()(x)
    x = Dense(8, activation="relu")(x)
    x = ReLU()(x)
    x = Dense(4, activation="relu")(x)
    x = ReLU()(x)
    outputs = Dense(2, activation="linear")(x)

    keras_model = Model(inputs=inputs, outputs=outputs)
    return keras_model


# 画像データセットに対してはデータセットごとにモデルが違うので個別に定義
# fmのkerasによるモデルクラス
def FashionModelKeras(input_dim):
    _input_dim = (input_dim[1], input_dim[2], input_dim[0])  # channel first -> channel last
    inputs = Input(shape=_input_dim)
    x = Conv2D(32, (5, 5), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(2)(x)
    x = Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = MaxPooling2D(2)(x)
    x = Permute((3, 1, 2))(x)
    # ↑flattenでtorch側と不整合がでないように一旦channel firstにする. NOTE: indexは1からスタートなので注意. バッチ次元は含まれず, HWCが123に対応する.これをCHWにするので312.
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.4)(x)  # 推論にしか使わないのでdropoutいらない
    x = Dense(10)(x)
    return Model(inputs=inputs, outputs=x)


# gtsrbのkerasによるモデルクラス
def GTSRBModelKeras(input_dim):
    _input_dim = (input_dim[1], input_dim[2], input_dim[0])  # channel first -> channel last
    inputs = Input(shape=_input_dim)
    x = Conv2D(100, (3, 3), padding="valid", activation="relu")(inputs)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)  # NOTE: このハイパラ設定はtorch側のデフォルト値に合わせるためのもの
    x = MaxPooling2D(2)(x)
    x = Conv2D(150, (4, 4), padding="valid", activation="relu")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = MaxPooling2D(2)(x)
    x = Conv2D(250, (3, 3), padding="valid", activation="relu")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = MaxPooling2D(2)(x)
    x = Permute((3, 1, 2))(x)
    # ↑flattenでtorch側と不整合がでないように一旦channel firstにする. NOTE: indexは1からスタートなので注意. バッチ次元は含まれず, HWCが123に対応する.これをCHWにするので312.
    x = Flatten()(x)
    x = Dense(200, activation="relu")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Dense(43)(x)
    return Model(inputs=inputs, outputs=x)


# c10のkerasによるモデルクラス
def C10ModelKeras(input_dim):
    _input_dim = (input_dim[1], input_dim[2], input_dim[0])  # channel first -> channel last
    inputs = Input(shape=_input_dim)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(2)(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(2)(x)
    x = Permute((3, 1, 2))(x)  # flattenのtorch.tensor.viewとの整合性のため
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(10)(x)
    return Model(inputs=inputs, outputs=x)


if __name__ == "__main__":
    # このプログラムのファイル名を取得
    file_name = os.path.splitext(sys.argv[0])[0]
    dataset_type = sys.argv[1]
    # 実験のディレクトリと実験名を取得
    exp_dir = "/src/experiments/arachne"
    # ログファイルの生成
    logger = set_exp_logging(exp_dir, file_name)

    # tabular datasets
    dic_tabular = {
        "credit": {
            "num_folds": 5,
            "model_path_format": "../models/credit/credit-training-setting1/trained_model_fold-{}.pt",
        },
        "census": {
            "num_folds": 10,
            "model_path_format": "../models/census/census-training-setting1/trained_model_fold-{}.pt",
        },
        "bank": {
            "num_folds": 10,
            "model_path_format": "../models/bank/bank-training-setting1/trained_model_fold-{}.pt",
        },
    }
    # image datasets
    dic_image = {
        # "fm": {
        #     "num_folds": 5,
        #     "model_path_format": "../models/fm/fm-training-setting1/trained_model_fold-{}.pt",
        #     "keras_model": FashionModelKeras,
        # },
        # "c10": {
        #     "num_folds": 5,
        #     "model_path_format": "../models/c10/c10-training-setting1/trained_model_fold-{}.pt",
        #     "keras_model": C10ModelKeras,
        # },
        "gtsrb": {
            "num_folds": 5,
            "model_path_format": "../models/gtsrb/gtsrb-training-setting1/trained_model_fold-{}.pt",
            "keras_model": GTSRBModelKeras,
        },
    }

    # dummy inputに対してtorch_modelとkeras_modelの出力が一致してそうか確認
    def check_output_for_dummy(torch_model, keras_model, stdout=True):
        torch.manual_seed(0)
        # バッチサイズ5のダミー入力生成
        if type(torch_model.input_dim) == int:
            dummy_in = torch.randn(5, torch_model.input_dim)
        elif type(torch_model.input_dim) == tuple:
            dummy_in = torch.randn(5, *torch_model.input_dim)
        # torch_modelの出力を計算
        torch_out = torch_model(dummy_in).detach().numpy()
        # keras_modelの出力を計算
        if (
            type(torch_model.input_dim) == tuple and len(torch_model.input_dim) == 3
        ):  # 画像の場合はchannel first -> channel lastにする
            keras_out = keras_model(dummy_in.detach().numpy().transpose(0, 2, 3, 1), training=False)
        else:
            keras_out = keras_model(dummy_in.detach().numpy(), training=False)
        # print(np.argmax(torch_out, axis=1))
        # print(np.argmax(keras_out, axis=1))
        # torch_outとkeras_outのL1ノルムを計算
        logger.info(f"L1 norm of torch_out and keras_out: {np.linalg.norm(torch_out - keras_out, ord=1)}")
        if stdout:
            print(f"L1 norm of torch_out and keras_out: {np.linalg.norm(torch_out - keras_out, ord=1)}")

    if dataset_type == "tabular":
        dic = dic_tabular
        for ds in dic.keys():
            # torch_modelの初期化
            torch_model = select_model(ds)
            for k in range(dic[ds]["num_folds"]):
                logger.info(f"dataset={ds}, k={k}")
                # 学習済みのモデルの重みをロード
                torch_model.load_state_dict(torch.load(dic[ds]["model_path_format"].format(k)))
                # torch_modelと同じ構造のkeras_modelを定義
                keras_model = TabularModelKeras(torch_model.input_dim)
                # table dataの場合torch modelの入れ子構造をflattenにする
                torch_layers = [l for seq in torch_model.layers for l in seq]
                # keras_modelにtorch_modelの重みをセットする
                # NOTE: keras_modelの0番目はInput層なので飛ばす
                for i, (tl, kl) in enumerate(zip(torch_layers, keras_model.layers[1:])):
                    tp = tl.named_parameters()
                    kp = kl.get_weights()
                    if len(kp):  # has weights
                        # torchとkerasで重みのshapeの決め方が異なるので転置を挟んでnumpyに変換してセット
                        kl.set_weights([torch.t(p).detach().numpy() for _, p in tp])

                # 各レイヤの各重みの一致性の確認============================================
                for i, (tl, kl) in enumerate(zip(torch_layers, keras_model.layers[1:])):
                    tp = tl.named_parameters()
                    kp = kl.get_weights()
                    if len(kp):  # has weights
                        for pk, pt in zip(kp, tp):
                            pt = torch.t(pt[1]).detach().numpy()
                            print(pt.shape, pk.shape)
                            print(f"sum of abs. of diff. of torch and keras weights: {np.sum(np.abs(pk - pt))}")
                # ============================================

                # ダミー入力に対する両モデルの出力の一致性を確認
                check_output_for_dummy(torch_model, keras_model, stdout=True)

                # keras_modelを保存
                save_dir = os.path.dirname(dic[ds]["model_path_format"])
                save_path = os.path.join(save_dir, "keras_model_fold-{}.h5".format(k))
                keras_model.save(save_path)
                logger.info(f"keras_model is saved to {save_path}")

    elif dataset_type == "image":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        conv2d_convert_axes = (2, 3, 1, 0)
        dic = dic_image
        for ds in dic.keys():
            # torch_modelの初期化
            torch_model = select_model(ds)
            for k in range(dic[ds]["num_folds"]):
                # 学習済みのモデルの重みをロード
                torch_weights = torch.load(dic[ds]["model_path_format"].format(k), map_location=device)
                torch_model.load_state_dict(torch_weights)
                torch_model.eval()
                # torchの重みの名前と値のリスト
                torch_keys, torch_vals = list(torch_weights.keys()), list(torch_weights.values())
                # torch_modelと同じ構造のkeras_modelを定義
                keras_model = dic[ds]["keras_model"](input_dim=torch_model.input_dim)
                for i, l in enumerate(keras_model.layers):
                    l_name = l.name
                    param = l.get_weights()
                    # kerasのレイヤが重みを持つ場合
                    if len(param):
                        # レイヤの持つ各重みについて, torchの重みを適切に変形してset_weights_listに追加
                        set_weights_list = []
                        # batchnorm用
                        if l_name.startswith("batch_normalization"):
                            # torchの重みの値のリストの先頭を取り出す
                            bn_w = torch_vals.pop(0).detach().numpy()
                            # torchの重みの値のリストの先頭を取り出す
                            bn_b = torch_vals.pop(0).detach().numpy()
                            # torchの重みの値のリストの先頭を取り出す
                            bn_rm = torch_vals.pop(0).detach().numpy()
                            # torchの重みの値のリストの先頭を取り出す
                            bn_rv = torch_vals.pop(0).detach().numpy()
                            # batch_trackedとかいう使わない奴
                            bn_unused = torch_vals.pop(0)
                            print(
                                [bn_kp.shape for bn_kp in param], [bn_tp.shape for bn_tp in [bn_w, bn_b, bn_rm, bn_rv]]
                            )
                            set_weights_list = [bn_w, bn_b, bn_rm, bn_rv]
                        # batchnorm以外
                        else:
                            for kp in param:
                                # torchの重みの値のリストの先頭を取り出す
                                tp = torch_vals.pop(0).detach().numpy()
                                print(kp.shape, tp.shape)
                                if len(tp.shape) == 4:
                                    tp = np.transpose(tp, conv2d_convert_axes)
                                elif len(tp.shape) == 2:
                                    tp = np.transpose(tp, (1, 0))
                                set_weights_list.append(tp)
                        # set_weights_listを使ってkerasの重みを設定
                        l.set_weights(set_weights_list)

                # 各レイヤの各重みの一致性の確認============================================
                torch_vals = list(torch_weights.values())
                for i, l in enumerate(keras_model.layers):
                    print(f"layer {i}")
                    l_name = l.name
                    param = l.get_weights()
                    # kerasのレイヤが重みを持つ場合
                    if len(param):
                        # batchnorm用
                        if l_name.startswith("batch_normalization"):
                            # torchの重みの値のリストの先頭を取り出す
                            bn_w = torch_vals.pop(0).detach().numpy()
                            # torchの重みの値のリストの先頭を取り出す
                            bn_b = torch_vals.pop(0).detach().numpy()
                            # torchの重みの値のリストの先頭を取り出す
                            bn_rm = torch_vals.pop(0).detach().numpy()
                            # torchの重みの値のリストの先頭を取り出す
                            bn_rv = torch_vals.pop(0).detach().numpy()
                            # batch_trackedとかいう使わない奴
                            bn_unused = torch_vals.pop(0)
                            print(
                                [bn_kp.shape for bn_kp in param], [bn_tp.shape for bn_tp in [bn_w, bn_b, bn_rm, bn_rv]]
                            )
                            print(f"sum of abs. of diff. of torch and keras weights: {np.sum(np.abs(bn_w - param[0]))}")
                            print(f"sum of abs. of diff. of torch and keras weights: {np.sum(np.abs(bn_b - param[1]))}")
                            print(
                                f"sum of abs. of diff. of torch and keras weights: {np.sum(np.abs(bn_rm - param[2]))}"
                            )
                            print(
                                f"sum of abs. of diff. of torch and keras weights: {np.sum(np.abs(bn_rv - param[3]))}"
                            )
                        # batchnorm以外
                        else:
                            for kp in param:
                                # torchの重みの値のリストの先頭を取り出す
                                tp = torch_vals.pop(0).detach().numpy()
                                if len(tp.shape) == 4:
                                    tp = np.transpose(tp, conv2d_convert_axes)
                                elif len(tp.shape) == 2:
                                    tp = np.transpose(tp, (1, 0))
                                # torchとkerasの重みのL1ノルムを計算
                                print(tp.shape, kp.shape)
                                print(f"sum of abs. of diff. of torch and keras weights: {np.sum(np.abs(kp - tp))}")
                # ============================================

                # 各レイヤの出力の一致性をチェック============================================
                torch.manual_seed(0)
                dummy_in = torch.randn(5, *torch_model.input_dim)  # バッチサイズ5のダミー入力
                # torch_modelの各レイヤのアウトプットを記録
                _, torch_out = torch_model.forward_with_layer_output(dummy_in)
                # keras_modelの各レイヤのアウトプットを記録
                keras_out = []
                out = dummy_in.detach().numpy().transpose(0, 2, 3, 1)
                for l in keras_model.layers:
                    # dummy_inを入力としてlの出力を計算
                    out = l(out)
                    if not l.name.startswith("permute"):
                        keras_out.append(out)
                # torch_modelとkeras_modelそれぞれの出力を比較する
                for i, (ko, to, kl) in enumerate(zip(keras_out, torch_out, keras_model.layers)):
                    to = to.detach().numpy()
                    print(f"layer {i}", kl.name, ko.shape, to.shape)
                    if len(to.shape) == 4:
                        to = to.transpose(0, 2, 3, 1)  # channel first -> channel last
                    assert ko.shape == to.shape
                    # # toとkoの差の絶対値を計算する
                    # diff = np.abs(ko - to)
                    # # diffの大きい方から上位100件の値とインデックスを表示
                    # print(np.sort(np.array(ko).flatten())[::-1][:100])
                    # print(np.sort(to.flatten())[::-1][:100])
                    # print(np.sort(diff.flatten())[::-1][:100])
                    # XXX: XXX: XXX: batchnormの出力が合わない;o; 数値計算の誤差か.
                    print(f"sum of abs. of diff. of torch and keras layer outs: {np.sum(np.abs(ko - to))}")
                # 確認用終了============================================

                # ダミー入力に対する両モデルの出力の一致性を確認
                check_output_for_dummy(torch_model, keras_model, stdout=True)

                # keras_modelを保存
                save_dir = os.path.dirname(dic[ds]["model_path_format"])
                save_path = os.path.join(save_dir, "keras_model_fold-{}.h5".format(k))
                keras_model.save(save_path)
                logger.info(f"keras_model is saved to {save_path}")
