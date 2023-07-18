import keras as K
import torch
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, ReLU, Input
from lib.model import select_model
from lib.log import set_exp_logging
import warnings

warnings.filterwarnings("ignore")


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


if __name__ == "__main__":
    # このプログラムのファイル名を取得
    file_name = os.path.splitext(sys.argv[0])[0]
    # 実験のディレクトリと実験名を取得
    exp_dir = "/src/experiments/arachne"
    # ログファイルの生成
    logger = set_exp_logging(exp_dir, file_name)

    dic = {
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
            if ds in ["credit", "census", "bank"]:
                torch_layers = [l for seq in torch_model.layers for l in seq]
            # keras_modelにtorch_modelの重みをセットする
            # NOTE: keras_modelの0番目はInput層なので飛ばす
            for i, (tl, kl) in enumerate(zip(torch_layers, keras_model.layers[1:])):
                tp = tl.named_parameters()
                kp = kl.get_weights()
                if len(kp):  # has weights
                    # torchとkerasで重みのshapeの決め方が異なるので転置を挟んでnumpyに変換してセット
                    kl.set_weights([torch.t(p).detach().numpy() for _, p in tp])
            # keras_modelを保存
            save_dir = os.path.dirname(dic[ds]["model_path_format"])
            save_path = os.path.join(save_dir, "keras_model_fold-{}.h5".format(k))
            keras_model.save(save_path)
            logger.info(f"keras_model is saved to {save_path}")

            # dummy inputに対してtorch_modelとkeras_modelの出力が一致してそうか確認
            dummy_in = torch.randn(5, torch_model.input_dim)  # バッチサイズ5のダミー入力
            torch_out = torch_model(dummy_in).detach().numpy()
            keras_out = keras_model(dummy_in.detach().numpy())
            # torch_outとkeras_outのL1ノルムを計算
            logger.info(f"L1 norm of torch_out and keras_out: {np.linalg.norm(torch_out - keras_out, ord=1)}")