import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef

# for logging
from logging import getLogger

logger = getLogger("base_logger")


# FIXME:良くない定数
CREDIT_INPUT_DIM = 23
CENSUS_INPUT_DIM = 38
BANK_INPUT_DIM = 33


class TabularModel(nn.Module):
    """
    テーブルデータのためのNNの基底クラス.
    共通のメソッドはここにまとめて，各データセットのモデルで上書きできるようにする.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """順伝搬"""
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def count_neurons_params(self):
        """ニューロン数と学習できるパラメータ数を数える"""
        num_neurons, num_params = self.input_dim, 0
        for name, param in self.named_parameters():
            if not "bias" in name:
                num_neurons += param.shape[0]
            num_params += np.prod(param.shape)
        return {"num_neurons": num_neurons, "num_params": num_params}

    def predict(self, x):
        """バッチの予測を実行"""
        out = self.forward(x)
        prob = nn.Softmax(dim=1)(out)
        pred = torch.argmax(prob, dim=1)
        return {"prob": prob, "pred": pred}

    def get_layer_distribution(self, ds, target_lid):
        """データxを入力した際の指定したレイヤのニューロンの値（活性化関数通す前, 全結合の後）の分布を出力する.
        出力されるndarrayの形状は, (データ数, target_lidのニューロン数)となる.


        Args:
            ds (torch.Dataset): 入力データセット.
            target_lid (int): 対象となるニューロンがあるレイヤのインデックス (0-indexed). e.g., 最初の全結合層の出力ニューロンに対しては0. 2番目に対しては1.

        Returns:
            ndarray: 各データサンプルに対する指定したレイヤにおける各ニューロンの値が入ったndarray
        """
        # 返したいリスト
        hvals = []
        for x, _ in ds:
            o = x
            # 各レイヤに対して順伝搬
            for lid, layer in enumerate(self.layers):
                fc_out = layer[0](o)  # linear
                o = layer[1](fc_out)  # relu
                # 指定したレイヤの各ニューロンの値を取り出して次のデータへ
                if lid == target_lid:
                    hvals.append(fc_out.tolist())
                    break
        return np.array(hvals)

    def get_neuron_distribution(self, ds, target_lid, target_nid):
        """データxを入力した際の指定したニューロンの値（活性化関数通す前, 全結合の後）の分布を出力する.
        出力されるリストの長さは, データ数と同じになる.


        Args:
            ds (torch.Dataset): 入力データセット.
            target_lid (int): 対象となるニューロンがあるレイヤのインデックス (0-indexed). e.g., 最初の全結合層の出力ニューロンに対しては0. 2番目に対しては1.
            target_nid (int): 対象となるニューロンのインデックス (0-indexed).

        Returns:
            list: 各データサンプルに対する指定したニューロンの値が入ったリスト
        """
        # 返したいリスト
        hvals = []
        for x, _ in ds:
            o = x
            # 各レイヤに対して順伝搬
            for lid, layer in enumerate(self.layers):
                fc_out = layer[0](o)  # linear
                o = layer[1](fc_out)  # relu
                # 指定したレイヤに対して，指定したニューロンの値を取り出して次のデータへ
                if lid == target_lid:
                    hvals.append(fc_out[target_nid].item())
                    break
        return hvals

    def predict_with_intervention(self, x, hval, target_lid=None, target_nid=None):
        """指定したニューロンの値に介入した場合 (= あるニューロンの値を固定する) の予測結果を返す.

        Args:
            x (torch.Tensor): データのtensor.
            hval (float): 介入後の指定したニューロンの値.
            target_lid (int): 対象となるニューロンがあるレイヤのインデックス (0-indexed). e.g., 最初の全結合層の出力ニューロンに対しては0. 2番目に対しては1.
            target_nid (int): 対象となるニューロンのインデックス (0-indexed).

        Returns:
            dict: 予測確率と予測ラベルの辞書.
        """
        o = x
        # layerごとの順伝搬
        for lid, layer in enumerate(self.layers):
            fc_out = layer[0](o)  # linear

            # 介入したニューロン値に対しては全結合層の後のニューロンを介入後の値で置き換える
            if lid == target_lid:
                fc_out[0][target_nid] = torch.tensor(hval, dtype=torch.float32)

            # 最終層かどうかで場合分け
            if len(layer) == 2:
                o = layer[1](fc_out)  # relu
            elif len(layer) == 1:
                o = fc_out  # 最終層

        # 最終層の出力から予測確率とラベルを取得
        prob = nn.Softmax(dim=1)(o.view(1, -1))  # バッチ次元を追加するため
        pred = torch.argmax(prob, dim=1)
        return {"prob": prob, "pred": pred}

    def predict_with_repair(self, x, hvals, neuron_location):
        """対象のニューロンに摂動を加えた場合 (=パッチの候補を適用した場合) の予測結果を返す.

        Args:
            x (torch.Tensor): データのTensor.
            hvals (list of float): 修正後のニューロンの値のリスト. indexは第三引数のneuron_locationと対応.
            neuron_location (list of tuple(int, int)): 修正するニューロンの位置(レイヤ番号, ニューロン番号)を表すタプルのリスト.

        Returns:
            dict: 予測確率と予測ラベルの辞書.
        """
        o = x
        # 修正したいニューロンの数の確認
        assert len(neuron_location) == len(hvals), f"Error: len(neuron_location) != len(hvals)."
        repair_num = len(neuron_location)

        # layerごとの順伝搬
        for lid, layer in enumerate(self.layers):
            fc_out = layer[0](o)  # linear

            # 修正後の値を対象のニューロンにセット（現在のlidに対して）
            for hval, (target_lid, target_nid) in zip(hvals, neuron_location):
                if lid == target_lid:
                    # NOTE: 公式実装に準拠 (lib_models.pyのapply_repair_fixed())
                    fc_out[0][target_nid] *= 1 + torch.tensor(hval, dtype=torch.float32)

            # 最終層かどうかで場合分け
            if len(layer) == 2:
                o = layer[1](fc_out)  # relu
            elif len(layer) == 1:
                o = fc_out  # 最終層

        # 最終層の出力から予測確率とラベルを取得
        prob = nn.Softmax(dim=1)(o.view(1, -1))  # バッチ次元を追加するため
        pred = torch.argmax(prob, dim=1)
        return {"prob": prob, "pred": pred}


class CreditModel(TabularModel):
    """creditデータセットのためのモデルクラス.

    (Extended Summary)
    German Creditのためのデータセット.
    CARE論文のTable1のNN2と違うのは,入力の次元数.
    CARE論文では特徴量エンジニアリングの詳細について述べられておらず,モデルの再現ができないという問題がある.
    そこで,特徴量エンジニアリングした列を入力とするため,オリジナルの列数の20から変わっている.
    この値は上の定数CREDIT_INPUT_DIMで指定している.
    """

    def __init__(self):
        super().__init__()
        self.input_dim, self.output_dim = CREDIT_INPUT_DIM, 2
        self.layers = nn.Sequential(
            nn.Sequential(torch.nn.Linear(self.input_dim, 64), nn.ReLU()),
            nn.Sequential(torch.nn.Linear(64, 32), nn.ReLU()),
            nn.Sequential(torch.nn.Linear(32, 16), nn.ReLU()),
            nn.Sequential(torch.nn.Linear(16, 8), nn.ReLU()),
            nn.Sequential(torch.nn.Linear(8, 4), nn.ReLU()),
            nn.Sequential(torch.nn.Linear(4, self.output_dim)),
        )
        logger.info(self.count_neurons_params())


class CensusModel(TabularModel):
    """censusデータセットのためのモデルクラス.

    (Extended Summary)
    Census Incomeのためのデータセット.
    CARE論文のTable1のNN1と違うのは,入力の次元数.
    CARE論文では特徴量エンジニアリングの詳細について述べられておらず,モデルの再現ができないという問題がある.
    そこで,特徴量エンジニアリングした列を入力とするため,オリジナルの列数の13から変わっている.
    この値は上の定数CENSUS_INPUT_DIMで指定している.
    """

    def __init__(self):
        super().__init__()
        self.input_dim, self.output_dim = CENSUS_INPUT_DIM, 2
        self.layers = nn.Sequential(
            nn.Sequential(torch.nn.Linear(self.input_dim, 64), nn.ReLU()),
            nn.Sequential(torch.nn.Linear(64, 32), nn.ReLU()),
            nn.Sequential(torch.nn.Linear(32, 16), nn.ReLU()),
            nn.Sequential(torch.nn.Linear(16, 8), nn.ReLU()),
            nn.Sequential(torch.nn.Linear(8, 4), nn.ReLU()),
            nn.Sequential(torch.nn.Linear(4, self.output_dim)),
        )
        logger.info(self.count_neurons_params())


class BankModel(TabularModel):
    """bankデータセットのためのモデルクラス.

    (Extended Summary)
    Bank Marketingのためのデータセット.
    CARE論文のTable1のNN3と違うのは,入力の次元数.
    CARE論文では特徴量エンジニアリングの詳細について述べられておらず,モデルの再現ができないという問題がある.
    そこで,特徴量エンジニアリングした列を入力とするため,オリジナルの列数の13から変わっている.
    この値は上の定数BANK_INPUT_DIMで指定している.
    """

    def __init__(self):
        super().__init__()
        self.input_dim, self.output_dim = BANK_INPUT_DIM, 2
        self.layers = nn.Sequential(
            nn.Sequential(torch.nn.Linear(self.input_dim, 64), nn.ReLU()),
            nn.Sequential(torch.nn.Linear(64, 32), nn.ReLU()),
            nn.Sequential(torch.nn.Linear(32, 16), nn.ReLU()),
            nn.Sequential(torch.nn.Linear(16, 8), nn.ReLU()),
            nn.Sequential(torch.nn.Linear(8, 4), nn.ReLU()),
            nn.Sequential(torch.nn.Linear(4, self.output_dim)),
        )
        logger.info(self.count_neurons_params())


class ImageModel(nn.Module):
    """
    画像データのためのNNの基底クラス.
    共通のメソッドはここにまとめて，各データセットのモデルで上書きできるようにする.
    """

    def __init__(self):
        super().__init__()

    def count_neurons_params(self):
        """ニューロン数と学習できるパラメータ数を数える. CNNでも同様の実装で動くはず."""
        num_neurons, num_params = 1, 0
        for i in self.input_dim:
            num_neurons *= i
        for name, param in self.named_parameters():
            if not "bias" in name:
                num_neurons += param.shape[0]
            num_params += np.prod(param.shape)
        return {"num_neurons": num_neurons, "num_params": num_params}

    def predict(self, x, device="cpu"):
        x = x.to(device)
        """バッチの予測を実行"""
        out = self.forward(x)
        prob = nn.Softmax(dim=1)(out)
        pred = torch.argmax(prob, dim=1)
        return {"prob": prob, "pred": pred}

    # TODO: CARE適用のために以下の実装を行う. Apricotとかでは使わないのでApricot適用中にやるのがよさげ
    # ======================================================
    def get_layer_distribution(self, ds, target_lid):
        pass

    def get_neuron_distribution(self, ds, target_lid, target_nid):
        pass

    def predict_with_intervention(self, x, hval, target_lid=None, target_nid=None):
        pass

    def predict_with_repair(self, x, hvals, neuron_location):
        pass

    # ======================================================


class FashionModel(ImageModel):
    """Fashion-MNIST datasetのためのモデルクラス."""

    def __init__(self):
        super().__init__()
        self.input_dim = (1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.dense1 = nn.Linear(7 * 7 * 64, 1024)
        self.dropout = nn.Dropout(0.4)
        self.dense2 = nn.Linear(1024, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.dense1(out))
        out = self.dropout(out)
        out = self.dense2(out)
        return out

    def get_layer_distribution(self, data, target_lid=None, device="cpu"):
        """データdataを入力した際の, 指定したレイヤのニューロンの値（活性化関数通す前, i.e., 全結合の後）の分布を出力する.
        出力されるndarrayの形状は, (データ数, target_lidのニューロン数)となる.

        Args:
            data (torch.Tensor): 入力データ (呼び出し側でバッチに分割されてる想定).
            target_lid (int): 対象となるニューロンがあるレイヤのインデックス (0-indexed).

        Returns:
            ndarray: 各データサンプルに対する指定したレイヤにおける各ニューロンの値が入ったndarray
        """
        # 順伝搬を所望のレイヤまで行う
        out = data
        # layer 0
        out = F.relu(self.conv1(out))
        # layer 1
        out = F.max_pool2d(out, 2)
        # layer 2
        out = F.relu(self.conv2(out))
        # layer 3
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        # layer 4
        # NOTE: hard coding the target lid is 4. i.e. The output of dense1 (1,024 neurons) are subject to repair.
        out = self.dense1(out)
        return out.detach().cpu().numpy()

        # out = F.relu(out)
        # # layer 5
        # out = self.dropout(out)
        # # layer 6
        # out = self.dense2(out)

    def predict_with_intervention(self, data, hval, target_lid=None, target_nid=None, device="cpu"):
        # 順伝搬を行う
        out = data
        # layer 0
        out = F.relu(self.conv1(out))
        # layer 1
        out = F.max_pool2d(out, 2)
        # layer 2
        out = F.relu(self.conv2(out))
        # layer 3
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        # layer 4
        # NOTE: hard coding the target lid is 4. i.e. The output of dense1 (1,024 neurons) are subject to repair.
        out = self.dense1(out)
        out[:, target_nid] = torch.tensor(hval, dtype=torch.float32, device=device)
        out = F.relu(out)
        # layer 5
        out = self.dropout(out)
        # layer 6
        out = self.dense2(out)
        # 最終層の出力から予測確率とラベルを取得
        prob = nn.Softmax(dim=1)(out)  # バッチ次元を追加するため
        pred = torch.argmax(out, dim=1)
        return {"prob": prob, "pred": pred}

    def predict_with_repair(self, ds, hvals, target_lid=None, neuron_location=None, device="cpu"):
        # データ, ラベルの部分を取り出してそれぞれテンソルにする
        data = torch.zeros((len(ds), *ds[0][0].shape), device=device)
        labels = torch.zeros((len(ds),), device=device)
        for i, (d, l) in enumerate(ds):
            data[i] = d
            labels[i] = l
        # 順伝搬を行う
        out = data
        # layer 0
        out = F.relu(self.conv1(out))
        # layer 1
        out = F.max_pool2d(out, 2)
        # layer 2
        out = F.relu(self.conv2(out))
        # layer 3
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        # layer 4
        # NOTE: hard coding the target lid is 4. i.e. The output of dense1 (1,024 neurons) are subject to repair.
        out = self.dense1(out)
        # tabular modelの実装と同様に各neuronに対するhvalの適用を行う
        for hval, target_nid in zip(hvals, neuron_location):
            out[:, target_nid] *= 1 + torch.tensor(hval, dtype=torch.float32, device=device)
        out = F.relu(out)
        # layer 5
        out = self.dropout(out)
        # layer 6
        out = self.dense2(out)
        # 最終層の出力から予測確率とラベルを取得
        prob = nn.Softmax(dim=1)(out)  # バッチ次元を追加するため
        pred = torch.argmax(out, dim=1)
        return {"prob": prob, "pred": pred, "labels": labels}


class GTSRBModel(ImageModel):
    """GTSRB datasetのためのモデルクラス."""

    def __init__(self):
        super().__init__()
        self.input_dim = (3, 48, 48)
        self.conv1 = nn.Conv2d(3, 100, 3)
        self.batch_normalization1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, 4)
        self.batch_normalization2 = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, 3)
        self.batch_normalization3 = nn.BatchNorm2d(250)
        self.dense1 = nn.Linear(250 * 4 * 4, 200)
        self.batch_normalization4 = nn.BatchNorm1d(200)
        self.dense2 = nn.Linear(200, 43)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.batch_normalization1(out)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = self.batch_normalization2(out)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = self.batch_normalization3(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.dense1(out))
        out = self.batch_normalization4(out)
        # out = F.softmax(self.dense2(out), dim=1) #元々の実装ではここだけsoftmaxとられており統一性がない
        out = self.dense2(out)
        return out

    def get_layer_distribution(self, data, target_lid=None, device="cpu"):
        # 順伝搬を所望のレイヤまで行う
        out = data
        # layer 0
        out = F.relu(self.conv1(out))
        # layer 1
        out = self.batch_normalization1(out)
        # layer 2
        out = F.max_pool2d(out, 2)
        # layer 3
        out = F.relu(self.conv2(out))
        # layer 4
        out = self.batch_normalization2(out)
        # layer 5
        out = F.max_pool2d(out, 2)
        # layer 6
        out = F.relu(self.conv3(out))
        # layer 7
        out = self.batch_normalization3(out)
        # layer 8
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        # layer 9
        # NOTE: hard coding the target lid is 9. i.e. The output of dense1 (200 neurons) are subject to repair.
        out = self.dense1(out)
        return out.detach().cpu().numpy()
        # out = F.relu(out)
        # out = self.batch_normalization4(out)
        # # out = F.softmax(self.dense2(out), dim=1) #元々の実装ではここだけsoftmaxとられており統一性がない
        # out = self.dense2(out)

    def predict_with_intervention(self, data, hval, target_lid=None, target_nid=None, device="cpu"):
        # 順伝搬を行う
        out = data
        # layer 0
        out = F.relu(self.conv1(out))
        # layer 1
        out = self.batch_normalization1(out)
        # layer 2
        out = F.max_pool2d(out, 2)
        # layer 3
        out = F.relu(self.conv2(out))
        # layer 4
        out = self.batch_normalization2(out)
        # layer 5
        out = F.max_pool2d(out, 2)
        # layer 6
        out = F.relu(self.conv3(out))
        # layer 7
        out = self.batch_normalization3(out)
        # layer 8
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        # layer 9
        # NOTE: hard coding the target lid is 9. i.e. The output of dense1 (200 neurons) are subject to repair.
        out = self.dense1(out)
        out[:, target_nid] = torch.tensor(hval, dtype=torch.float32, device=device)
        out = F.relu(out)
        # layer 10
        out = self.batch_normalization4(out)
        # layer 11
        out = self.dense2(out)
        # 最終層の出力から予測確率とラベルを取得
        prob = nn.Softmax(dim=1)(out)  # バッチ次元を追加するため
        pred = torch.argmax(out, dim=1)
        return {"prob": prob, "pred": pred}

    def predict_with_repair(self, ds, hvals, target_lid=None, neuron_location=None, device="cpu"):
        # データ, ラベルの部分を取り出してそれぞれテンソルにする
        data = torch.zeros((len(ds), *ds[0][0].shape), device=device)
        labels = torch.zeros((len(ds),), device=device)
        for i, (d, l) in enumerate(ds):
            data[i] = d
            labels[i] = l
        # 順伝搬を行う
        out = data
        # layer 0
        out = F.relu(self.conv1(out))
        # layer 1
        out = self.batch_normalization1(out)
        # layer 2
        out = F.max_pool2d(out, 2)
        # layer 3
        out = F.relu(self.conv2(out))
        # layer 4
        out = self.batch_normalization2(out)
        # layer 5
        out = F.max_pool2d(out, 2)
        # layer 6
        out = F.relu(self.conv3(out))
        # layer 7
        out = self.batch_normalization3(out)
        # layer 8
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        # layer 9
        # NOTE: hard coding the target lid is 9. i.e. The output of dense1 (200 neurons) are subject to repair.
        out = self.dense1(out)
        for hval, target_nid in zip(hvals, neuron_location):
            out[:, target_nid] *= 1 + torch.tensor(hval, dtype=torch.float32, device=device)
        out = F.relu(out)
        # layer 10
        out = self.batch_normalization4(out)
        # layer 11
        out = self.dense2(out)
        # 最終層の出力から予測確率とラベルを取得
        prob = nn.Softmax(dim=1)(out)  # バッチ次元を追加するため
        pred = torch.argmax(out, dim=1)
        return {"prob": prob, "pred": pred, "labels": labels}


class C10Model(ImageModel):
    """CIFAR-10 datasetのためのモデルクラス."""

    def __init__(self):
        super().__init__()
        self.input_dim = (3, 28, 28)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.dense1 = nn.Linear(2048 * 4, 256)
        self.dense2 = nn.Linear(256, 256)
        self.dense3 = nn.Linear(256, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.dense1(out))
        out = F.relu(self.dense2(out))
        out = self.dense3(out)
        return out

    def get_layer_distribution(self, data, target_lid=None, device="cpu"):
        # 順伝搬を所望のレイヤまで行う
        out = data
        # layer 0
        out = F.relu(self.conv1(out))
        # layer 1
        out = F.relu(self.conv2(out))
        # layer 2
        out = F.max_pool2d(out, 2)
        # layer 3
        out = F.relu(self.conv3(out))
        # layer 4
        out = F.relu(self.conv4(out))
        # layer 5
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        # layer 6
        out = F.relu(self.dense1(out))
        # layer 7
        # NOTE: hard coding the target lid is 7. i.e. The output of dense1 (256 neurons) are subject to repair.
        out = self.dense2(out)
        return out.detach().cpu().numpy()
        out = F.relu(out)
        # layer 8
        # out = self.dense3(out)

    def predict_with_intervention(self, data, hval, target_lid=None, target_nid=None, device="cpu"):
        # 順伝搬を行う
        out = data
        # layer 0
        out = F.relu(self.conv1(out))
        # layer 1
        out = F.relu(self.conv2(out))
        # layer 2
        out = F.max_pool2d(out, 2)
        # layer 3
        out = F.relu(self.conv3(out))
        # layer 4
        out = F.relu(self.conv4(out))
        # layer 5
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        # layer 6
        out = F.relu(self.dense1(out))
        # layer 7
        # NOTE: hard coding the target lid is 7. i.e. The output of dense1 (256 neurons) are subject to repair.
        out = self.dense2(out)
        out[:, target_nid] = torch.tensor(hval, dtype=torch.float32, device=device)
        out = F.relu(out)
        # layer 8
        out = self.dense3(out)
        # 最終層の出力から予測確率とラベルを取得
        prob = nn.Softmax(dim=1)(out)  # バッチ次元を追加するため
        pred = torch.argmax(out, dim=1)
        return {"prob": prob, "pred": pred}

    def predict_with_repair(self, ds, hvals, target_lid=None, neuron_location=None, device="cpu"):
        # データ, ラベルの部分を取り出してそれぞれテンソルにする
        data = torch.zeros((len(ds), *ds[0][0].shape), device=device)
        labels = torch.zeros((len(ds),), device=device)
        for i, (d, l) in enumerate(ds):
            data[i] = d
            labels[i] = l
        # 順伝搬を行う
        out = data
        # layer 0
        out = F.relu(self.conv1(out))
        # layer 1
        out = F.relu(self.conv2(out))
        # layer 2
        out = F.max_pool2d(out, 2)
        # layer 3
        out = F.relu(self.conv3(out))
        # layer 4
        out = F.relu(self.conv4(out))
        # layer 5
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        # layer 6
        out = F.relu(self.dense1(out))
        # layer 7
        # NOTE: hard coding the target lid is 7. i.e. The output of dense1 (256 neurons) are subject to repair.
        out = self.dense2(out)
        for hval, target_nid in zip(hvals, neuron_location):
            out[:, target_nid] *= 1 + torch.tensor(hval, dtype=torch.float32, device=device)
        out = F.relu(out)
        # layer 8
        out = self.dense3(out)
        # 最終層の出力から予測確率とラベルを取得
        prob = nn.Softmax(dim=1)(out)  # バッチ次元を追加するため
        pred = torch.argmax(out, dim=1)
        return {"prob": prob, "pred": pred, "labels": labels}


def train_model(model, dataloader, num_epochs):
    """学習用の関数

    Args:
        model (nn.Module): 初期化済みのモデル.
        dataloader (DataLoader): 学習データのdataloader.
        num_epochs (int): エポック数

    Returns:
        nn.Module: 学習したモデル
        list: 各エポックでのロスの値のリスト
    """
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # 最適化手法の設定 (lr, betasはデフォルト値)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # 誤差関数を定義
    criterion = nn.CrossEntropyLoss()
    # ネットワークをデバイスへ
    model.to(device)
    # モデルを訓練モードに
    model.train()
    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True
    # エポックごとのロスのリスト
    epoch_loss_list = []

    # epochのループ
    for epoch in range(num_epochs):
        # 開始時刻を保存
        t_epoch_start = time.time()
        epoch_loss = 0.0  # epochの損失和
        print("-------------")
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-------------")
        print("（train）")
        # dataloaderからミニバッチを取り出すループ
        for x, y in dataloader:
            # GPUが使えるならGPUにデータを送る
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            # バックプロパゲーション
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # エポックごとのロスを記録
            epoch_loss += loss.item()
        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        print("-------------")
        print("epoch {} || Epoch_Loss: {:.4f} ".format(epoch, epoch_loss / len(dataloader.dataset)))
        print("timer:  {:.4f} sec.".format(t_epoch_finish - t_epoch_start))
        epoch_loss_list.append(epoch_loss / len(dataloader.dataset))
    return model, epoch_loss_list


def sm_correctness(model, dataloader, is_repair=False, hvals=None, neuron_location=None):
    """各データに対して, 予測が分類すべきラベルと合っているかどうか (あってたら1, ちがったら0) を表す配列を返す.

    Args:
        model (nn.Module): 学習済みの, 評価対象のモデル.
        dataloader (DataLoader): 評価に使いたいデータセットのdataloader.
        is_repair (bool): repair後の重みを使って予測するかどうか. ここをTrueにした場合は必ずこの後ろの2つの引数も指定しなければならない.
        hvals (list of float): 修正後のニューロンの値のリスト. indexは第三引数のneuron_locationと対応.
        neuron_location (list of tuple(int, int)): 修正するニューロンの位置(レイヤ番号, ニューロン番号)を表すタプルのリスト.

    Returns:
        真のラベルのリスト, 予測ラベルのリスト, correctnesのリスト
    """
    # 予測結果と真のラベルの配列
    y_true = []
    y_pred = []
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    # ネットワークをデバイスへ
    model.to(device)
    # モデルを推論モードに
    model.eval()
    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # データセットの各バッチを予測
    for x, y in dataloader.dataset:
        x = x.to(device)
        if not is_repair:
            # 修正前の重みで予測. NOTE: .predict はバッチ想定なので .view が必要.
            out = model.predict(torch.unsqueeze(x, 0))
        else:
            # is_repairがTrueなのにhvalsやneuron_locationがNoneならassertion errorにする.
            assert (
                hvals is not None and neuron_location is not None
            ), "despite is_repair=True, hvals and neuron_location are None!!!"
            # 修正後の重みで予測. NOTE: .predict はバッチ想定なので .view が必要.
            out = model.predict_with_repair(torch.unsqueeze(x, 0), hvals, neuron_location)
        prob, pred = out["prob"], out["pred"]
        y_true.append(y)
        y_pred.append(pred.item())
    # HACK: 以下assertionは2値分類のみの場合なのでコメントアウトしました
    # 二値分類なので0, 1以外の値があったらassertion errorにする
    # assert (
    #     np.unique(np.array(y_true)).size == 2
    # ), f"np.unique(np.array(y_true)).size == {np.unique(np.array(y_true)).size}"
    # assert (
    #     np.unique(np.array(y_pred)).size <= 2
    # ), f"np.unique(np.array(y_pred)).size == {np.unique(np.array(y_pred)).size}"
    # numpy配列にする
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correctness_arr = np.where(y_true == y_pred, 1, 0)
    return y_true, y_pred, correctness_arr


def eval_model(model, dataloader, is_repair=False, hvals=None, neuron_location=None, num_class=2):
    """モデルの評価用の関数.

    Args:
        model (nn.Module): 学習済みの, 評価対象のモデル.
        dataloader (DataLoader): 評価に使いたいデータセットのdataloader.
        is_repair (bool): repair後の重みを使って予測するかどうか. ここをTrueにした場合は必ずこの後ろの2つの引数も指定しなければならない.
        hvals (list of float): 修正後のニューロンの値のリスト. indexは第三引数のneuron_locationと対応.
        neuron_location (list of tuple(int, int)): 修正するニューロンの位置(レイヤ番号, ニューロン番号)を表すタプルのリスト.
        num_class (int): 分類クラス数. 2値分類なら2, 多値分類ならクラス数. メトリクスの取り方を決めるのに必要.

    Returns:
        *float: dataloaderでモデルを評価した各種メトリクス.
    """
    # correctnessを評価
    y_true, y_pred, correctness_arr = sm_correctness(model, dataloader, is_repair, hvals, neuron_location)
    average = "binary" if num_class == 2 else "macro"
    # 各評価指標を算出
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)
    pre = precision_score(y_true, y_pred, average=average)
    rec = recall_score(y_true, y_pred, average=average)
    mcc = matthews_corrcoef(y_true, y_pred)
    # dictにして返す
    ret_dict = {}
    ret_dict["metrics"] = acc, f1, pre, rec, mcc
    ret_dict["y_true"] = y_true
    ret_dict["y_pred"] = y_pred
    ret_dict["correctness_arr"] = correctness_arr
    return ret_dict


def select_model(task_name):
    """taskごとに対応するモデルのインスタンス（初期状態）を返す.

    Args:
        task_name (str): taskの名前.

    Raises:
        NotImplementedError: 未実装のtaskについてはこのエラー.

    Returns:
        nn.Module: 初期化したモデル
    """
    if task_name == "credit":
        return CreditModel()
    elif task_name == "census":
        return CensusModel()
    elif task_name == "bank":
        return BankModel()
    elif task_name == "fm":
        return FashionModel()
    elif task_name == "c10":
        return C10Model()
    elif task_name == "gtsrb":
        return GTSRBModel()
    else:
        raise NotImplementedError


def get_misclassified_index(y_true, y_pred):
    """予測ラベルと正解ラベルの組み合わせに対して，誤分類したサンプルのインデックスのリストを取得

    Args:
        y_true (array-like): 正解ラベルのリスト
        y_pred (array-like): 予測ラベルのリスト

    Returns:
        misclf_idx: 誤分類ごとにインデックスのリストをまとめた辞書. キーは (y_true, y_pred), 値がデータのインデックスのリスト.
    """

    # y_predとy_trueが同じ長さか確認するassertion
    assert len(y_pred) == len(y_true), "y_pred and y_true must have the same length"

    # 誤分類したサンプル数をカウントするための辞書を初期化
    misclf_idx = defaultdict(list)
    for idx, (yp, yt) in enumerate(zip(y_pred, y_true)):
        if yp != yt:
            misclf_idx[(yt, yp)].append(idx)
    return misclf_idx


def sort_keys_by_cnt(misclf_dic):
    """
    Args:
        misclf_dic: get_misclassified_indexで得られた辞書

    Returns:
        sorted_keys: 誤分類の多い順にソートされたキーのリスト
    """

    cnts = []
    for misclf_key, misclf_list in misclf_dic.items():
        cnts.append([misclf_key, len(misclf_list)])
    sorted_keys = [v[0] for v in sorted(cnts, key=lambda v: v[1], reverse=True)]
    return sorted_keys
