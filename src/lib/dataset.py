from collections import namedtuple
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from torch.nn.utils.rnn import pad_sequence

# for logging
from logging import getLogger

logger = getLogger(__name__)


class TableDataset(Dataset):
    """テーブルデータ用のデータセットクラス."""

    def __init__(self, path, target, df=None):
        df = pd.read_csv(path) if path is not None else df
        self.x = torch.tensor(df.drop(target, axis=1).values, dtype=torch.float)
        self.y = torch.tensor(df[target].values, dtype=torch.int64)
        logger.info(f"x.shape={self.x.shape}, y.shape={self.y.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        LabeledData = namedtuple("LabeledData", ["x", "y"])
        return LabeledData(self.x[idx], self.y[idx])

class VectorizedTextDataset(Dataset):
    """学習済みのword2vecを使ったテキストデータ用のデータセットクラス."""

    def __init__(self, data, label):
        # NOTE: dataはすでにベクトル化されていることを想定
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y)

def pad_collate(batch):
    """バッチ内のサンプルの長さを揃えるためのcollate_fn (DataLoaderでバッチを取得する際に最初に呼ばれるメソッド)."""

    (xs, ys) = zip(*batch)
    # バッチ内の各サンプルの長さを配列に記録（文なので可変長）
    x_lens = [len(x) for x in xs]
    xs_pad = pad_sequence(xs, batch_first=True, padding_value=0)
    # ysをtenosorのtupleからtensorにする
    ys = torch.stack(ys)
    return xs_pad, ys, x_lens

def sent2tensor(sent, input_dim, word2idx, wv_matrix, device):
    """
    文 -> tensorへの変換
    文中の各単語を埋め込み行列によってベクトルに変換する
    
    Args:
        sent: list of str
            tensorに変換するべき文
        input_dim: int
            RNNへの入力次元数(埋め込みベクトルの次元数)
        word2idx: dict
            単語 -> 単語idへの対応表
        wv_matrix: list of list of float
            単語idから埋め込みベクトルへの対応表(埋め込み行列)
            実態は(語彙数, input_dim)と言う形状の2次元配列
    Returns:
        seq: list of list of float(torch.tensor)
            各単語を埋め込み行列で変化したベクトルの列
            形状は(文中の単語数, input_dim)という2次元配列
    """
    idx_seq = []
    # 文中に現れる単語のIDの列をidx_seqに格納
    for w in sent:
        if w in word2idx:
            idx = word2idx[w]
        elif w.lower() in word2idx:
            idx = word2idx[w.lower()]
        # w が語彙になかった場合
        else:
            idx = wv_matrix.shape[0] - 1
        idx_seq.append(idx)
    seq = torch.zeros(len(idx_seq), input_dim).to(device)
    # i番目の単語に対応するベクトルを埋め込み行列から取り出す
    for i, w_idx in enumerate(idx_seq):
        seq[i] = torch.tensor(wv_matrix[w_idx])
    return seq

class BalancedSubsetDataLoader(DataLoader):
    """ApricotでrDLMを作るためのサブデータセットのためのデータロードのクラス."""

    def __init__(self, dataloader, num_samples_per_class, **kwargs):
        self.original_dataset = dataloader.dataset
        self.num_samples_per_class = num_samples_per_class
        self.indices = self._create_subset_indices(self.original_dataset, num_samples_per_class)
        sub_dataset = self._create_subset_dataset(self.original_dataset, self.indices)
        super().__init__(sub_dataset, **kwargs)

    def _create_subset_indices(self, dataset, num_samples_per_class):
        label_counts = {}
        # 各ラベルのサンプル数を数える
        for index in range(len(dataset)):
            _, label = dataset[index]
            if isinstance(label, torch.Tensor) and label.shape == torch.Size([]):
                label = label.item()  # tensorからintに直す
            if label not in label_counts:
                label_counts[label] = [index]
            else:
                label_counts[label].append(index)

        subset_indices = []
        for label_indices in label_counts.values():
            # シャッフルして num_samples_per_class 個選択
            np.random.shuffle(label_indices)
            subset_indices.extend(label_indices[:num_samples_per_class])
        return subset_indices

    def _create_subset_dataset(self, dataset, indices):
        return torch.utils.data.Subset(dataset, indices)


def divide_train_repair(ori_train_dataset, num_fold=5, batch_size=16, dataset_type="tabular"):
    """ori_train_datasetをtrainとrepairにK-foldで分割する.train/repairに対する,各foldのdataloaderのlistを返す.

    Args:
        ori_train_dataset (Dataset): 分割するtrainデータセット
        num_fold (int, optional): fold数. Defaults to 5.
        batch_size (int, optional): バッチサイズ. Defaults to 16.

    Returns:
        list: 各foldのtrain set.
        list: 各foldのrepair set.
    """

    # 各foldのtrain/repairのdataloaderを保存するlist
    train_loader_list, repair_loader_list = (
        [],
        [],
    )

    # original train setをtrainとrepairに分ける
    # この際はk-foldを使う
    # original train setの各データはちょうど1回ずつrepair setに含まれる
    kf = KFold(n_splits=num_fold, shuffle=True, random_state=777)
    for ti, ri in kf.split(ori_train_dataset):
        # tabular datasetの場合
        if dataset_type == "tabular":
            # 各foldでのデータセットの作成
            train = ori_train_dataset[list(ti)]
            repair = ori_train_dataset[list(ri)]
            train_ds = TensorDataset(train.x, train.y)
            repair_ds = TensorDataset(repair.x, repair.y)
        # image or text datasetの場合
        elif dataset_type in ["image", "text"]:
            train_ds = Subset(ori_train_dataset, list(ti))
            repair_ds = Subset(ori_train_dataset, list(ri))
        # dataloaderのバッチ取得時に呼ぶ関数の設定（textのみ設定）
        collate_fn = None
        if dataset_type == "text":
            collate_fn = pad_collate
        # dataloaderの作成
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        repair_loader = DataLoader(repair_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        # 結果のlistに追加
        train_loader_list.append(train_loader)
        repair_loader_list.append(repair_loader)
    logger.info(
        f"len(train_loader_list[0].dataset)={len(train_loader_list[0].dataset)}, len(repair_loader_list[0].dataset)={len(repair_loader_list[0].dataset)}"
    )
    return train_loader_list, repair_loader_list
