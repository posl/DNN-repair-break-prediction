from collections import namedtuple
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

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
            label = label.item() # tensorからintに直す
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


def divide_train_repair(ori_train_dataset, num_fold=5, batch_size=16):
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
        # 各foldでのデータセットの作成
        train = ori_train_dataset[list(ti)]
        repair = ori_train_dataset[list(ri)]

        # dataset, dataloaderの作成
        # FIXME: データセットの各データがx, yでアクセスできる必要がある
        train_ds = TensorDataset(train.x, train.y)
        repair_ds = TensorDataset(repair.x, repair.y)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        repair_loader = DataLoader(repair_ds, batch_size=batch_size, shuffle=True)

        # 結果のlistに追加
        train_loader_list.append(train_loader)
        repair_loader_list.append(repair_loader)
    logger.info(
        f"len(train_loader_list[0].dataset)={len(train_loader_list[0].dataset)}, len(repair_loader_list[0].dataset)={len(repair_loader_list[0].dataset)}"
    )
    return train_loader_list, repair_loader_list


def divide_train_repair_test(ori_dataset, test_ratio=0.1, num_fold=5, batch_size=16):
    """
    データセットをtrain, repair, testに分ける.trainとrepairはnum_foldだけのfoldに分割される.

    Args:
        ori_dataset (Dataset): オリジナルのデータセット（全く分割されてない状態）.
        test_ratio (float, optional): test setの割合. Defaults to 0.1.
        num_fold (int, optional): fold数. Defaults to 5.
        batch_size (int, optional): バッチサイズ. Defaults to 16.

    Returns:
        list: 各foldのtrain set.
        list: 各foldのrepair set.
        DataLoader: test setのデータローダ.
    """

    # train_repairとtestに分割
    train_repair, test = random_split(
        dataset=ori_dataset,
        lengths=[1 - test_ratio, test_ratio],
        generator=torch.Generator().manual_seed(629),
    )

    # train_repairをtrain, testに分割
    train_loader_list, repair_loader_list = divide_train_repair(
        ori_train_dataset=train_repair, num_fold=num_fold, batch_size=batch_size
    )

    # test用のデータローダも作成
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader_list, repair_loader_list, test_loader
