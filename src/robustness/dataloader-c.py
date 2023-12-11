"""
FMNIST-C, CIFAR-10-Cそれぞれのデータローダをpt形式で保存する
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
import pandas as pd
import numpy as np
from lib.util import json2dict, dataset_type
from lib.log import set_exp_logging
import torch
from torch.utils.data import DataLoader, TensorDataset

if __name__ == "__main__":
    # original dataset
    for ds in ["fmc", "c10c"]:
        ori_ds = ds.rstrip("c")
        ds_type = dataset_type(ori_ds)
    
        # モデルとデータの読み込み先のディレクトリ
        data_dir = f"/src/data/{ds}/raw_data"
        data_files = os.listdir(data_dir)
        model_dir = f"/src/models/{ori_ds}/{ori_ds}-training-setting1"

        # データを読み込んでnpyの辞書を作る
        npy_dic = {}
        for file_name in data_files:
            file_path = os.path.join(data_dir, file_name)
            # file_nameからnpyを除いた部分だけ取得
            key = file_name.replace(".npy", "")
            # npyをロード
            npy_dic[key] = torch.from_numpy(np.load(file_path).astype(np.float32)).clone()

        # 読み込んだnpyからdataloaderを作成
        # データセットにより異なる処理をしないといけない
        dl_dic = {}
        if ds == "fmc":
            # train, testに対するTensorDatasetを作成
            train_x = npy_dic["fmnist-c-train"] / 255
            test_x = npy_dic["fmnist-c-test"] / 255
            # channelの次元をバッチの次に追加
            train_x = train_x.unsqueeze(1)
            test_x = test_x.unsqueeze(1)
            train_y = npy_dic["fmnist-c-train-labels"].to(int)
            test_y = npy_dic["fmnist-c-test-labels"].to(int)
            train_ds = TensorDataset(train_x, train_y)
            test_ds = TensorDataset(test_x, test_y)
            # DataLoaderにして辞書に入れる
            dl_dic["train"] = DataLoader(train_ds, batch_size=32, shuffle=False)
            dl_dic["test"] = DataLoader(test_ds, batch_size=32, shuffle=False)
        elif ds == "c10c":
            labels = npy_dic["labels"].to(int)
            dl_dic = {}
            # Corruptionの種類ごとにTensorDatasetを作成
            for k, v in npy_dic.items():
                if "labels" in k:
                    continue
                # まずはtensor datasetをつくる
                # vはchannel_lastで保存されてるのでtorchのモデルで処理できるようにchannel_firstにする
                v = v.permute(0, 3, 1, 2) / 255
                corruption_ds = TensorDataset(v, labels)
                dl_dic[k] = DataLoader(corruption_ds, batch_size=32, shuffle=False)
        
        # dl_dicの各val (dataloader) をptで保存
        for k, dl in dl_dic.items():
            save_path = os.path.join(f"/src/data/{ds}", f"{k}_loader.pt")
            torch.save(dl, save_path)
            print(f"saved {k} dataloader as {save_path}")