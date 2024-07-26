import os, sys
import pickle
from collections import defaultdict
from itertools import permutations
import pandas as pd
import numpy as np
from lib.log import set_exp_logging
import matplotlib.pyplot as plt
import seaborn as sns
# plot setting
sns.set()

if __name__ == "__main__":
    # このプログラムのファイル名を取得
    datasets = ["credit", "census", "bank", "fm", "c10", "gtsrb", "imdb", "rtmr"]
    # 保存先のディレクトリ
    transferability_dir = os.path.join("/src/experiments/", "method-transferability")
    compatibility_dir = os.path.join("/src/experiments/", "method-compatibility")
    
    # 各組み合わせについて実行
    for rb in ["repair", "break"]:
        for dir in [transferability_dir, compatibility_dir]:
            df_list = []
            for dataset in datasets:
                df_list.append(pd.read_csv(os.path.join(dir, f"{dataset}-{rb}.csv")))
            # trans_df_listを縦に結合
            trans_df = pd.concat(df_list)
            trans_df.to_csv(os.path.join(dir, f"all-{rb}.csv"), index=False)
