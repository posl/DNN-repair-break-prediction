import os, sys, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
import pandas as pd
import numpy as np
from lib.model import select_model
from lib.explanatory_metrics import get_pcs, get_entropy, get_lps, get_loss
from lib.util import json2dict, dataset_type
from lib.log import set_exp_logging
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# plot setting
sns.set()

if __name__ == "__main__":
    # gpuが使用可能かをdeviceにセット
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["fmc", "c10c"], help="dataset name ('fmc' or 'c10c')")
    args = parser.parse_args()
    ds = args.dataset
    # log setting
    exp_dir = "/src/experiments/care/"
    expmet_dir = os.path.join(exp_dir, "explanatory_metrics", ds)
    os.makedirs(expmet_dir, exist_ok=True)
    this_file_name = os.path.basename(__file__).replace(".py", "").replace("_", "-")
    log_file_name = f"{ds}-{this_file_name}"
    logger = set_exp_logging(exp_dir, log_file_name, log_file_name)
    logger.info(f"target dataset: {ds}, device={device}")
    
    # original dataset
    if ds in ["fmc", "c10c"]:
        ori_ds = ds.rstrip("c")
        num_fold = 5
        ds_type = dataset_type(ori_ds)
        is_binary = False
    else:
        raise ValueError(f"Invalid dataset name: {ds}")

    # モデルとデータの読み込み先のディレクトリ
    data_dir = f"/src/data/{ds}/"
    dl_files = [f for f in os.listdir(data_dir) if f.endswith("loader.pt")]
    model_dir = f"/src/models/{ori_ds}/{ori_ds}-training-setting1"

    # dataloaderを読み込む
    dl_dic = {}
    for file_name in dl_files:
        file_path = os.path.join(data_dir, file_name)
        # file_nameから'_loader.pt'を除いた部分だけ取得
        key = file_name.replace("_loader.pt", "")
        dl_dic[key] = torch.load(file_path)
        logger.info(f"loaded {key} dataloader.")
    
    # 各foldのtrain/repairをロードして予測
    for k in range(num_fold):
        logger.info(f"processing fold {k}...")

        # 学習済みモデルをロード
        model = select_model(task_name=ori_ds)
        model_path = os.path.join(model_dir, f"trained_model_fold-{k}.pt")
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        # dl_dicにあるdlのそれぞれに対して, evaluation criteriaの計算
        for key, dl in dl_dic.items():
            df = pd.DataFrame(columns=["pcs", "lps", "entropy", "loss"])
            logger.info(f"processing dl {key}...")
            for x, y in tqdm(dl, desc=key):
                out = model.predict(x, device=device)
                probs = out["prob"].cpu()
                for i, prob in enumerate(probs):
                    row_dict = {}
                    # probを使ったexplanatory metricsの計算
                    row_dict["pcs"] = get_pcs(prob)
                    row_dict["entropy"] = get_entropy(prob)
                    row_dict["lps"] = get_lps(prob, torch.tensor(y[i]))
                    row_dict["loss"] = get_loss(prob, torch.tensor(y[i]))
                    df = df.append(row_dict, ignore_index=True)

            # 保存先を指定してsaveする
            save_path = os.path.join(expmet_dir, f"{key}_fold{k+1}.csv")
            df.to_csv(save_path, index=False)
            logger.info(f"expmet saved to {save_path}")
