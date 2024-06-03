import os, sys, json, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
import pandas as pd
import numpy as np
from lib.model import select_model
from lib.explanatory_metrics import get_pcs, get_entropy, get_lps, get_loss
from lib.log import set_exp_logging
from lib.safety import check_safety_prop
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold


PID_LIST = [2, 7, 8, 9]
SAMPLE_SIZE = 5000 # 一度にサンプリングするサンプル数
K = 5

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=str)
    parser.add_argument("lower", type=int)
    parser.add_argument("upper", type=int)
    parser.add_argument("pid", type=int)
    args = parser.parse_args()
    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(args.exp_dir)
    exp_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    # log setting
    logger = set_exp_logging(exp_dir, exp_name)
    nnid = args.lower, args.upper
    pid = args.pid
    logger.info(f"nnid: {nnid}, pid: {pid}")

    # pidが2, 7, 8のいずれでもなかったらエラー終了
    if pid not in PID_LIST:
        logger.error(f"pid must be {PID_LIST} for now.")
        sys.exit(1)
    
    # pidごとにほしい正例/負例の数
    if pid in [2, 8]:
        true_ex_size = 5000
        counter_ex_size = 5000
    elif pid in [7]:
        true_ex_size = 5000
        counter_ex_size = 500
    logger.info(f"num true_ex_size: {true_ex_size}, num counter_ex_size: {counter_ex_size}")


    # modelのロード
    logger.info(f"device: {device}")
    model = select_model("acasxu", nnid)
    model.to(device)

    # safety propertyのロード
    spec_dir = f"/src/models/acasxu/specs/prop{pid}"
    spec_path = os.path.join(spec_dir, f"prop{pid}_nnet_{nnid[0]}_{nnid[1]}.json")
    with open(spec_path, "r") as f:
        spec = json.load(f)
    # 各変数の取りうる範囲（正規化後）
    bounds = eval(spec["model"]["bounds"])

    # 正例と負例をメモするlist
    true_ex_list = []
    counter_ex_list = []
    # 正例と負例をそれぞれ決まった数えられるまでランダムサンプリングする
    while not (len(true_ex_list) >= true_ex_size and len(counter_ex_list) >= counter_ex_size):
        # random sampling
        samples = np.array([[np.float32(np.random.uniform(low, high)) for low, high in bounds] for _ in range(SAMPLE_SIZE)])
        samples_tensor = torch.tensor(samples)
        # model prediction
        outputs = model.predict(samples_tensor, device=device, is_normalized=True)
        # pidとoutputsから各サンプルが安全性条件を満たしたかどうかを0/1の配列に直す (正常=性質を満たす=0, 異常=性質を満たさない=1)
        indices = check_safety_prop(outputs, pid).cpu().numpy().astype(bool)
        true_ex = samples[~indices]
        counter_ex = samples[indices]
        for ds_size, ex_list, ex in zip([true_ex_size, counter_ex_size],[true_ex_list, counter_ex_list], [true_ex, counter_ex]):
            if len(ex_list) == ds_size:
                pass
            elif len(ex_list) + len(ex) <= ds_size:
                ex_list.extend(ex)
            elif len(ex_list) + len(ex) > ds_size:
                ex_list.extend(ex[:ds_size-len(ex_list)])
        logger.info(f"len(true_ex_list): {len(true_ex_list)}, len(counter_ex_list): {len(counter_ex_list)}")
    
    true_ex_list = np.array(true_ex_list) # (DATASET_SIZE, 5)
    counter_ex_list = np.array(counter_ex_list) # (DATASET_SIZE, 5)
    # true_ex_listとcounter_ex_listを結合してds_arrayにする
    ds_array = np.concatenate([true_ex_list, counter_ex_list], axis=0)
    total_ex_size = len(ds_array)
    # true_exには0, counter_exには1のラベル付
    labels = np.concatenate([np.zeros(true_ex_size), np.ones(counter_ex_size)]).astype(int)
    logger.info(f"RADON SAMPLES GENERATION IS DONE. TOTAL NUM OF SAMPLES = {total_ex_size}")
    
    # ds_array, labelsをシャッフル
    perm = np.random.permutation(total_ex_size)
    ds_array = ds_array[perm]
    labels = labels[perm]
    
    # ds_array, labelsの30%をtest dataにする
    test_size = int(0.3 * total_ex_size)
    test_ds_array = ds_array[:test_size]
    test_labels = labels[:test_size]
    test_ds = TensorDataset(torch.from_numpy(test_ds_array), torch.from_numpy(test_labels))
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
    # test_dlを保存
    ds_dir = f"/src/data/acasxu/n{nnid[0]}_{nnid[1]}_prop{pid}"
    os.makedirs(ds_dir, exist_ok=True)
    save_path = os.path.join(ds_dir, "test_loader.pt")
    torch.save(test_dl, save_path)
    logger.info(f"test_dl is saved to {save_path} ({len(test_ds)} samples.)")
    # labelの内訳もログで残す
    logger.info(f"test_labels: {test_labels.sum()} positive samples, {len(test_labels)-test_labels.sum()} negative samples.")
    
    # testに選ばれなかった残り7割から, 5-foldで1つずつ抜いた物をrepair setとして保存
    remain_array = ds_array[test_size:]
    remain_labels = labels[test_size:]
    repair_dl_list = []
    # 5-foldに分ける
    kf = KFold(n_splits=K, shuffle=True, random_state=1234)
    # foldごとにrepair setを作成する
    for k, (used_indices, _) in enumerate(kf.split(remain_array, remain_labels)):
        repair_ds_array = remain_array[used_indices]
        repair_labels = remain_labels[used_indices]
        repair_ds = TensorDataset(torch.from_numpy(repair_ds_array), torch.from_numpy(repair_labels))
        repair_dl = DataLoader(repair_ds, batch_size=32, shuffle=False)
        repair_dl_list.append(repair_dl)
        # repair_dlを保存
        save_path = os.path.join(ds_dir, f"repair_loader_fold-{k}.pt")
        torch.save(repair_dl, save_path)
        logger.info(f"repair_dl (fold{k}) is saved to {save_path} ({len(repair_ds)} samples.)")
        # labelの内訳もログで残す
        logger.info(f"repair_labels (fold{k}): {repair_labels.sum()} positive samples, {len(repair_labels)-repair_labels.sum()} negative samples.")
    
    # 二度手間だけどここで修正前モデルの各サンプルへの説明変数取得もやりたい
    # exp. metrics保存用のディレクトリを作成
    expmet_dir = f"/src/experiments/care/explanatory_metrics/acasxu_n{nnid[0]}_{nnid[1]}_prop{pid}"
    os.makedirs(expmet_dir, exist_ok=True)
    # test_dlと各foldのrepair_dlに対してexplanatory metricsを計算
    for filename, dataloader in zip(["test.csv"]+[f"repair_fold{k+1}.csv" for k in range(K)], [test_dl]+[repair_dl_list[k] for k in range(K)]):
        df = pd.DataFrame(columns=["pcs", "entropy", "x0", "x1", "x2", "x3", "x4"])
        for x, _ in dataloader.dataset:
            row_dict = {}
            x = x.to(device)
            out = model.predict(torch.unsqueeze(x, 0), device=device, is_normalized=True)
            prob = out["prob"][0].cpu()
            # probを使ったexplanatory metricsの計算
            row_dict["pcs"] = get_pcs(prob)
            row_dict["entropy"] = get_entropy(prob)
            for i, xi in enumerate(x):
                # xiはtensorからnumpyにする
                row_dict[f"x{i}"] = xi.cpu().numpy()
            df = df.append(row_dict, ignore_index=True)
        print(df)
        save_path = os.path.join(expmet_dir, filename)
        df.to_csv(save_path, index=False)
        logger.info(f"saved explanatory metrics to {save_path}. df.shape={df.shape}")