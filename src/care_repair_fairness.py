import os, sys, time

from lib.model import select_model
from lib.util import json2dict, dataset_type
from lib.log import set_exp_logging
from lib.fairness import calc_fairness_ub
import numpy as np

np.random.seed(0)  # 乱数固定
from sklearn.metrics import accuracy_score
import torch
import pyswarms as ps
from pyswarms.utils import Reporter

import matplotlib.pyplot as plt
import seaborn as sns

# plot setting
sns.set_style("white")

# tabluerデータセットにおいてfairness repairをするかどうか
TABULAR_FAIRNESS_SW = False  # FIXME: 最悪なので外部化するs

# FIXME: 定数外部化
num_reps = 5


def pso_fitness_tabular(particles, model, dataloader, sens_idx, sens_vals, repaired_positions, alpha=0.1):
    """PSOの目的関数 (for tabular dataset). 入力の形状は (PSOの粒子数, repairするニューロン数).

    Args:

        particles (ndarray): ニューロンの重み?(TODO)をどれくらい調整するか. (PSOの粒子数, repairするニューロン数)という形状. PSOにおけるswarmに対応.
        model (nn.Module): 修正対象のモデル.
        dataloader (torch.DataLoader): repairに使うデータセットのデータローダ
        sens_idx (int): sensitive featureのインデックス.
        sens_vals (list): sensitive featureのとりうる値.
        repaired_positions (list of tuple(int, int)): 修正するニューロンの位置(レイヤ番号, ニューロン番号)を表すタプルのリスト.
        alpha (float, optional): UBとaccのバランスを決めるパラメータ. Defaults to 0.1.

    Returns:
        result (list of float): 各粒子の評価値 (論文の式で示されている目的関数の値). 形状は(粒子数, )の1次元配列.
    """
    result = []
    # 各粒子に対してCAREのPSOの目的関数の値を算出し, resultにappendする
    for p in particles:
        acc, di_rate = 0.0, 0.0
        # alphaが0のときはacc計算しなくていい
        if alpha != 0:
            # ===== acc =====
            y_true = dataloader.dataset[:][1].tolist()
            y_pred = []
            for x in dataloader.dataset[:][0]:
                pred_dict = model.predict_with_repair(x=x.view(1, -1), hvals=p, neuron_location=repaired_positions)
                pred, prob = pred_dict["pred"], pred_dict["prob"]
                y_pred.append(pred.item())
            acc = accuracy_score(y_true, y_pred)
        # alphaが1.0のときはdi_rate計算しなくていい
        if alpha != 1:
            # ===== di_rate; P(N(x) \neq N(x^{\prime}))の計算 =====
            di_rate = calc_fairness_ub(model, dataloader, sens_idx, p, repaired_positions, sens_vals)
        # ===== 目的関数の計算 =====
        cost = (1.0 - alpha) * di_rate + alpha * (1.0 - acc)
        result.append(cost)
    return result


def pso_fitness_acc(particles, model, dataloader, repaired_positions, device):
    """PSOの目的関数 (for image dataset). 入力の形状は (PSOの粒子数, repairするニューロン数).
    公平性でなくaccのみを考慮する.

    Args:

        particles (ndarray): ニューロンの重み?(TODO)をどれくらい調整するか. (PSOの粒子数, repairするニューロン数)という形状. PSOにおけるswarmに対応.
        model (nn.Module): 修正対象のモデル.
        dataloader (torch.DataLoader): repairに使うデータセットのデータローダ
        repaired_positions (list of int): 修正するニューロンの位置(ニューロン番号)を表すリスト.
        device (str): GPUかCPUか.

    Returns:
        result (list of float): 各粒子の評価値 (論文の式で示されている目的関数の値). 形状は(粒子数, )の1次元配列.
    """
    result = []
    # 各粒子に対してCAREのPSOの目的関数の値を算出し, resultにappendする
    for p in particles:
        # TODO: ここをバッチ処理する
        acc_tmp = 0
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            ret_dicts = model.predict_with_repair(data, hvals=p, neuron_location=repaired_positions, device=device)
            preds = ret_dicts["pred"].cpu()
            num_corr = sum(preds == labels)
            acc_tmp += num_corr / len(preds)
        # バッチごとのaccをまとめて全体のaccにする(NOTE: 除算の誤差がきになる)
        acc_tmp /= len(dataloader)
        # デフォルトは最小化問題になってるので, 1-accにする
        cost = 1 - acc_tmp
        result.append(cost)
    return result


if __name__ == "__main__":
    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(sys.argv[1])
    exp_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    # log setting
    # {dataset}-repair-fairness-{feature}-setting{NO}.logというファイルにログを出す
    log_file_name = exp_name.replace("fairness", "repair-fairness")
    logger = set_exp_logging(exp_dir, exp_name, log_file_name)
    # FIXME: pyswarmsのせいで標準出力とreport.logに勝手にログが出るのをなんとかしたい．
    rep = Reporter(logger=logger)

    # GPUが使えるか確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # 設定用のjsonファイルをdictとしてロード
    # HACK: 共通しているので関数にまとめて自動化したい
    org_setting_dict = json2dict(sys.argv[1])
    logger.info(f"Settings: {org_setting_dict}")

    # INHERIT_SETTINGが設定されている場合はまず継承先の設定をロード
    is_inherit = False
    try:
        inherit_setting_path = org_setting_dict["INHERIT_SETTING"]
    # INHERIT_SETTINGなしの場合
    except KeyError as e:
        setting_path = sys.argv[1]
        # 実際に読み込むべき設定のロード
        setting_dict = json2dict(setting_path)
    # INHERIT_SETTINGありの場合
    else:
        is_inherit = True
        setting_path = os.path.join(exp_dir, inherit_setting_path)
        setting_dict = json2dict(setting_path)
        inherit_exp_name = os.path.splitext(os.path.basename(setting_path))[0]
        logger.info(f"Inherit from {inherit_exp_name}")

    train_setting_path = setting_dict["TRAIN_SETTING_PATH"]
    # 訓練時の設定名を取得
    train_setting_name = os.path.splitext(train_setting_path)[0]

    # 訓練時の設定も読み込む
    train_setting_dict = json2dict(os.path.join(exp_dir, train_setting_path))
    logger.info(f"TRAIN Settings: {train_setting_dict}")
    num_fold = train_setting_dict["NUM_FOLD"]
    task_name = train_setting_dict["TASK_NAME"]

    # fairnessの計算のための情報をパース
    # 画像データセットに関してはfairness関連の情報はいらない
    if dataset_type(task_name) is "tabular":
        sens_name = setting_dict["SENS_NAME"]
        sens_idx = setting_dict["SENS_IDX"]
        sens_vals = eval(setting_dict["SENS_VALS"])  # ない場合はNone, ある場合は直接listで定義したりrangeで定義できるようにする. そのためにevalを使う
        target_cls = setting_dict["TARGET_CLS"]

    # リペアする割合 (疑惑値の上位何％をrepair対象にするか)
    # repair_ratioの型がintの場合はそれをrepairするニューロン数として（上位何件）解釈し,
    # floatの場合はそれをrepairするニューロンの割合として（上位何％）解釈する
    repair_ratio = setting_dict["REPAIR_RATIO"]

    # PSOの設定ファイルをロード
    pso_setting_path = setting_dict["PSO_SETTING_PATH"] if not is_inherit else org_setting_dict["PSO_SETTING_PATH"]
    pso_setting_dict = json2dict(os.path.join(exp_dir, pso_setting_path))
    logger.info(f"PSO Settings: {pso_setting_dict}")
    n_particles = pso_setting_dict["N_PARTICLES"]
    ftol = pso_setting_dict["FTOL"]
    ftol_iter = pso_setting_dict["FTOL_ITER"]
    alpha = pso_setting_dict["ALPHA"]
    pso_iters = pso_setting_dict["PSO_ITERS"]
    c1, c2, w = pso_setting_dict["C1"], pso_setting_dict["C2"], pso_setting_dict["W"]

    # モデルとデータの読み込み先のディレクトリ
    data_dir = f"/src/data/{task_name}/{train_setting_name}"
    model_dir = f"/src/models/{task_name}/{train_setting_name}"

    # care resultのディレクトリを作成する
    care_dir = os.path.join(model_dir, "care-result")
    os.makedirs(care_dir, exist_ok=True)

    # 各foldのtrain/repairをロードして予測
    for k in range(num_fold):
        logger.info(f"processing fold {k}...")
        # 学習済みモデルをロード
        model = select_model(task_name=task_name)
        model_path = os.path.join(model_dir, f"trained_model_fold-{k}.pt")
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        # foldに対するdataloaderをロード
        # NOTE: repairだけ使うことにしてみる
        repair_data_path = os.path.join(data_dir, f"repair_loader_fold-{k}.pt")
        repair_loader = torch.load(repair_data_path)

        # FLの結果をロードする
        fl_scores_path = (
            f"/src/experiments/care/repair_results/flscore_{exp_name}_fold{k+1}.npy"
            if not is_inherit
            else f"/src/experiments/care/repair_results/flscore_{inherit_exp_name}_fold{k+1}.npy"
        )
        fl_scores = np.load(fl_scores_path)
        # repair_ratioの型がintの場合はそれをrepairするニューロン数として（上位何件）解釈し,
        # floatの場合はそれをrepairするニューロンの割合として（上位何％）解釈する
        repair_num = (
            repair_ratio if isinstance(repair_ratio, int) else np.ceil(len(fl_scores) * repair_ratio).astype(int)
        )
        # 修正対象のニューロンと疑惑値をログで表示
        logger.info(f"# of repaired neuron = {repair_num}\n{fl_scores[:repair_num]}")
        # 修正対象のニューロンの位置(何層目の,何番目のニューロンか)
        repaired_positions = fl_scores[:repair_num][:, 0]
        # target_lidが固定かどうかチェック
        if "fixed" in repaired_positions[0]:
            # fixedの部分は-1に変えてから, evalする
            repaired_positions = np.array([s.replace("fixed", "-1") for s in repaired_positions])
        repaired_positions = np.array(list(map(eval, repaired_positions)))

        # ランダム性排除のために適用をリピートする
        time_for_reps = []  # repごとの時間記録用
        for rep in range(num_reps):
            logger.info(f"starting rep {rep}...")
            care_save_dir = os.path.join(care_dir, f"rep{rep}")
            os.makedirs(care_save_dir, exist_ok=True)

            # ===== PSOのブロック =====
            # repair開始時刻
            s = time.clock()
            logger.info("Start Repairing...")
            optimizer = ps.single.GlobalBestPSO(
                n_particles=n_particles,
                dimensions=repair_num,
                options={"c1": c1, "c2": c2, "w": w},
                bounds=([[-1.0] * repair_num, [1.0] * repair_num]),  # NOTE: なぜこのようなバウンドを設定しているかわからない
                init_pos=np.zeros((n_particles, repair_num), dtype=float),
                ftol=ftol,
                ftol_iter=ftol_iter,
            )
            # arguments of objective function
            # whether do fairness repair or not.
            if TABULAR_FAIRNESS_SW:
                obj_args = {
                    "model": model,
                    "dataloader": repair_loader,
                    "sens_idx": sens_idx,
                    "sens_vals": sens_vals,
                    "alpha": alpha,
                    "repaired_positions": repaired_positions,
                }
                # Run optimization
                best_cost, best_pos = optimizer.optimize(pso_fitness_tabular, iters=pso_iters, **obj_args)
            # for image dataset
            else:
                obj_args = {
                    "model": model,
                    "dataloader": repair_loader,
                    "repaired_positions": repaired_positions,
                    "device": device,
                }
                # Run optimization
                best_cost, best_pos = optimizer.optimize(pso_fitness_acc, iters=pso_iters, **obj_args)
            # repair開始時刻
            e = time.clock()
            # repごとの経過時間
            time_for_rep = e - s
            time_for_reps.append(time_for_rep)
            # log表示
            logger.info(f"[fold: {k}, rep: {rep}] Finish Repairing!\n(best_cost, best_pos):\n{(best_cost, best_pos)}")
            logger.info(f"rep time:{time_for_rep} sec.")
            # PSOの結果得られた修正後のニューロン値を保存
            care_save_path = os.path.join(care_save_dir, f"patch_{exp_name}_fold{k}.npy")
            np.save(care_save_path, best_pos)
            logger.info(f"saved to {care_save_path}")
        # repごとの経過時間を平均してfoldごとの時間をログに出す
        time_for_fold = np.mean(time_for_reps)
        logger.info(f"Total execution time for Repair: {time_for_fold} sec.")
