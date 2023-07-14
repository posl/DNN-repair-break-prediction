import os, sys, time
import warnings

warnings.filterwarnings("ignore")

from lib.log import set_exp_logging
from lib.util import json2dict
from lib.model import train_model, eval_model, select_model

import torch
import numpy as np

# Hyper-parameters
num_restrict = 1000  # rDLMを作るための縮小データセットを作る際に，各ラベルからどれだけサンプルするか
rDLM_num = 20
batch_size = 64
num_reps = 5


# utility function(s)
def setWeights(model, weight_list):
    """
    Set the weights of the model to the given list of weights.
    """
    for weight, (name, v) in zip(weight_list, model.named_parameters()):
        attrs = name.split(".")
        obj = model
        for attr in attrs:
            obj = getattr(obj, attr)
        obj.data = weight


def AdjustWeights(baseWeights, corrDiff, incorrDiff, a, b, strategy="both-org", lr=1e-3):
    if "org" in strategy:
        sign = 1
    else:
        sign = -1

    p_corr, p_incorr = a / (a + b), b / (a + b)

    if "both" in strategy:
        return [
            b_w + sign * lr * (p_corr * cD - p_incorr * iD) for b_w, cD, iD in zip(baseWeights, corrDiff, incorrDiff)
        ]
    elif "corr" in strategy:
        return [b_w + sign * lr * p_corr * cD for b_w, cD in zip(baseWeights, corrDiff)]
    elif "incorr" in strategy:
        return [b_w - sign * lr * p_incorr * iD for b_w, iD in zip(baseWeights, incorrDiff)]
    else:
        raise ValueError(f"Unrecognized strategy {strategy}")


if __name__ == "__main__":
    """
    {dataset名}-training-setting{sid}を入力として必要な情報の準備
    """

    # 実験のディレクトリと実験名を取得
    exp_dir = os.path.dirname(sys.argv[1])
    exp_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]

    # log setting
    # {dataset}-repair-fairness-{feature}-setting{NO}.logというファイルにログを出す
    log_file_name = exp_name.replace("training", "apply-apricot")
    logger = set_exp_logging(exp_dir.replace("care", "apricot"), exp_name, log_file_name)

    # 設定用のjsonファイルをdictとしてロード
    # HACK: 共通しているので関数にまとめて自動化したい
    setting_dict = json2dict(sys.argv[1])
    logger.info(f"Settings: {setting_dict}")
    task_name = setting_dict["TASK_NAME"]
    train_repair_data_path = setting_dict["TRAIN-REPAIR_DATA_PATH"]
    test_data_path = setting_dict["TEST_DATA_PATH"]
    target_column = setting_dict["TARGET_COLUMN"]
    num_epochs = setting_dict["NUM_EPOCHS"]
    num_epochs_rdlm = num_epochs // 4  # rDLM訓練用のエポック数
    batch_size = setting_dict["BATCH_SIZE"]
    num_fold = setting_dict["NUM_FOLD"]

    # モデルとデータの読み込み先のディレクトリ
    data_dir = f"/src/data/{task_name}/{exp_name}"
    model_dir = f"/src/models/{task_name}/{exp_name}"

    # rDLMの保存先として, models/以下の各設定のディレクトリにrDLM用のディレクトリを作る
    rdlm_dir = os.path.join(model_dir, "rDLM")

    # iDLMやrDLMsのロード
    for k in range(num_fold):
        logger.info(f"processing fold {k}...")

        # iDLMのロード (こいつの重みをrDLMの重みを使って変える)
        idlm = select_model(task_name=task_name)
        idlm_path = os.path.join(model_dir, f"trained_model_fold-{k}.pt")
        idlm.load_state_dict(torch.load(idlm_path))
        idlm.eval()

        # foldに対するdataloaderをロード
        train_data_path = os.path.join(data_dir, f"train_loader_fold-{k}.pt")
        train_loader = torch.load(train_data_path)
        repair_data_path = os.path.join(data_dir, f"repair_loader_fold-{k}.pt")
        repair_loader = torch.load(repair_data_path)

        # rDLMを作って訓練して保存することを一定数繰り返す(randomnessの排除のため)
        for rep in range(num_reps):
            logger.info(f"starting rep {rep}...")
            rdlm_list = []
            # rDLMの読み込み
            for rdlm_idx in range(rDLM_num):
                logger.info(f"loading rdlm_idx {rdlm_idx}...")
                rdlm_save_dir = os.path.join(rdlm_dir, f"rep{rep}")

                # rDMLのロード
                rdlm = select_model(task_name=task_name)
                rdlm_path = os.path.join(rdlm_save_dir, f"trained_model_fold-{k}_rDLM-{rdlm_idx}.pt")
                rdlm.load_state_dict(torch.load(rdlm_path))
                rdlm.eval()
                rdlm_list.append(rdlm)
            logger.info(f"len(rdlm_list)={len(rdlm_list)}")

            # ===================================
            # actual process
            # ===================================
            logger.info("starting Apricot actual process...")
            base_acc = eval_model(idlm, repair_loader)["metrics"][0]  # 最後にimprovementを表示したいのでここでbaseのaccを保存しておく
            best_acc = base_acc  # 暫定的なbest_acc
            logger.info(f"initial iDLM repair acc.={base_acc}")
            with torch.no_grad():
                best_weights = list(map(lambda x: x.data, idlm.parameters()))

            # early stoppingのために使用
            last_improvement = 0

            s = time.clock()
            logger.info(f"Start time: {s}")

            # train loaderからバッチを取り出して処理を実行
            for b_idx, (x, x_class) in enumerate(train_loader):
                with torch.no_grad():
                    yOrigin = torch.argmax(idlm(x), dim=1)  # バッチの各サンプルに対する予測結果. 形状は(batch_size, )
                    ySubList = [
                        torch.argmax(rdlm(x), dim=1) for rdlm in rdlm_list
                    ]  # 各rDLMのバッチの各サンプルに対する予測結果. 形状は(rdlm_num, batch_size).

                    # バッチ内の各サンプルに対して実行
                    for i_idx, (_, y) in enumerate(zip(x, x_class)):
                        if yOrigin[i_idx] == y:  # iDLMの予測が正しい
                            continue
                        else:
                            # 予測が合ってたrDLM
                            correctSubModels = [
                                rdlm for r_idx, rdlm in enumerate(rdlm_list) if ySubList[r_idx][i_idx] == y
                            ]
                            # 予測がちがったrDLM
                            incorrectSubModels = [
                                rdlm for r_idx, rdlm in enumerate(rdlm_list) if ySubList[r_idx][i_idx] != y
                            ]
                            # 全てのrDLMが正解したor不正解だった場合はスキップ
                            if len(correctSubModels) == 0 or len(incorrectSubModels) == 0:
                                continue  # slightly different from paper

                            # 予測が合ってたrDLMの重み平均
                            correctWeightSum = [sum(t) for t in zip(*[m.parameters() for m in correctSubModels])]
                            correctWeights = [e / len(correctSubModels) for e in correctWeightSum]
                            # 予測がちがったrDLMの重み平均
                            incorrWeightSum = [sum(t) for t in zip(*[m.parameters() for m in incorrectSubModels])]
                            incorrWeights = [e / len(incorrectSubModels) for e in incorrWeightSum]
                            baseWeights = list(map(lambda x: x.data, idlm.parameters()))
                            # 現在の重みとの差分
                            corrDiff = [b_w - c_w for b_w, c_w in zip(baseWeights, correctWeights)]
                            incorrDiff = [b_w - i_w for b_w, i_w in zip(baseWeights, incorrWeights)]

                            # 重みの調整を行う
                            baseWeights = AdjustWeights(
                                baseWeights,
                                corrDiff,
                                incorrDiff,
                                len(correctSubModels),
                                len(incorrectSubModels),
                                strategy="both-org",
                                lr=1e-3,
                            )
                            setWeights(idlm, baseWeights)

                # trainに使ってないデータセットでaccを確認
                curr_acc = eval_model(idlm, repair_loader)["metrics"][0]

                # b_idxのバッチでadjust後にaccが向上した場合
                if best_acc < curr_acc:
                    # bestの重みやaccを更新
                    with torch.no_grad():
                        best_weights = list(map(lambda x: x.data, idlm.parameters()))
                    best_acc = curr_acc
                    last_improvement = b_idx
                # 向上しなかった場合 (bestが更新されない場合)
                else:
                    # 悪くなった場合は現在のbestの重みに戻す
                    if best_acc != curr_acc:
                        with torch.no_grad():
                            setWeights(idlm, best_weights)
                    # バッチ100回分で向上が見られなかったら打ち切り
                    if last_improvement + 100 < b_idx:
                        logger.info("No improvement for too long, terminating")
                        break
                logger.info(f"batch {b_idx} done, last improvement {b_idx-last_improvement} batches ago.")
                # 短いエポックで再度訓練する
                train_model(idlm, train_loader, num_epochs=20)
                # 再びaccを確認
                curr_acc = eval_model(idlm, repair_loader)["metrics"][0]
                logger.info(f"new accuracy post training: {curr_acc}")

            # 時間計測
            e = time.clock()
            logger.info(f"End time: {e}")
            logger.info(f"Total execution time: {e-s}")
            # repair setに対するaccがどれほど伸びたかを出力
            logger.info(f"improvement={best_acc}-{base_acc}={best_acc - base_acc}")

            # 最終的なbestの重みをidlmの重みにセットして保存する
            setWeights(idlm, best_weights)
            weight_save_dir = os.path.join(model_dir, "apricot-weight", f"rep{rep}")
            os.makedirs(weight_save_dir, exist_ok=True)
            torch.save(idlm.state_dict(), os.path.join(weight_save_dir, f"adjusted_weights_fold-{k}.pt"))
            logger.info(f"saved to {os.path.join(weight_save_dir, f'adjusted_weights_fold-{k}.pt')}")
