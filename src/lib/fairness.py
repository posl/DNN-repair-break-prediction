import numpy as np
import torch

# for logging
from logging import getLogger

logger = getLogger("base_logger")


def get_discriminatory_instances_candidates(inst, sens_idx, sens_vals=None):
    """sensitive featureだけを変えたインスタンスのリストを返す.
    数値データの場合にはsensitive featureの列インデックス（int）と値の範囲を, カテゴリデータ(OHE済)に対してはsensitive featureの列インデックス（list）を指定.

    Args:
        inst (torch.Tensor): sensitive featureだけを変えたいインスタンス.
        sens_idx (int or list): 変えたいfeatureを表す. intの場合は数値変数として, listの場合はOHEされたカテゴリ変数として扱う.
        sens_vals (list): 数値変数であるsensitive featureのとりうる値のリスト.

    Returns:
        list: sensitive featureだけを変えたインスタンスのリスト.
    """
    # 返したいリスト
    inst_list = []

    # 数値データの場合（= 1列だけ指定された場合）
    if isinstance(sens_idx, int):
        # sens_valsを渡す必要があるのでsens_valsの確認
        assert sens_vals is not None, "Error: sens_vals is None."

        for sv in sens_vals:
            # sensitive featureの同じ値が来たらスキップ
            if sv == inst[sens_idx]:
                continue
            # 元のtensorをクローン（値渡し）
            _inst = inst.clone()
            # sensitive featureの値だけ変える
            _inst[sens_idx] = sv
            inst_list.append(_inst)
        return inst_list

    # カテゴリデータの場合（=複数列を指定された場合 NOTE: カテゴリデータはone-hot encodingされてる前提）
    elif isinstance(sens_idx, list):
        # sens_idxの長さが2以上なのを確認
        assert len(sens_idx) >= 2, f"Error: len(sens_idx)={len(sens_idx)}, but len(sens_idx) should be greater than 2."

        for bit in range(len(sens_idx)):
            # 指定したsensitive featureのOHE vectorのbit番目が1の場合（=元のinstと同じvector）はスキップ
            if inst[sens_idx][bit] == 1:
                continue
            # 元のtensorをクローン（値渡し）
            _inst = inst.clone()
            # 新たなOHE vector用のtensorを生成しbit番目だけ1にする
            sv = torch.zeros(len(sens_idx), dtype=inst[sens_idx].dtype)
            sv[bit] = 1
            # sensitive featureの値だけ変える
            _inst[sens_idx] = sv
            inst_list.append(_inst)
        return inst_list


def sm_fairness(
    model, dataloader, sens_idx, sens_vals=None, target_cls=1, is_repair=False, hvals=None, neuron_location=None
):
    """CARE論文で言われているindependence-basedのfairnessを各データに対して計算してlistにして返す.

    .. math::
    sm_{fair} = \max_{p \in P} |N_t(x_{ip}) - N_t(x^{\prime}_{ip})|

    Args:
        model (nn.Module): 対象のモデル.
        dataloader (DataLoader): 対象のデータセットのデータローダ.
        sens_idx (int): sensitive featureの列のインデックス（何列目か）.
        sens_vals (list, optional): 対象のsensitive featureのとりうる値. Defaults to None.
        target_cls (int, optional): 出力値を得る対象のクラス. Defaults to 1.
        is_repair (bool): repair後の重みを使って予測するかどうか. ここをTrueにした場合は必ずこの後ろの2つの引数も指定しなければならない.
        hvals (list of float): 修正後のニューロンの値のリスト. indexは第三引数のneuron_locationと対応.
        neuron_location (list of tuple(int, int)): 修正するニューロンの位置(レイヤ番号, ニューロン番号)を表すタプルのリスト.

    Returns:
        float: 各データの上式のsm_fairのリスト.
    """
    # 返したいリスト
    sm_fair_list = []
    dataset = dataloader.dataset
    for i, (inst, _) in enumerate(dataset):
        # サンプルに対するfairnessみたいな値（sensitive featureだけ変えた時の予測確率の変化の最大値）
        max_diff = 0.0
        if not is_repair:
            # 修正前の重みで予測. NOTE: .predict はバッチ想定なので .view が必要.
            o = model.predict(inst.view(1, -1))["prob"].view(-1)[target_cls]
        else:
            # is_repairがTrueなのにhvalsやneuron_locationがNoneならassertion errorにする.
            assert (
                hvals is not None and neuron_location is not None
            ), "despite is_repair=True, hvals and neuron_location are None!!!"
            # 修正後の重みで予測. NOTE: .predict はバッチ想定なので .view が必要.
            o = model.predict_with_repair(inst.view(1, -1), hvals, neuron_location)["prob"].view(-1)[target_cls]
        # sensitive featureだけを変えたインスタンスのリストを得る
        inst_disc_list = get_discriminatory_instances_candidates(inst=inst, sens_idx=sens_idx, sens_vals=sens_vals)
        # 変更後の各サンプルについて, 予測確率の差分を取得
        for inst_disc in inst_disc_list:
            if not is_repair:
                o_prime = model.predict(inst_disc.view(1, -1))["prob"].view(-1)[target_cls]
            else:
                o_prime = model.predict_with_repair(inst_disc.view(1, -1), hvals, neuron_location)["prob"].view(-1)[
                    target_cls
                ]
            # 予測確率の差の絶対値を取得
            tmp_diff = torch.abs(o - o_prime).item()
            # サンプルに対するマックスを更新
            if tmp_diff > max_diff:
                max_diff = tmp_diff
        sm_fair_list.append(max_diff)
    return np.array(sm_fair_list)


def eval_independence_fairness(
    model, dataloader, sens_idx, sens_vals=None, target_cls=1, is_repair=False, hvals=None, neuron_location=None
):
    """CARE論文で言われているindependence-basedのfairnessを計算する.

    .. math::
    y_{fair} = \frac{1}{|X|} \sum_{x_{ip} \in X} \max_{p \in P} |N_t(x_{ip}) - N_t(x^{\prime}_{ip})|

    Args:
        model (nn.Module): 対象のモデル.
        dataloader (DataLoader): 対象のデータセットのデータローダ.
        sens_idx (int): sensitive featureの列のインデックス（何列目か）.
        sens_vals (list, optional): 対象のsensitive featureのとりうる値. Defaults to None.
        target_cls (int, optional): 出力値を得る対象のクラス. Defaults to 1.

    Returns:
        上式のy_fair, サンプルごとのsm_fairのリスト
    """
    sm_fair_list = sm_fairness(model, dataloader, sens_idx, sens_vals, target_cls, is_repair, hvals, neuron_location)
    # データセット全体の平均として返す
    y_fair = sum(sm_fair_list) / len(sm_fair_list)
    return y_fair, sm_fair_list


def calc_average_causal_effect(
    model, dataloader, sens_idx, target_lid, target_nid, hvals, sens_vals=None, target_cls=1
):
    """対象モデルのニューロンに対するFLスコアとして, そのニューロンを変えた時のfairnessの変化の度合いを算出する.

    Args:
        model (nn.Module): 対象のモデル.
        dataloader (DataLoader): 対象のデータセットのデータローダ.
        sens_idx (int): sensitive featureの列のインデックス（何列目か）.
        target_lid (int): 対象のレイヤのインデックス（最初の全結合層が0）.
        target_nid (_type_): 対象レイヤ内の対象ニューロンのインデックス（最初が0）.
        hvals (list): 対象のニューロンを変化させる値のリスト.
        sens_vals (list, optional): 対象のsensitive featureのとりうる値. Defaults to None.
        target_cls (int, optional): 出力値を得る対象のクラス. Defaults to 1.

    Returns:
        list: ニューロンの値を変化させた時それぞれにおける, fairnessの値（データセット全体での平均）.
    """
    dataset = dataloader.dataset
    fairness_list = []

    # hvalの各要素に対するループ
    for hval in hvals:
        sum_diff = 0.0

        # dataset内の各データに対するループ
        for inst, _ in dataset:
            # サンプルに対するfairnessみたいな値（sensitive featureだけ変えた時の予測確率の変化の最大値）
            max_diff = 0.0
            # 元のサンプルへの予測確率
            # model.predictはバッチ入力を前提としているので形状の変換が必要
            pred_tensor = model.predict_with_intervention(inst.view(1, -1), hval, target_lid, target_nid)
            # target_clsに対する予測確率のみ取り出す
            o = pred_tensor["prob"].view(-1)[target_cls]
            # sensitive featureだけを変えたインスタンスのリストを得る
            inst_disc_list = get_discriminatory_instances_candidates(inst=inst, sens_idx=sens_idx, sens_vals=sens_vals)

            # 各サンプルに対してsensitive featureだけ変更した後のサンプルについて, 予測確率の差分を取得
            for inst_disc in inst_disc_list:
                pred_tensor = model.predict_with_intervention(inst_disc.view(1, -1), hval, target_lid, target_nid)
                o_prime = pred_tensor["prob"].view(-1)[target_cls]
                # 予測確率の差の絶対値を取得
                tmp_diff = torch.abs(o - o_prime).item()
                # サンプルに対するマックスを更新
                if tmp_diff > max_diff:
                    max_diff = tmp_diff
            # データセット全体の合計を更新
            sum_diff += max_diff
        # データセット全体の平均として返す
        fairness_list.append(sum_diff / len(dataset))
    return fairness_list


def calc_fairness_ub(model, dataloader, sens_idx, hvals, neuron_location, sens_vals=None):
    """PSOのfitness functionの計算のために, Fairnessの観点でのUB (Unexpected Behavior) の度合いを返す.
    具体的には, P(N(x)!=N(x'))を算出して変えす.

    Args:
        model (nn.Module): 対象のモデル.
        dataloader (DataLoader): 対象のデータセットのデータローダ.
        sens_idx (int): sensitive featureの列のインデックス（何列目か）.
        hvals (list of float): 修正後のニューロンの値のリスト.
        neuron_location (list of tuple(int, int)): 修正するニューロンの位置(レイヤ番号, ニューロン番号)を表すタプルのリスト.
        sens_vals (list, optional): 対象のsensitive featureのとりうる値. Defaults to None.

    Returns:
        list: ニューロンの値を変化させた時それぞれにおける, fairnessの値（データセット全体での平均）.
    """
    dataset = dataloader.dataset
    # instanceごとのP(...)のリスト
    inst_diff_list = []

    # dataset内の各データに対するループ
    for inst, _ in dataset:
        # instanceごとのP(...)
        inst_diff = 0.0
        # サンプルに対するfairnessみたいな値（sensitive featureだけ変えた時の予測確率の変化の最大値）
        # 元のサンプルへの予測確率
        # model.predictはバッチ入力を前提としているので形状の変換が必要
        pred_dict = model.predict_with_repair(inst.view(1, -1), hvals, neuron_location)
        # target_clsに対する予測ラベルのみ取り出す
        o = pred_dict["pred"].item()
        # sensitive featureだけを変えたインスタンスのリストを得る
        inst_disc_list = get_discriminatory_instances_candidates(inst=inst, sens_idx=sens_idx, sens_vals=sens_vals)

        # 各サンプルに対してsensitive featureだけ変更した後のサンプルについて, 予測ラベルが異なってしまう割合を取得
        for inst_disc in inst_disc_list:
            pred_tensor = model.predict_with_repair(inst_disc.view(1, -1), hvals, neuron_location)
            o_prime = pred_tensor["pred"].item()
            # 予測がオリジナルのインスタンスと変わったかどうか
            if o != o_prime:
                inst_diff += 1
        inst_diff_list.append(inst_diff)
    return np.mean(inst_diff_list)


def calc_acc_average_causal_effect(model, dataloader, target_lid, target_nid, hvals, acc_org, device):
    acc_diff_list = []

    # hvalの各要素に対するループ
    for hval in hvals:
        sum_diff = 0.0
        acc_tmp = 0  # acc計算用
        # メモリ不足対策のためバッチに分けてaccを計算してからまとめる
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            ret_dicts = model.predict_with_intervention(data, hval, target_lid, target_nid, device)
            preds = ret_dicts["pred"].cpu()
            num_corr = sum(preds == labels)
            acc_tmp += num_corr / len(preds)
        # バッチごとのaccをまとめて全体のaccにする(NOTE: 除算の誤差がきになる)
        acc_tmp /= len(dataloader)
        # 各hvalにおけるaccuracyの差 (どれだけ悪くなったか) を計算し配列に入れていく
        acc_diff = acc_org - acc_tmp
        acc_diff_list.append(acc_diff)
    # print(acc_diff_list)
    return acc_diff_list
