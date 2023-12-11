import time, pickle, os, sys
from tqdm import tqdm

sys.path.append("../../lib")
from lib.util import keras_lid_to_torch_layers

# for logging
from logging import getLogger

logger = getLogger("base_logger")

import numpy as np
import torch


class Searcher(object):
    """docstring for Searcher"""

    np = __import__("numpy")
    os = __import__("os")
    importlib = __import__("importlib")
    kfunc_util = importlib.import_module("src_arachne.utils.kfunc_util")
    model_util = importlib.import_module("src_arachne.utils.model_util")
    data_util = importlib.import_module("src_arachne.utils.data_util")
    gen_frame_graph = importlib.import_module("src_arachne.utils.gen_frame_graph")

    def __init__(
        self,
        inputs,
        labels,
        indices_to_correct,
        indices_to_wrong,
        num_label,
        indices_to_target_layers,
        task_name,
        device,
        max_search_num=200,
        model=None,
        batch_size=None,
        is_multi_label=True,
        is_lstm=False,
        len_for_repair=None,
    ):
        """ """
        super(Searcher, self).__init__()
        self.device = device

        # data related initialisation
        self.num_label = num_label
        self.inputs = inputs
        self.lens = len_for_repair
        self.task_name = task_name
        from collections.abc import Iterable

        # ラベルの設定
        if is_multi_label:
            # ラベルが1次元配列になっているのでonehotベクトル化(2次元になる)
            if not isinstance(labels[0], Iterable):
                from src_arachne.utils.data_util import format_label

                self.ground_truth_labels = labels
                self.labels = format_label(labels, self.num_label)
            else:
                self.labels = labels
                self.ground_truth_labels = self.np.argmax(self.labels, axis=1)
        else:
            self.labels = labels
            self.ground_truth_labels = labels
        if self.lens is not None:
            logger.info(f"inputs.shape: {self.inputs.shape}, labels.shape : {self.labels.shape}, lens.shape : {self.lens.shape}")
        else:
            logger.info(f"inputs.shape: {self.inputs.shape}, labels.shape : {self.labels.shape}")


        self.mdl = model
        self.is_lstm = is_lstm

        self.indices_to_correct = indices_to_correct
        self.indices_to_wrong = indices_to_wrong

        # model related initialisation
        # self.path_to_keras_model = path_to_keras_model
        self.indices_to_target_layers = indices_to_target_layers
        self.targeted_layer_names = None
        self.batch_size = batch_size
        self.is_multi_label = is_multi_label

        self.maximum_fitness = 0.0  # the maximum fitness value

        # kerasでのレイヤインデックスとtorchのレイヤとの対応
        self.dic_keras_lid_to_torch_layers = keras_lid_to_torch_layers(task_name=self.task_name, model=self.mdl)

        # set target weights
        self.set_target_weights()
        # set chunks
        self.chunks = self.data_util.return_chunks(len(self.inputs), batch_size=self.batch_size)

        # set search relate parameters
        self.max_search_num = max_search_num
        self.indices_to_sampled_correct = None

    def set_target_weights(self):
        """
        Store the weights of the target layers
        """
        self.init_weights = {}
        self.init_biases = {}
        for idx_to_tl in self.indices_to_target_layers:
            tl = self.dic_keras_lid_to_torch_layers[idx_to_tl]
            lname = tl.__class__.__name__
            if lname == "Conv2d" or lname == "Linear":
                self.init_weights[idx_to_tl] = tl.weight.cpu().detach().numpy()
                self.init_biases[idx_to_tl] = tl.bias.cpu().detach().numpy()
            elif lname == "LSTM":
                for i, (name, param) in enumerate(tl.named_parameters()):
                    # logger.info(f"name={name}, param.shape={param.shape}")
                    self.init_weights[(idx_to_tl, i)] = param.cpu().detach().numpy()
            else:
                print("Not supported layer: {}".format(lname))
                assert False

    def move(self, deltas):
        """
        *** should be checked and fixed
        """

        labels = self.labels

        # deltasをモデルにセットして予測ラベルのリストとロスのリストを得る
        predictions, losses_of_all = self.predict_with_new_delta(deltas)

        if len(predictions.shape) > len(labels.shape) and predictions.shape[1] == 1:
            predictions = self.np.squeeze(predictions, axis=1)

        if self.is_multi_label:
            correct_predictions = predictions == self.np.argmax(labels, axis=1)
        else:
            correct_predictions = self.np.round(predictions).flatten() == labels

        # 以下で目的関数の計算（論文のeq3）
        losses_of_correct = losses_of_all[self.indices_to_correct]
        indices_to_corr_false = self.np.where(correct_predictions[self.indices_to_correct] == 0.0)[0]
        num_corr_true = len(self.indices_to_correct) - len(indices_to_corr_false)
        new_losses_of_correct = num_corr_true + self.np.sum(1 / (losses_of_correct[indices_to_corr_false] + 1))

        losses_of_wrong = losses_of_all[self.indices_to_wrong]
        indices_to_wrong_false = self.np.where(correct_predictions[self.indices_to_wrong] == 0.0)[0]
        num_wrong_true = len(self.indices_to_wrong) - len(indices_to_wrong_false)
        new_losses_of_wrong = num_wrong_true + self.np.sum(1 / (losses_of_wrong[indices_to_wrong_false] + 1))

        combined_losses = (new_losses_of_correct, new_losses_of_wrong)
        # print (self.is_multi_label,
        # self.is_lstm, self.num_label, num_corr_true,
        # self.np.sum(1/(losses_of_correct[indices_to_corr_false] + 1)),
        # num_wrong_true, self.np.sum(1/(losses_of_wrong[indices_to_wrong_false] + 1)))
        return combined_losses

    def predict_with_new_delta(self, deltas):
        """
        predict with the model patched using deltas
        """
        from collections.abc import Iterable

        # prepare a new model to run by updating the weights from deltas

        # we only have a one model as this one accept any lenghts of an input,
        # which is actually the output of the previous layers
        for idx_to_tl, delta in deltas.items():  # either idx_to_tl or (idx_to_tl, i)
            if isinstance(idx_to_tl, Iterable):
                idx_to_t_mdl_l, idx_to_w = idx_to_tl
            else:
                idx_to_t_mdl_l = idx_to_tl

            tl = self.dic_keras_lid_to_torch_layers[idx_to_t_mdl_l]
            lname = tl.__class__.__name__
            if lname == "Conv2d" or lname == "Linear":
                tl.weight.data = torch.from_numpy(delta).to(self.device)
                # tl.set_weights([delta, self.init_biases[idx_to_t_mdl_l]])
            elif lname == "LSTM":
                for i, tp in enumerate(tl.parameters()):
                    if i == idx_to_w:
                        tp.data = torch.from_numpy(delta).to(self.device)
                # if idx_to_w == 0:  # kernelを新しくする
                #     new_kernel_w = delta  # use the full
                #     new_recurr_kernel_w = self.init_weights[(idx_to_t_mdl_l, 1)]
                # elif idx_to_w == 1: # recurrent kernelを新しくする
                #     new_recurr_kernel_w = delta
                #     new_kernel_w = self.init_weights[(idx_to_t_mdl_l, 0)]
                # else:
                #     print("{} not allowed".format(idx_to_w), idx_to_t_mdl_l, idx_to_tl)
                #     assert False
                # set kernel, recurr kernel, bias
                # fn_mdl.layers[idx_to_t_mdl_l].set_weights(
                #     [new_kernel_w, new_recurr_kernel_w, self.init_biases[idx_to_t_mdl_l]]
                # )
            else:
                print("{} not supported".format(lname))
                assert False

        # 各サンプルに対する損失をnumpy配列で取得
        losses_of_all = []
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none") # NOTE: バッチ内の各サンプルずつのロスを出すため. デフォルトのreduction="mean"だとバッチ内の平均になってしまう
        # NOTE: 一件ずつ取りだしてやるとめっちゃ遅いのでバッチ化
        y_preds, y_trues = [], []
        for chunk in self.chunks:
            # chunkを使ってバッチを取り出す
            i = self.inputs[chunk]
            l = self.ground_truth_labels[chunk]
            if self.is_lstm:
                lens = self.lens[chunk]
                # バッチへの予測
                out = self.mdl.predict(torch.from_numpy(i).to(self.device), lens, device=self.device)
                pred, prob = out["pred"].cpu(), out["prob"].cpu()
            else:
                # バッチへの予測
                out = self.mdl.predict(torch.from_numpy(i).to(self.device), device=self.device)
                pred, prob = out["pred"].cpu(), out["prob"].cpu()
            # ロスの計算
            loss = loss_fn(prob, torch.from_numpy(l)).cpu().detach().numpy()
            # 予測ラベルと真のラベル
            y_preds.append(pred)
            y_trues.append(l)
            # ロスの配列
            losses_of_all.append(loss)
        y_preds = np.concatenate(y_preds, axis=0)
        y_trues = np.concatenate(y_trues, axis=0)
        losses_of_all = np.concatenate(losses_of_all, axis=0)
        return y_preds, losses_of_all

    def move_lstm(self, deltas):
        """
        *** should be checked and fixed
        --> need to fix this...
        delatas -> key: idx_to_tl & inner_key: index to the weight
                        or key: (idx_to_tl, i) & inner_key
                        value -> the new value
        """
        # import time

        labels = self.labels
        predictions = self.predict_with_new_delta(deltas)
        # print (predictions)
        # due to the data dimention of fashion_mnist (..)
        if predictions.shape != labels.shape:
            to_this_shape = labels.shape
            predictions = self.np.reshape(predictions, to_this_shape)

        # t1 = time.time()
        if self.is_multi_label:
            correct_predictions = self.np.argmax(predictions, axis=1)
            correct_predictions = correct_predictions == self.np.argmax(labels, axis=1)
        else:
            correct_predictions = self.np.round(predictions).flatten() == labels

        # set self.k_fn_loss if it has been done yet.
        if self.k_fn_loss is None:
            loss_func = self.model_util.get_loss_func(is_multi_label=self.is_multi_label)
            self.k_fn_loss = self.kfunc_util.gen_pred_and_loss_ops(
                predictions.shape, predictions.dtype, labels.shape, labels.dtype, loss_func
            )

        losses_of_all = self.k_fn_loss([predictions, labels])[0]
        # t2 = time.time()
        # print ("Time for pred prob and loss: {}".format(t2 - t1))

        losses_of_correct = losses_of_all[self.indices_to_correct]
        indices_to_corr_false = self.np.where(correct_predictions[self.indices_to_correct] == 0.0)[0]
        num_corr_true = len(self.indices_to_correct) - len(indices_to_corr_false)
        new_losses_of_correct = num_corr_true + self.np.sum(1 / (losses_of_correct[indices_to_corr_false] + 1))

        losses_of_wrong = losses_of_all[self.indices_to_wrong]
        indices_to_wrong_false = self.np.where(correct_predictions[self.indices_to_wrong] == 0.0)[0]
        num_wrong_true = len(self.indices_to_wrong) - len(indices_to_wrong_false)
        new_losses_of_wrong = num_wrong_true + self.np.sum(1 / (losses_of_wrong[indices_to_wrong_false] + 1))

        combined_losses = (new_losses_of_correct, new_losses_of_wrong)
        return predictions, correct_predictions, combined_losses

    def get_results_of_target(self, deltas, indices_to_target, mode="target"):
        """
        Return the results of the target (can be accessed by indices_to_target)
                -> results are compute for currnet self.mdl
        Ret (int, float):
                int: the number of patched
                float: percentage of the number of patched)
        """
        # 引数modeに対応するモデルリストとラベルを返すための辞書
        mode_dict = {
            "target": (self.fn_mdl_lst, self.labels),
            "train": (self.fn_mdl_lst_train, self.y_train),
            "repair": (self.fn_mdl_lst_repair, self.y_repair),
            "test": (self.fn_mdl_lst_test, self.y_test),
        }

        mdl_list, labels = mode_dict[mode]

        deltas_as_lst = [deltas[idx_to_tl] for idx_to_tl in self.indices_to_target_layers]
        # ここで予測値を計算
        predictions = self.kfunc_util.compute_predictions(mdl_list, labels, deltas_as_lst, batch_size=self.batch_size)

        if self.is_multi_label:
            correct_predictions = self.np.argmax(predictions, axis=-1)
            y_labels = self.np.argmax(labels, axis=1)
            if correct_predictions.shape != y_labels.shape:
                correct_predictions = correct_predictions.reshape(y_labels.shape)
            correct_predictions = correct_predictions == y_labels
        else:
            correct_predictions = self.np.round(predictions).flatten() == labels

        if indices_to_target == "all":
            indices_to_target = list(range(len(correct_predictions)))

        target_corr_predcs = correct_predictions[indices_to_target]
        num_of_total_target = len(target_corr_predcs)
        msg = "%d vs %d" % (num_of_total_target, len(indices_to_target))
        assert num_of_total_target == len(indices_to_target), msg
        correctly_classified = self.np.sum(target_corr_predcs)

        return correctly_classified, correctly_classified / num_of_total_target, correct_predictions.astype(int)

    def get_results_of_target_lstm(self, deltas, indices_to_target):
        """
        Return the results of the target (can be accessed by indices_to_target)
                -> results are compute for currnet self.mdl
        Ret (int, float):
                int: the number of patched
                float: percentage of the number of patched)
        """
        predictions = self.predict_with_new_delta(deltas)
        if predictions.shape != self.labels.shape:
            to_this_shape = self.labels.shape
            predictions = self.np.reshape(predictions, to_this_shape)

        if self.is_multi_label:
            correct_predictions = self.np.argmax(predictions, axis=1)
            correct_predictions = correct_predictions == self.np.argmax(self.labels, axis=1)
        else:
            correct_predictions = self.np.round(predictions).flatten() == self.labels

        target_corr_predcs = correct_predictions[indices_to_target]

        num_of_total_target = len(target_corr_predcs)
        msg = "%d vs %d" % (num_of_total_target, len(indices_to_target))
        assert num_of_total_target == len(indices_to_target), msg
        correctly_classified = self.np.sum(target_corr_predcs)

        return (correctly_classified, correctly_classified / num_of_total_target)

    def get_number_of_patched(self, deltas):
        """
        Return a number of patched initially wrongly classified outputs
        (can be accessed by self.indices_to_wrong)
        => compute for currnent self.mdl
        Ret (int, float):
                int: the number of patched
                float: percentage of the number of patched)
        """
        if self.is_lstm:
            correctly_classified, perc_correctly_classifed = self.get_results_of_target_lstm(
                deltas, self.indices_to_wrong
            )
        else:
            correctly_classified, perc_correctly_classifed, _ = self.get_results_of_target(
                deltas, self.indices_to_wrong
            )

        return (correctly_classified, perc_correctly_classifed)

    def get_number_of_violated(self, deltas):
        """
        Return a number of patched initially correctly classified outputs
        (can be accessed by self.indices_to_correct)
        => compute for currnent self.mdl
        Ret (int, float):
                int: the number of patched
                float: percentage of the number of patched)
        """
        target_indices = self.indices_to_correct
        if self.is_lstm:
            correctly_classified, perc_correctly_classifed = self.get_results_of_target_lstm(deltas, target_indices)
        else:
            correctly_classified, perc_correctly_classifed, _ = self.get_results_of_target(deltas, target_indices)

        num_of_initially_correct = len(target_indices)
        return (num_of_initially_correct - correctly_classified, 1.0 - perc_correctly_classifed)

    def get_num_patched_and_broken(self, predictions):
        """
        compute fitness using a given predictions and correctly
        Ret (boolean, float):
                boolean: True if restriction is not violated, otherwise, False
                float: new fitness value
        """
        if self.is_multi_label:
            new_classifcation_results = self.np.argmax(predictions, axis=1)
        else:
            new_classifcation_results = self.np.round(predictions).flatten()

        # correct -> incorrect
        num_violated = self.np.sum((new_classifcation_results != self.ground_truth_labels)[self.indices_to_correct])
        # incorrect (wrong) -> corret
        num_patched = self.np.sum((new_classifcation_results == self.ground_truth_labels)[self.indices_to_wrong])

        return num_violated, num_patched

    def check_early_stop(self, fitness_value, new_weights, model_name=None):
        """
        Check whether early stop is possible or not
        Arguments:
                model_name: the name of model to examine
        Ret (bool):
                True (early stop)
                False (not yet)
        """

        num_of_patched, perc_num_of_patched = self.get_number_of_patched(new_weights)
        num_of_violated, perc_num_of_violated = self.get_number_of_violated(new_weights)

        if num_of_patched == len(self.indices_to_wrong) and num_of_violated == 0:
            print("In early stop checking:%d, %d" % (num_of_patched, num_of_violated))
            print("\t fitness values", fitness_value)
            print("\t", num_of_patched == len(self.indices_to_wrong) and num_of_violated == 0)
            return True, num_of_patched
        else:
            print(
                "in early stop checking",
                "{} ({}), {} ({})".format(num_of_patched, perc_num_of_patched, num_of_violated, perc_num_of_violated),
            )
            return False, num_of_patched

    def summarise_results(self, deltas):
        """
        Print out the current result of model_name
        => compute for currnent self.mdl
        """
        logger.info("***Patching results***")

        num_of_patched, perc_num_of_patched = self.get_number_of_patched(deltas)
        num_of_violated, perc_num_of_violated = self.get_number_of_violated(deltas)

        logger.info(
            "For initially wrongly classified:%d -> %d(%f)"
            % (len(self.indices_to_wrong), len(self.indices_to_wrong) - num_of_patched, perc_num_of_patched)
        )
        logger.info(
            "For initially correctly classified(violation):%d -> %d(%f)"
            % (len(self.indices_to_correct), len(self.indices_to_correct) - num_of_violated, perc_num_of_violated)
        )

        is_corr = {}
        # 修正後のモデルを使ってtrain, repair, testの各データを予測してみる
        # NOTE: もしメモリエラーになるようならdivの部分を引数で取るようにして1つずつ実行するようにする
        for div in ["train", "repair", "test"]:
            num_corr, acc, is_corr_arr = self.get_results_of_target(deltas=deltas, indices_to_target="all", mode=div)
            logger.info(f"({div}) num_corr : {num_corr}, acc : {acc} = {num_corr}/{len(is_corr_arr)}")
            is_corr[div] = is_corr_arr
        return is_corr
