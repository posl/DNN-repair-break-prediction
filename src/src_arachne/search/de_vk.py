"""
Use differential evolution
"""
import pickle
from src_arachne.search.searcher_vk import Searcher

# for logging
from logging import getLogger

logger = getLogger("base_logger")


class DE_searcher(Searcher):
    """docstring for DE_searcher"""

    random = __import__("random")

    importlib = __import__("importlib")
    base = importlib.import_module("deap.base")
    creator = importlib.import_module("deap.creator")
    tools = importlib.import_module("deap.tools")

    logger.info("Initializing DE_searcher...")

    def __init__(
        self,
        inputs,
        labels,
        indices_to_correct,
        indices_to_wrong,
        num_label,
        indices_to_target_layers,
        mutation=(0.5, 1),
        recombination=0.7,
        max_search_num=100,
        initial_predictions=None,
        model=None,
        patch_aggr=None,
        batch_size=None,
        act_func=None,
        is_lstm=False,
        is_multi_label=True,
        X_train=None,
        X_repair=None,
        X_test=None,
        y_train=None,
        y_repair=None,
        y_test=None,
    ):
        """ """
        super(DE_searcher, self).__init__(
            inputs,
            labels,
            indices_to_correct,
            indices_to_wrong,
            num_label,
            indices_to_target_layers,
            max_search_num=max_search_num,
            initial_predictions=initial_predictions,
            model=model,
            batch_size=batch_size,
            act_func=act_func,
            is_lstm=is_lstm,
            is_multi_label=is_multi_label,
            X_train=X_train,
            X_repair=X_repair,
            X_test=X_test,
            y_train=y_train,
            y_repair=y_repair,
            y_test=y_test,
        )

        # fitness computation related initialisation
        self.fitness = 0.0

        # deap関連の設定
        self.creator.create("FitnessMax", self.base.Fitness, weights=(1.0,))  # maximisation
        self.creator.create("Individual", self.np.ndarray, fitness=self.creator.FitnessMax, model_name=None)

        # store the best performace seen so far
        self.the_best_performance = None
        self.max_num_of_unchanged = int(self.max_search_num / 10) if self.max_search_num is not None else None

        if self.max_num_of_unchanged < 10:
            self.max_num_of_unchanged = 10

        self.num_iter_unchanged = 0

        # DE specific
        self.mutation = mutation
        self.recombination = recombination

        # fitness
        self.patch_aggr = patch_aggr

        logger.info("Finish Initializing DE_searcher...")

    def eval(self, patch_candidate, places_to_fix):
        """
        Evaluate a given trial vector
        Use a given part, a new weight value vector, to update
        a target tensor, here, weight (& bias), and evaluate a new DNN model
        wiht the updated weight tensor

        Argumnets:
                patch_candidate: a trial candiate vector
                places_to_fix: [(idx_to_tl, inner_indices) .. ]
        Ret: (fitness)
        """
        deltas = {}  # this is deltas for set update op
        for i, (idx_to_tl, inner_indices) in enumerate(places_to_fix):
            if idx_to_tl not in deltas.keys():
                deltas[idx_to_tl] = self.init_weights[
                    idx_to_tl
                ]  ## *** HAVE CHANGED TO ACCEPT (idx_to_tl, idx_to_w(0 or 1))
            # since our op is set
            deltas[idx_to_tl][tuple(inner_indices)] = patch_candidate[i]

        # 以下で目的関数の計算（論文のeq3）
        if not self.lstm_mdl:
            predictions, _, loss_v = self.move(deltas, update_op="set")
        else:
            # here, we don't have to be worry about the corre_predictins -> alrady done in move_v3.
            predictions, _, loss_v = self.move_lstm(deltas)
        # assert predictions is not None

        # 以下で目的関数の計算（論文のeq4）
        losses_of_correct, losses_of_target = loss_v
        if self.patch_aggr is None:
            final_fitness = losses_of_correct + losses_of_target
        else:
            final_fitness = losses_of_correct + self.patch_aggr * losses_of_target

        # logging
        # num_violated, num_patched = self.get_num_patched_and_broken(predictions) # logging
        # num_still_correct = len(self.indices_to_correct) - num_violated # N_intac
        # print ("New fitness (num_violatd, num_patched)",
        # 	self.patch_aggr, losses_of_correct, losses_of_target, num_violated, num_patched, final_fitness)
        return (final_fitness,)

    def is_the_performance_unchanged(self, curr_best_patch_candidate):
        """
        curr_best_performance: fitness of curr_best_patch_candidate:
        Ret (bool):
                True: the performance
        """
        curr_best_performance = curr_best_patch_candidate.fitness.values[0]

        if self.the_best_performance is None:
            self.the_best_performance = curr_best_performance
            return False

        if self.np.float32(curr_best_performance) == self.np.float32(self.the_best_performance):
            self.num_iter_unchanged += 1
        else:
            self.num_iter_unchanged = 0  # look for subsequent
            if curr_best_performance > self.the_best_performance:
                self.the_best_performance = curr_best_performance

        if self.max_num_of_unchanged < self.num_iter_unchanged:
            return True
        else:
            return False

    def set_bounds(self, init_weight_value):
        """ """
        min_v = self.np.min(init_weight_value)
        min_v = min_v * 2 if min_v < 0 else min_v / 2

        max_v = self.np.max(init_weight_value)
        max_v = max_v * 2 if max_v > 0 else max_v / 2

        bounds = (min_v, max_v)
        return bounds

    def search(self, places_to_fix, save_path):
        """ """
        import time

        # ターゲットとなるレイヤ達の重みの平均や分散を取得
        num_places_to_fix = len(places_to_fix)  # the number of places to fix # NDIM of a patch candidate
        bounds = []
        self.mean_values = []
        self.std_values = []
        for idx_to_tl, _ in places_to_fix:  # ** HAVE CHANGED TO HANDLE (idx_to_tl, idx_to_w (0 or 1))
            _init_weight = self.init_weights[idx_to_tl]
            mean_value = self.np.mean(_init_weight)
            self.mean_values.append(mean_value)
            std_value = self.np.std(_init_weight)
            self.std_values.append(std_value)
            bounds.append(self.set_bounds(_init_weight))  # for clipping

        # set search parameters
        pop_size = 100
        toolbox = self.base.Toolbox()

        # 初期値はターゲットレイヤの重みの平均と分散を用いて正規分布からサンプリング (len(places_to_fix)個をサンプリング)
        def init_indiv():
            v_sample = lambda mean_v, std_v: self.np.random.normal(loc=mean_v, scale=std_v, size=1)[0]
            ind = self.np.float32(list(map(v_sample, self.mean_values, self.std_values)))
            return ind

        # DEのためのパラメータ設定
        toolbox.register("individual", self.tools.initIterate, self.creator.Individual, init_indiv)
        toolbox.register("population", self.tools.initRepeat, list, toolbox.individual)
        toolbox.register("select", self.np.random.choice, size=3, replace=False)
        toolbox.register("evaluate", self.eval)

        # set logbook
        stats = self.tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", self.np.mean)
        stats.register("std", self.np.std)
        stats.register("min", self.np.min)
        stats.register("max", self.np.max)

        logbook = self.tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields
        # logbook setting end

        # ここで，長さplaces_to_fixの初期値ベクトルをpop_size個生成
        pop = toolbox.population(n=pop_size)

        hof = self.tools.HallOfFame(1, similar=self.np.array_equal)

        # update fitness
        # print ("Places to fix", places_to_fix)

        # 各個体のfitnessを計算
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind, places_to_fix)
            ind.model_name = None

        # 初期のベスト（暫定）を更新
        hof.update(pop)
        best = hof[0]
        best.model_name = "initial"

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(pop), **record)

        search_start_time = time.time()
        # search start
        import time

        # 世代の繰り返し
        for iter_idx in range(self.max_search_num):
            iter_start_time = time.time()
            MU = self.random.uniform(self.mutation[0], self.mutation[1])

            # 各個体について繰り返し
            for pop_idx, ind in enumerate(pop):
                t0 = time.time()
                # set model name
                new_model_name = "iter{}-pop{}".format(iter_idx, pop_idx)

                # select
                target_indices = [_i for _i in range(pop_size) if _i != pop_idx]
                a_idx, b_idx, c_idx = toolbox.select(target_indices)
                a = pop[a_idx]
                b = pop[b_idx]
                c = pop[c_idx]

                y = toolbox.clone(ind)

                index = self.random.randrange(num_places_to_fix)
                for i, value in enumerate(ind):
                    # 一部を変異させる
                    if i == index or self.random.random() < self.recombination:
                        # パッチ候補の値は対象レイヤの重みから計算したバウンドに収める
                        y[i] = self.np.clip(a[i] + MU * (b[i] - c[i]), bounds[i][0], bounds[i][1])

                y.fitness.values = toolbox.evaluate(y, places_to_fix)
                if y.fitness.values[0] >= ind.fitness.values[0]:  # better
                    pop[pop_idx] = y  # upddate
                    # set new model name
                    pop[pop_idx].model_name = new_model_name
                    # update best
                    if best.fitness.values[0] < pop[pop_idx].fitness.values[0]:
                        hof.update(pop)
                        best = hof[0]
                        # print ("New best one is set: {},
                        # fitness: {}, model_name: {}".format(
                        # best, best.fitness.values[0], best.model_name))

            # 全population終わったらその世代でのベストのパッチ候補を表示
            hof.update(pop)
            best = hof[0]

            # うるさくなるのでコメントアウト
            # logger.info(
            # f"[The best at Gen {iter_idx}] fitness={best.fitness.values[0]} at X_best={best}, model_name: {best.model_name}"
            # )

            # logging for this generation
            record = stats.compile(pop)
            logbook.record(gen=iter_idx, evals=len(pop), **record)

            #########################################################
            # update for best value to check for early stop #########
            #########################################################
            # レイヤのインデックスをキー，そのレイヤの修正後の重みを値とする辞書
            deltas = {}  # this is deltas for set update op
            for i, (idx_to_tl, inner_indices) in enumerate(places_to_fix):
                if idx_to_tl not in deltas.keys():
                    deltas[idx_to_tl] = self.init_weights[idx_to_tl]
                # since our op is set
                deltas[idx_to_tl][tuple(inner_indices)] = best[i]

            # check for two stop coniditions
            # ここでearly stopの判定を行う (fitnessが変化しないエポックが一定数続いたら終了)
            if self.is_the_performance_unchanged(best):
                logger.info("Performance has not been changed over {} iterations".format(self.num_iter_unchanged))
                break

            curr_time = time.time()
            local_run_time = curr_time - iter_start_time
            run_time = curr_time - search_start_time
            logger.info("Time for a single iter: {}, ({})".format(run_time, local_run_time))

        # with these two cases, the new model has not been saved
        # if self.empty_graph is not None:
        logger.info(f"best ind.: {best}, fitness: {best.fitness.values[0]}")

        # 重みの辞書更新
        deltas = {}  # this is deltas for set update op
        for i, (idx_to_tl, inner_indices) in enumerate(places_to_fix):
            if idx_to_tl not in deltas.keys():
                deltas[idx_to_tl] = self.init_weights[idx_to_tl]
            # since our op is set
            deltas[idx_to_tl][tuple(inner_indices)] = best[i]
        # pklに保存
        with open(save_path, "wb") as f:
            pickle.dump(deltas, f)
        logger.info("The model is saved to {}".format(save_path))

        # target, train ,repair, testデータに対して結果を確認
        is_corr_dic = self.summarise_results(deltas)
        return is_corr_dic
