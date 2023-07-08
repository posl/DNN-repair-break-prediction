import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from lib.log import set_exp_logging
from lib.util import json2dict


def purpose_col_transform(jid):
    """Purposeの列のカテゴリ数が多いので減らすための関数.

    Args:
        pid (int): 元データにおいてpurposeを表すid.

    Returns:
        int: purposeを表すidを, car related, home related, othersの3種類に分けたid (0, 1, 2のいずれか).
    """
    car_related_pids = [1, 2]
    home_related_pids = [3, 4, 5, 6]
    ohters_pids = [8, 9, 10, 0]
    if jid in car_related_pids:
        return 0
    elif jid in home_related_pids:
        return 1
    elif jid in ohters_pids:
        return 2


def round_fill(col, lpct, train_col):
    """
    外れ値をパーセンタイル値で置き換える処理．

    Parameters
    ------------------
    col (pd.Series): 置き換えを行いたい対象の列
    lpct (int): (lpct)%タイル以下の値は(lpct)%タイルの値に，(100-lpct)%タイル以上の値は(100-lpct)%タイルの値に置き換える
    train_col (pd.Series): パーセンタイル値を計算するための訓練データの列

    Returns
    ------------------
    col (pd.Series): 外れ値の置き換え処理ごの列
    """

    pct_low = np.percentile(train_col, lpct)
    pct_high = np.percentile(train_col, 100 - lpct)
    col[col >= pct_high] = pct_high
    col[col <= pct_low] = pct_low
    return col


if __name__ == "__main__":
    # log setting
    exp_dir = os.path.dirname(sys.argv[1])
    exp_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    logger = set_exp_logging(exp_dir, exp_name)

    # 設定ファイルから定数をロード
    setting_dict = json2dict(sys.argv[1])
    logger.info(f"Settings: {setting_dict}")
    task_name = setting_dict["TASK_NAME"]
    raw_data_path = setting_dict["RAW_DATA_PATH"]
    trainrepair_save_path = setting_dict["TRAIN-REPAIR_SAVE_PATH"]
    test_save_path = setting_dict["TEST_SAVE_PATH"]
    test_ratio = setting_dict["TEST_RATIO"]
    random_seed = setting_dict["RANDOM_SEED"]

    # TODO:以下は，タスク名に応じて前処理を変える（関数化or外部化したい）
    # credit dataset
    if task_name == "credit":
        # パスからデータを読み込み
        df_ori = pd.read_csv(raw_data_path)
        logger.info(f"df_ori.shape={df_ori.shape}")
        logger.info(f"df_ori.columns={df_ori.columns}")

        # ===================================================
        # train / test 分けない状態で適用しても問題ない処理
        # ===================================================

        # 使う列とそれらの変更後の名前
        used_columns_rename_dict = {
            "Age (years)": "Age",
            "Sex & Marital Status": "Gender",
            "Occupation": "Job",
            "Type of apartment": "Housing",
            "Value Savings/Stocks": "Saving accounts",
            "Account Balance": "Balance",
            "Credit Amount": "Credit amount",
            "Duration": "Duration of Credit (month)",
            "Purpose": "Purpose",
            "Creditability": "Creditability",
        }
        # 使わない列
        removed_columns = [col for col in df_ori.columns if col not in list(used_columns_rename_dict.keys())]

        # 使わない列を落として，使う列はリネームする
        df_new = df_ori.drop(columns=removed_columns).rename(columns=used_columns_rename_dict)
        logger.info(f"After delete / rename columns: {df_new.columns}")

        # Purposeの列のカテゴリ数を減らす
        df_purp = df_new.copy()
        df_purp["Purpose"] = df_purp["Purpose"].apply(purpose_col_transform)
        # 前後のカテゴリ数を表示
        logger.info(
            f"Before #Purpose: {len(df_new['Purpose'].value_counts())}, After #Purpose: {len(df_purp['Purpose'].value_counts())}"
        )

        # ダミー変数にする
        ohe_features = ["Balance", "Purpose", "Saving accounts", "Housing", "Job"]
        df_ohe = pd.get_dummies(df_purp, columns=ohe_features)
        logger.info(f"After OHE columns: {df_ohe.columns}")

        # Genderの列は，1,2,3がMale，4がFemaleに対応
        # https://online.stat.psu.edu/stat857/node/216/
        df_ge = df_ohe.copy()
        df_ge["Gender"] = df_ge["Gender"].apply(lambda x: "Male" if x <= 3 else "Female")
        # Male, Femaleの数
        logger.info(f'#Male, Female\n{df_ge["Gender"].value_counts() / len(df_ge["Gender"])}')
        # ダミー変数にする（Gender_Female, Gender_Male)
        df_ge = pd.get_dummies(df_ge, columns=["Gender"])

        # Credit amountは対数変換して標準化
        # 対数変換
        df_log = df_ge.copy()
        df_log["Credit amount"] = df_log["Credit amount"].apply(np.log)

        # ===================================================
        # データの分割処理
        # ===================================================

        # train and repair / test に分ける
        # 割合をログ表示しておく
        logger.info(f"train-repair : test = {1-test_ratio} : {test_ratio}")
        # 分割を実行
        df_train_repair, df_test = train_test_split(df_log, test_size=test_ratio, random_state=random_seed)

        # ===================================================
        # いったんtrainだけを対象に適用して，
        # その後testに適用しなきゃいけない処理（標準化など）
        # ===================================================

        # train and repairに前処理を適用
        # 標準化
        std = StandardScaler()
        scaled_val = std.fit_transform(df_train_repair["Credit amount"].values.reshape(-1, 1)).reshape(-1)
        df_train_repair_std = df_train_repair.copy()
        df_train_repair_std["Credit amount"] = scaled_val
        df_train_repair_std.to_csv(trainrepair_save_path, index=False)
        logger.info(f"saved to {trainrepair_save_path}")
        logger.info(f"df_train_repair_std.shape = {df_train_repair_std.shape}")

        # testにtrain and repair出やったのと同じ前処理を適用
        # 標準化
        scaled_val = std.transform(df_test["Credit amount"].values.reshape(-1, 1)).reshape(-1)
        df_test_std = df_test.copy()
        df_test_std["Credit amount"] = scaled_val
        df_test_std.to_csv(test_save_path, index=False)
        logger.info(f"saved to {test_save_path}")
        logger.info(f"df_test_std.shape = {df_test_std.shape}")

    # census dataset
    elif task_name == "census":
        # パスからデータを読み込み
        df_ori = pd.read_csv(raw_data_path, header=None)

        # ===================================================
        # train / test 分けない状態で適用しても問題ない処理
        # ===================================================

        # 生データには列名がないので自前でつける
        df_ori.columns = [
            "Age",
            "WorkClass",
            "fnlwgt",
            "Education",
            "EducationNum",
            "MaritalStatus",
            "Occupation",
            "Relationship",
            "Race",
            "Gender",
            "CapitalGain",
            "CapitalLoss",
            "HoursPerWeek",
            "NativeCountry",
            "Income",
        ]
        logger.info(f"df_ori.shape={df_ori.shape}")
        logger.info(f"df_ori.columns={df_ori.columns}")
        # カテゴリ変数の空白削除
        for i, col in enumerate(df_ori.select_dtypes(include="object").columns):
            df_ori[col] = df_ori[col].str.strip()
        # 目的変数を変換
        df_ori["Income"] = (df_ori["Income"] == ">50K").astype("int8")
        # Education列を順序変数に変換
        order = [
            "Preschool",
            "1st-4th",
            "5th-6th",
            "7th-8th",
            "9th",
            "10th",
            "11th",
            "12th",
            "HS-grad",
            "Prof-school",
            "Assoc-acdm",
            "Assoc-voc",
            "Some-college",
            "Bachelors",
            "Masters",
            "Doctorate",
        ]
        edu_map = {col: i for i, col in enumerate(order)}
        df_ori["Education"] = df_ori["Education"].map(edu_map)
        # NativeCountry列をアメリカかそれ以外に分ける
        df_ori["NativeCountry"] = (df_ori["NativeCountry"] == "United-States").astype(int)
        # Occupation 列をグルーピングする
        gdic = {}
        gdic["Occupation_1"] = ["?", "Armed-Forces"]
        gdic["Occupation_2"] = ["Handlers-cleaners", "Other-service", "Priv-house-serv"]
        gdic["Occupation_3"] = ["Adm-clerical", "Machine-op-inspct"]
        gdic["Occupation_4"] = ["Craft-repair", "Transport-moving"]
        gdic["Occupation_5"] = ["Exec-managerial", "Prof-specialty"]
        gdic["Occupation_6"] = ["Protective-serv", "Sales", "Tech-support"]
        gdic["Occupation_7"] = ["Farming-fishing"]
        # グループ分け
        for k, v in gdic.items():
            df_ori[k] = (df_ori["Occupation"].isin(v)).astype("int8")
        # 下の列を削除
        df_ori.drop(columns="Occupation", axis=1, inplace=True)
        # 残りのカテゴリ変数をワン・ホット・エンコーディングする
        rest_cols = df_ori.select_dtypes(include=("O")).columns
        ohe = OneHotEncoder(sparse=False)
        encoded_cols = ohe.fit_transform(df_ori[rest_cols])
        df_ori = pd.concat([df_ori, pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out(rest_cols))], axis=1)
        df_ori.drop(columns=rest_cols, axis=1, inplace=True)
        # 冗長であるためダミー変数1次元削減する
        for k, v in {rcol: [col for col in df_ori.columns if col.find(rcol) > -1] for rcol in rest_cols}.items():
            df_ori.drop(columns=v[-1], inplace=True)
        df_ori.drop(columns="Occupation_7", inplace=True)

        # ===================================================
        # データの分割処理
        # ===================================================

        # train and repair / test に分ける
        # 割合をログ表示しておく
        logger.info(f"train-repair : test = {1-test_ratio} : {test_ratio}")
        # 分割を実行
        df_train_repair, df_test = train_test_split(df_ori, test_size=test_ratio, random_state=random_seed)

        # ===================================================
        # いったんtrainだけを対象に適用して，
        # その後testに適用しなきゃいけない処理（標準化など）
        # ===================================================

        # 外れ値処理
        # 訓練データにおける99%タイルを使用して外れ値を置き換える
        for col in [col for col in df_train_repair.columns if len(df_train_repair[col].unique()) > 10]:
            round_fill(df_train_repair[col], 100 - 99, df_train_repair[col])  # 訓練データの外れ値処理
            round_fill(df_test[col], 100 - 99, df_train_repair[col])  # テストデータの外れ値処理（訓練データのパーセンタイル値を使用）

        # 訓練データに対して標準化を行う
        # 説明変数と目的変数の分離
        X_cols = df_train_repair.columns.to_list()
        X_cols.remove("Income")
        X_train = df_train_repair[X_cols]
        y_train = df_train_repair["Income"]
        # 標準化
        std = StandardScaler()
        scaled_vals = std.fit_transform(X_train)
        df_train_repair_std = df_train_repair.copy()
        df_train_repair_std[X_cols] = scaled_vals
        print(df_train_repair_std[X_cols].describe())
        df_train_repair_std.to_csv(trainrepair_save_path, index=False)
        logger.info(f"saved to {trainrepair_save_path}")
        logger.info(f"df_train_repair_std.shape = {df_train_repair_std.shape}")

        # testにtrain and repair出やったのと同じ標準化を適用
        X_test = df_test[X_cols]
        y_test = df_test["Income"]
        scaled_vals = std.transform(X_test)
        df_test_std = df_test.copy()
        df_test_std[X_cols] = scaled_vals
        print(df_test_std[X_cols].describe())
        df_test_std.to_csv(test_save_path, index=False)
        logger.info(f"saved to {test_save_path}")
        logger.info(f"df_test_std.shape = {df_test_std.shape}")

    # bank dataset
    elif task_name == "bank":
        # ===================================================
        # train / test 分けない状態で適用しても問題ない処理
        # ===================================================

        # パスからデータを読み込み
        df_ori = pd.read_csv(raw_data_path, sep=";")
        logger.info(f"df_ori.shape={df_ori.shape}")
        logger.info(f"df_ori.columns={df_ori.columns}")

        # yes/noの列を1/0に置き換え
        yn_map = {"yes": 1, "no": 0}
        df_ori["default"] = df_ori["default"].map(yn_map)
        df_ori["housing"] = df_ori["housing"].map(yn_map)
        df_ori["loan"] = df_ori["loan"].map(yn_map)
        df_ori["y"] = df_ori["y"].map(yn_map)

        # poutcomeがsuccessかどうかに変換
        df_ori["poutcome"] = (df_ori["poutcome"] == "success").astype(int)

        # jobのグループ化
        # グループ化辞書
        gdic = {}
        gdic["G1"] = ["blue-collar", "entrepreneur"]
        gdic["G2"] = ["housemaid", "services", "tecnhnician"]
        gdic["G3"] = ["self-employed", "admin.", "management"]
        gdic["G4"] = ["unemployed", "retired"]
        gdic["G5"] = ["student"]
        gdic["G6"] = ["unknown"]
        # グループ分け
        for k, v in gdic.items():
            df_ori[k] = (df_ori["job"].isin(v)).astype("int8")
        # もとのjob列を削除
        df_ori.drop(columns="job", axis=1, inplace=True)

        # balance列のビニング処理
        df_ori["balance"] = pd.qcut(df_ori["balance"], 4)

        # 残りのカテゴリ変数をワン・ホット・エンコーディングする
        rest_cols = df_ori.select_dtypes(include=("O", "category")).columns
        ohe = OneHotEncoder(sparse=False)
        encoded_cols = ohe.fit_transform(df_ori[rest_cols])
        # ohe後の列を横に結合
        df_ori = pd.concat([df_ori, pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out(rest_cols))], axis=1)
        # ohe前の列を削除
        df_ori.drop(columns=rest_cols, axis=1, inplace=True)

        # 冗長であるためダミー変数1次元削減する
        for k, v in {rcol: [col for col in df_ori.columns if col.find(rcol) > -1] for rcol in rest_cols}.items():
            print(k, v[-1])
            df_ori.drop(columns=v[-1], inplace=True)
        df_ori.drop(columns="G6", inplace=True)

        # 役に立たなそうな列の削除
        df_ori.drop(columns=["pdays", "campaign", "previous"], inplace=True)

        # duration列に対数変換する
        df_ori["duration"] = df_ori["duration"].apply(np.log1p)

        # ===================================================
        # データの分割処理
        # ===================================================

        # train and repair / test に分ける
        # 割合をログ表示しておく
        logger.info(f"train-repair : test = {1-test_ratio} : {test_ratio}")
        # 分割を実行
        df_train_repair, df_test = train_test_split(df_ori, test_size=test_ratio, random_state=random_seed)

        # ===================================================
        # いったんtrainだけを対象に適用して，
        # その後testに適用しなきゃいけない処理（標準化など）
        # ===================================================

        # 外れ値処理
        # 訓練データにおける99%タイルを使用して外れ値を置き換える
        for col in [col for col in df_train_repair.columns if len(df_train_repair[col].unique()) > 10]:
            round_fill(df_train_repair[col], 100 - 99, df_train_repair[col])  # 訓練データの外れ値処理
            round_fill(df_test[col], 100 - 99, df_train_repair[col])  # テストデータの外れ値処理（訓練データのパーセンタイル値を使用）

        # 訓練データに対して標準化を行う
        # 説明変数と目的変数の分離
        X_cols = df_train_repair.columns.to_list()
        X_cols.remove("y")
        X_train = df_train_repair[X_cols]
        y_train = df_train_repair["y"]
        # 標準化
        std = StandardScaler()
        scaled_vals = std.fit_transform(X_train)
        df_train_repair_std = df_train_repair.copy()
        df_train_repair_std[X_cols] = scaled_vals
        # print(df_train_repair_std[X_cols].describe())
        df_train_repair_std.to_csv(trainrepair_save_path, index=False)
        logger.info(f"saved to {trainrepair_save_path}")
        logger.info(f"df_train_repair_std.shape = {df_train_repair_std.shape}")

        # testにtrain and repair出やったのと同じ標準化を適用
        X_test = df_test[X_cols]
        y_test = df_test["y"]
        scaled_vals = std.transform(X_test)
        df_test_std = df_test.copy()
        df_test_std[X_cols] = scaled_vals
        # print(df_test_std[X_cols].describe())
        df_test_std.to_csv(test_save_path, index=False)
        logger.info(f"saved to {test_save_path}")
        logger.info(f"df_test_std.shape = {df_test_std.shape}")

    else:
        raise NotImplementedError
