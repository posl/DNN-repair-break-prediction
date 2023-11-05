import sys, re
from collections import defaultdict
import numpy as np

num_rdlms = 20

if __name__ == "__main__":
    dataset = sys.argv[1]
    is_parse_only = int(sys.argv[2])
    num_folds_from_ds = {
        "credit": 5,
        "census": 10,
        "bank": 10,
        "fm": 5,
        "c10": 5,
        "gtsrb": 5,
    }
    num_folds = num_folds_from_ds[dataset]
    log_file = f"/src/experiments/arachne/logs/{dataset}-arachne-repair-setting1.log"
    parsed_log_file = f"/src/experiments/arachne/logs/{dataset}-arachne-repair-setting1-parsed.log"

    if is_parse_only:
        # log_fileを読み込む
        with open(log_file, "r") as f:
            lines = f.readlines()

        # まずはパースしやすいように必要名行以外を省いたログファイルを出力したい
        output_lines = []
        for line in lines:
            if "fold=" in line or "Total execution time" in line:
                output_lines.append(line)
        with open(parsed_log_file, "w") as f:
            f.writelines(output_lines)

    else:
        # parsed_log_fileを読み込む
        with open(parsed_log_file, "r") as f:
            lines = f.readlines()

        # parsed_log_fileから実行時間を抽出する
        t = []  # foldsごとの時間
        pat1 = r"fold=(\d), rep=(\d)"  # fold数の部分を取り出す正規表現
        pat2 = r"Total execution time: (\d+\.\d+) sec\."  # 実行時間の部分を取り出す正規表現

        t = defaultdict(list)  # 辞書のキーはfold数、値はそのfoldの実行時間のリスト(repsごと)
        for i, line in enumerate(lines):
            # i行目がfold, rep数だったら...
            mat1 = re.search(pat1, line)
            if mat1:
                k, rep = int(mat1.group(1)), int(mat1.group(2))
                # i+1行目をみて実行時間を取得
                mat2 = re.search(pat2, lines[i + 1])
                t[k].append(float(mat2.group(1)))

        # 配列tをparsed_log_fileに1行ずつ追記する
        with open(parsed_log_file, "a") as f:
            f.write(f"repair time for each folds\n")
            for k, t_fold in t.items():
                print(f"fold {k}, {t_fold}, mean: {np.mean(t_fold)}")
                f.write(f"fold {k}, {t_fold}, mean: {np.mean(t_fold)}\n")
