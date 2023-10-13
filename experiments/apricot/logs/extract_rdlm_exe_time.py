"""
cut_logfile.pyで出力されたlogファイルから, 実行時間を抽出する.
"""

import sys, re
from datetime import datetime
import numpy as np

num_rdlms = 20


def extract_datetime_from_log_line(log_line):
    # 正規表現パターンを使用して日付と時刻を抽出
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    match = re.search(pattern, log_line)

    if match:
        # マッチした部分をdatetimeオブジェクトに変換
        datetime_str = match.group(1)
        return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S,%f")
    else:
        return None


if __name__ == "__main__":
    dataset = sys.argv[1]
    num_folds_from_ds = {
        "credit": 5,
        "census": 10,
        "bank": 10,
        "fm": 5,
        "c10": 5,
        "gtsrb": 5,
    }
    num_folds = num_folds_from_ds[dataset]
    parsed_log_file = f"/src/experiments/apricot/logs/{dataset}-train-rdlm-setting1-parsed.log"

    # parsed_log_fileを読み込む
    with open(parsed_log_file, "r") as f:
        lines = f.readlines()

    t = []  # foldsごとの時間
    for k in range(num_folds):
        t_fold = []  # repsごとの時間
        fmt = f"fold-{k}_rDLM-0.pt"
        for i, line in enumerate(lines):
            if fmt in line:
                t_start = extract_datetime_from_log_line(line)
                # lineの次の行の時間をt_endとする
                t_end = extract_datetime_from_log_line(lines[i + 1])
                td = (t_end - t_start).total_seconds()
                t_fold.append(td)
        print(f"fold {k}, {t_fold}, mean: {np.mean(t_fold)}")
        t.append(np.mean(t_fold))

    # 配列tをparsed_log_fileに1行ずつ追記する
    with open(parsed_log_file, "a") as f:
        f.write(f"rDLM time for each folds\n")
        for k, tk in enumerate(t):
            f.write(f"fold {k}: {tk}\n")
