"""
Apricot実行時間をログファイルから解析しやすくなるように, 関連する行だけ残してparse後のログファイルを出力.
"""

import sys, re


num_rdlms = 20

if __name__ == "__main__":
    dataset = sys.argv[1]
    num_folds_from_ds = {
        "credit": 5,
        "census": 10,
        "bank": 10,
        "fm": 5,
        "c10": 5,
        "gtsrb": 5,
        "imdb": 5,
        "rtmr": 5
    }
    num_folds = num_folds_from_ds[dataset]
    rdlm_log_file = f"/src/experiments/apricot/logs/{dataset}-train-rdlm-setting1.log"

    # rdlm_log_fileを読み込む
    with open(rdlm_log_file, "r") as f:
        lines = f.readlines()

    # まずはパースしやすいように必要名行以外を省いたログファイルを出力したい
    output_lines = []
    for line in lines:
        if "Start rDLM_train.py" in line:
            output_lines.append(line)
        elif "saved model in" in line:
            rdlm_id = int(re.search(r"rDLM-(\d{1,2}).pt", line).group(1))
            if rdlm_id == 0 or rdlm_id == num_rdlms - 1:
                output_lines.append(line)

    with open(f"/src/experiments/apricot/logs/{dataset}-train-rdlm-setting1-parsed.log", "w") as f:
        f.writelines(output_lines)
