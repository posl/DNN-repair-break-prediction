import os, sys, subprocess, argparse

if __name__ == "__main__":
    # 引数のパース
    parser = argparse.ArgumentParser()
    parser.add_argument("is_safety", type=int, default=0)
    args = parser.parse_args()
    is_safety = args.is_safety
    if is_safety == 1:
        datasets = ["acasxu_n1_9_prop7", "acasxu_n2_9_prop8", "acasxu_n3_5_prop2"]
        methods = ["care"]
    else:
        datasets = ["credit", "census", "bank", "fm", "c10", "gtsrb", "imdb", "rtmr"]
        methods = ["care", "arachne", "apricot"]

    # ソースのディレクトリへ移動
    os.chdir("../src")
    # foldとrepを指定して実行
    for method in methods:
        for dataset in datasets:
            print(f"run dataset={dataset}, method={method}...")
            result = subprocess.run(["python", "preprocess_repair_break_dataset.py", str(method), str(dataset)])
            # サブプロセスのエラーチェック
            if result.returncode != 0:
                # エラー終了コードを指定してメインプロセスを終了
                exit(1)
