import os, sys, subprocess, argparse

if __name__ == "__main__":
    # 引数のパース
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_safety", type=int, default=0)
    args = parser.parse_args()
    is_safety = args.is_safety
    if is_safety == 1:
        datasets = ["acasxu_n2_9_prop8", "acasxu_n3_5_prop2"]
        methods = ["aprnn"]
    else:
        datasets = ["credit", "census", "bank", "fm", "c10", "gtsrb", "imdb", "rtmr"]
        methods = ["care", "arachne", "apricot"]
        # methods = ["care"]

    # ソースのディレクトリへ移動
    os.chdir("../src")
    # foldとrepを指定して実行
    for mi, method in enumerate(methods):
        for di, dataset in enumerate(datasets):
            print(f"run dataset={dataset}, method={method}...")
            result = subprocess.run(["python", "build_repair_break_model.py", str(method), str(dataset), "--without_resampling"])
            # サブプロセスのエラーチェック
            if result.returncode != 0:
                # エラー終了コードを指定してメインプロセスを終了
                exit(1)
