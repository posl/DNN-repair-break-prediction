import os, sys, subprocess

if __name__ == "__main__":
    methods = ["care", "arachne", "apricot"]
    datasets = ["credit", "census", "bank", "fm", "c10", "gtsrb", "imdb", "rtmr"]
    # ソースのディレクトリへ移動
    os.chdir("../src")
    # foldとrepを指定して実行
    for method in methods:
        for dataset in datasets:
            print(f"run dataset={dataset}, method={method}...")
            result = subprocess.run(["python", "build_repair_break_model.py", str(method), str(dataset)])
            # サブプロセスのエラーチェック
            if result.returncode != 0:
                # エラー終了コードを指定してメインプロセスを終了
                exit(1)
