import os, sys, subprocess

if __name__ == "__main__":
    datasets = ["credit", "census", "bank", "fm", "c10", "gtsrb", "imdb", "rtmr"]
    # ソースのディレクトリへ移動
    os.chdir("../src")
    # foldとrepを指定して実行
    for dataset in datasets:
        result = subprocess.run(["python", "transfer_methods.py", str(dataset)])
        # サブプロセスのエラーチェック
        if result.returncode != 0:
            # エラー終了コードを指定してメインプロセスを終了
            exit(1)
