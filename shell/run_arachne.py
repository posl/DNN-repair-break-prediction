import os, sys, subprocess

if __name__ == "__main__":
    # dataset名はコマンドライン引数から取得
    dataset = sys.argv[1]
    # ソースのディレクトリへ移動
    os.chdir("../src")
    # reps数は全データセット共通
    num_reps = 5
    # fold数はデータセットごとに異なる
    if dataset in ["census", "bank"]:
        num_folds = 10
    else:
        num_folds = 5
    # 実験設定のパス
    path = f"/src/experiments/care/{dataset}-training-setting1.json"
    # foldとrepを指定して実行
    for fold in range(num_folds):
        for rep in range(num_reps):
            # fold0, rep0, 1はスキップ
            # if fold == 0 and (rep == 0 or rep == 1):
            #     continue
            print(f"run dataset={dataset}, fold={fold}, rep={rep}...")
            result = subprocess.run(["python", "arachne_repair.py", path, str(fold), str(rep)])
            # サブプロセスのエラーチェック
            if result.returncode != 0:
                # エラー終了コードを指定してメインプロセスを終了
                exit(1)
