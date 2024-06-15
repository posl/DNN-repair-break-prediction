METHOD_LIST = ["care", "apricot", "arachne"]
DS_LIST = ["credit", "census", "bank"]
import os, sys, subprocess

if __name__ == "__main__":
    for ds in DS_LIST:
        for method in METHOD_LIST:
            print(f"run dataset={ds}, method={method}...")
            result = subprocess.run(["python", "build_repair_break_model.py", ds, method])
            # サブプロセスのエラーチェック
            if result.returncode != 0:
                # エラー終了コードを指定してメインプロセスを終了
                exit(1)