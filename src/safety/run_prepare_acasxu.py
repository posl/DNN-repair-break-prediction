"""prepare_acasxu.pyを特定の入力引数に対して実行する"""

import os, sys, subprocess

if __name__ == "__main__":
    inputs = [
        # [2, 9, 8],
        [3, 5, 2],
        [1, 9, 7],
    ]
    for input in inputs:
        cmd = ["python", "prepare_acasxu.py", "../../experiments/care/", str(input[0]), str(input[1]), str(input[2])]
        print(f"executing: {cmd}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
                exit(1)