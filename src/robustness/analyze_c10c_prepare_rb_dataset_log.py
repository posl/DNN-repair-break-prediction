# c10cに対するrepair/break datasetのログを解析してそれぞれのcorruptionに対するrepair/breakの数を数える
import os, sys, re
import numpy as np

if __name__ == "__main__":
    res_arr = []
    for method in ["care", "apricot", "arachne"]:
        _res_list_method = []
        print(method)
        log_dir = f"/src/experiments/{method}/logs"
        log_file_name = f"c10c-prepare-repair-break-dataset.log"
        # ログファイルを開く
        with open(os.path.join(log_dir, log_file_name), "r") as f:
            log_txt = f.readlines()
        # df_train.shapeという文字列を含む行だけ残す
        log_txt = [line for line in log_txt if "#repaired" in line or "#broken" in line]
        pat1 = r"#repaired is True: (\d+) / (\d+)"
        pat2 = r"#broken is True: (\d+) / (\d+)"
        for lt in log_txt:
            rb = "r" if "#repaired" in lt else "b"
            pat = pat1 if "#repaired" in lt else pat2
            match = re.search(pat, lt)
            ratio = int(match.group(1)) / int(match.group(2))
            _res_list_method.append(int(match.group(1)))
            _res_list_method.append(ratio)
        # np.array(res_list) を4つずつ区切って行にしていく
        res_list_method = []
        for i in range(0, len(_res_list_method), 4):
            res_list_method.append(_res_list_method[i:i+4])
        res_list_method = np.array(res_list_method)
        res_arr.append(res_list_method)
    res_arr = np.concatenate(res_arr, axis=1)
    print(res_arr.shape)
    np.savetxt("./tmp.csv", res_arr, delimiter=",")
    
