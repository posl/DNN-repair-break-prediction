import re
from datetime import datetime


def extract_datetime_from_log_line(log_line):
    # 正規表現パターンを使用して日付と時刻を抽出
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    match = re.search(pattern, log_line)

    if match:
        # マッチした部分をdatetimeオブジェクトに変換
        datetime_str = match.group(1)
        return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    else:
        return None


# ログ行から日付と時刻を取得する例
log_line1 = "2023-07-10 11:52:34,040 - rDLM_train.py - INFO - processing rdlm_idx 1..."
log_line2 = "2023-07-10 11:53:34,040 - rDLM_train.py - INFO - processing rdlm_idx 1..."
result1 = extract_datetime_from_log_line(log_line1)
result2 = extract_datetime_from_log_line(log_line2)
print("抽出された日付と時刻:", result1)
print("抽出された日付と時刻:", result2)
td = result2 - result1
print(td.seconds)
