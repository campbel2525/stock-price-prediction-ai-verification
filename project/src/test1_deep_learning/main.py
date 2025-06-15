from nixtla import NixtlaClient
from config.settings import TIME_GPT_API_KEY
import pandas as pd
from services.stock_data import convert_timeframe
from enums import ChatType
from services.learning_ai import run_deep_learning
import csv
import os

"""
# 対象のcsvの形式

```
timestamp,open,high,low,close,volume
2018-01-02 07:05,1309.57,1309.62,1309.45,1309.48,209
2018-01-02 07:06,1309.49,1309.59,1309.34,1309.57,198
```
"""


CSV_PATH = "data/csv/total.csv"
TIMESTAMP_NAME = "timestamp"

chart_types = [
    "M1",
    "M5",
    "M15",
    "H1",
    "H4",
    "D1",
    "W1",
]

window_sizes = [
    10,
    20,
    30,
    40,
    50,
]

model_nos = [1, 2, 3]

epochs_list = [
    15,
    30,
]

BASE_MODEL_RESULT_DIR = "data/結果/検証1_deep_learning/モデル/"
BASE_RESULT_DIR = "data/結果/検証1_deep_learning/結果/"


def main():
    os.makedirs(BASE_MODEL_RESULT_DIR, exist_ok=True)
    os.makedirs(BASE_RESULT_DIR, exist_ok=True)

    for chart_type in chart_types:

        # 必要なデータのみ取得
        csv_path = f"data/データ/{chart_type}.csv"
        df = pd.read_csv(csv_path)
        df[df["timestamp"] >= "2025-01-01"]
        df = df[["open", "high", "low", "close"]]

        for window_size in window_sizes:
            for model_no in model_nos:
                for epochs in epochs_list:
                    test_name = f"chart_type_{chart_type}__window_size_{window_size}__model_no_{model_no}__epochs_{epochs}"
                    print(test_name)

                    # 学習を実行
                    run_deep_learning(
                        df=df,
                        model_no=model_no,
                        window_size=window_size,
                        epochs=epochs,
                        model_h5_save_path=f"{BASE_MODEL_RESULT_DIR}{test_name}.h5",
                        result_csv_save_path=f"{BASE_RESULT_DIR}{test_name}.csv",
                    )


def main2():
    chart_type = "H1"
    window_size = 30
    model_no = 1
    epochs = 20

    test_name = f"chart_type_{chart_type}__window_size_{window_size}__model_no_{model_no}__epochs_{epochs}"
    print(test_name)

    # 必要なデータのみ取得
    csv_path = f"data/データ/{chart_type}.csv"
    df = pd.read_csv(csv_path)
    df[df["timestamp"] >= "2025-01-01"]
    df = df[["open", "high", "low", "close"]]

    # 学習を実行
    run_deep_learning(
        df=df,
        model_no=model_no,
        window_size=window_size,
        epochs=epochs,
        model_h5_save_path=f"{BASE_MODEL_RESULT_DIR}{test_name}.h5",
        result_csv_save_path=f"{BASE_RESULT_DIR}{test_name}.csv",
    )


if __name__ == "__main__":
    # from config.debug import *

    main()
