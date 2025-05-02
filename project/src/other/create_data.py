import os
import csv
from datetime import datetime
from enums import ChatType
import pandas as pd
from services.stock_data import convert_timeframe

CSV_DATA_PATH = "data/データ/original.csv"


def create_data_and_save(chart_type: str):
    # 1分足のデータ作成
    output_csv = f"data/データ/{chart_type}.csv"
    df = pd.read_csv(CSV_DATA_PATH, encoding="utf-8")
    df_resampled = convert_timeframe(
        df=df,
        chat_type=chart_type,
    )
    df_resampled.to_csv(
        output_csv,
        index=False,
        encoding="utf-8",
    )


def main():
    for chat_type in ChatType:
        create_data_and_save(chat_type.value)


if __name__ == "__main__":
    main()
