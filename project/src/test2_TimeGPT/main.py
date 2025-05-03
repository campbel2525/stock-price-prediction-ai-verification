from nixtla import NixtlaClient
from config.settings import TIME_GPT_API_KEY
import pandas as pd
from services.stock_data import convert_timeframe
from enums import ChatType


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


def run_time_gpt(
    cav_pth: str,
    chat_type: str,
    target_forecast_column: str,
    horizon: int,
    time_column: str = TIMESTAMP_NAME,
    api_key: str = TIME_GPT_API_KEY,
):
    # CSV 読み込み、フォーマット
    df = pd.read_csv(cav_pth)
    df = convert_timeframe(
        df,
        chat_type=chat_type,
        fill_missing=True,
    )
    df = df.rename(columns={time_column: "ds", target_forecast_column: "y"})
    # df = df.iloc[-1200:-49].reset_index(drop=True)

    # 予想
    client = NixtlaClient(api_key=api_key)
    forecast_df = client.forecast(
        df=df,
        h=horizon,
    )

    return forecast_df


def main():
    # 一分足
    forecast_df = run_time_gpt(
        cav_pth=CSV_PATH,
        chat_type=ChatType.M1.value,
        target_forecast_column="open",
        horizon=24,
    )
    output_csv_path = "data/結果/検証_TimeGPT/M1_24.csv"
    forecast_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
