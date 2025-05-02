import pandas as pd


def chat_type_to_minutes(chat_type: str) -> int:
    """
    チャートタイプを分数に変換する。

    Parameters
    ----------
    chat_type : str
        チャートタイプ

    Returns
    -------
    int
        分数
    """
    if chat_type == "M1":
        return 1

    if chat_type == "M5":
        return 5

    if chat_type == "M15":
        return 15

    if chat_type == "M30":
        return 30

    if chat_type == "H1":
        return 60

    if chat_type == "H4":
        return 240

    if chat_type == "D1":
        return 1440

    if chat_type == "W1":
        return 10080

    if chat_type == "MN":
        return 43200

    raise ValueError(f"Invalid chart type: {chat_type}")


import pandas as pd


def convert_timeframe(
    df: pd.DataFrame,
    chat_type: str,
    fill_missing: bool = True,
) -> pd.DataFrame:
    """
    1分足データ（timestamp カラムあり）を指定分足に変換する。

    Returns:
        pd.DataFrame
            分足変換後の DataFrame。インデックスは分足の終端時刻。
            カラムは ['timestamp','open','high','low','close','volume']。
            'timestamp' は各期間の最初の時刻を示す。
    """
    # 1分足のチャートタイプを分数に変換
    minutes = chat_type_to_minutes(chat_type)

    # ① timestamp カラムを datetime index に (drop=False で列も保持)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp", drop=False)
    else:
        df.index = pd.to_datetime(df.index)
        df["timestamp"] = df.index
    df = df.sort_index()

    # ② 欠損分１分足の検出と任意で埋める
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="T")
    missing = full_idx.difference(df.index)
    if len(missing) > 0:
        print(
            f"⚠️ 欠損データ検出: {len(missing)} 件 "
            f"(開始: {missing.min()}, 終了: {missing.max()})"
        )
        if fill_missing:
            df = df.reindex(full_idx)
            df.index.name = "timestamp"
            df["timestamp"] = df.index
            df["close"] = df["close"].ffill()
            df["open"] = df["close"]
            df["high"] = df["close"]
            df["low"] = df["close"]
            df["volume"] = df["volume"].fillna(0)

    # ③ N 分足にリサンプル (例: 5分足なら 00:05,00:10,... の終了時刻に揃える)
    rule = f"{minutes}T"
    agg_dict = {
        "timestamp": "first",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df_resampled = (
        df.resample(rule, label="right", closed="right", origin="start_day")
        .agg(agg_dict)
        .dropna(subset=["open"])
    )

    return df_resampled
