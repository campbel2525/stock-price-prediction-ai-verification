import os
from typing import List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Dropout, Dense
import pandas as pd
import numpy as np


"""
学習でログを取るもの定義
"""
METRICS = [
    "mae",  # mean_absolute_error
    "mse",  # mean_squared_error
    "mape",  # mean_absolute_percentage_error
    tf.keras.metrics.RootMeanSquaredError(name="rmse"),
    tf.keras.metrics.MeanSquaredLogarithmicError(name="msle"),
]


def run_deep_learning(
    df: pd.DataFrame,
    model_no: int,
    window_size: int,
    epochs: int,
    model_h5_save_path: str,
    result_csv_save_path: str,
    train_data_percent: float = 0.8,
    learning_batch_size: int = 2,
    test_batch_size: int = 10,
):
    """
    学習を実行する
    """
    # 各学習データ用に整形
    arr = df.to_numpy()
    data_list = arr.tolist()
    train_inputs, train_outputs, test_inputs, test_outputs = _get_learning_data(
        data_list=data_list,
        window_size=window_size,
        train_data_percent=train_data_percent,
    )

    # モデル定義
    input_features = len(train_inputs[0][0])
    output_units = len(train_outputs[0])
    time_steps = len(train_inputs[0])
    model = _get_model(model_no, input_features, output_units, time_steps)

    # 学習
    history = _execute_learning(
        train_inputs,
        train_outputs,
        test_inputs,
        test_outputs,
        model,
        epochs=epochs,
        learning_batch_size=learning_batch_size,
        test_batch_size=test_batch_size,
    )

    # モデルの保存（HDF5形式で保存）
    if model_h5_save_path:
        os.makedirs(os.path.dirname(model_h5_save_path), exist_ok=True)
        model.save(model_h5_save_path)

    # 学習の結果
    if result_csv_save_path:
        os.makedirs(os.path.dirname(result_csv_save_path), exist_ok=True)
        df = pd.DataFrame(history.history)
        df.to_csv(result_csv_save_path, index=False)


def _execute_learning(
    train_inputs,
    train_outputs,
    test_inputs,
    test_outputs,
    model,
    epochs=1000,
    learning_batch_size=10,
    test_batch_size=10,
) -> tf.keras.callbacks.History:
    """
    Summary:
        学習を実行する関数
        学習したモデルを保存する

    Args:
        train_inputs (_type_): 学習用データ
        train_outputs (_type_): 学習用ラベル
        test_inputs (_type_): テスト用データ
        test_outputs (_type_): テスト用ラベル
        model (_type_): 学習するモデル
        epochs (int, optional): 学習のエポック数。デフォルトは1000。
        learning_batch_size (int): 学習のバッチサイズ。デフォルトは10。
        test_batch_size (int): テストのバッチサイズ。デフォルトは10。
        model_h5_save_path (str, optional): 学習したモデルを保存するパス。デフォルトはNone。
        result_csv_save_path (str, optional): 学習の結果を保存するパス。デフォルトはNone。

    Returns:
        tf.keras.callbacks.History: 学習の履歴
    """

    # 学習
    history = model.fit(
        train_inputs,
        train_outputs,
        epochs=epochs,
        batch_size=learning_batch_size,
        # verbose=2,
    )

    # テスト
    model.evaluate(
        test_inputs,
        test_outputs,
        batch_size=test_batch_size,  # ネットワークが一度に処理するデータ
        verbose=2,  # 訓練中の進行情報の表示レベルを指定 0: 何も表示しない、1: 進捗バーを表示、2: 各エポックの結果のみを一行で表示
    )

    return history


def _get_learning_data(
    data_list: List[list],
    window_size: int,
    train_data_percent: float = 0.8,
):
    """
    Summary:
        学習データを取得する関数
        入力データと出力データを作成する
        入力データは、window_size分のデータを1つのデータとして扱う
        出力データは、window_size分のデータの次のデータを出力とする

    Args:
        data_list (List[list]): _description_
        window_size (int): _description_
        train_data_percent (float, optional): _description_. Defaults to 0.8.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if len(data_list) <= window_size:
        raise ValueError("window_size is too large")

    inputs = []
    outputs = []
    for i in range(len(data_list) - window_size):
        flattened = [x for sub in data_list[i : i + window_size] for x in sub]
        inputs.append([flattened])
        outputs.append(data_list[i + window_size])

    # inputとoutputsをシャッフルする
    combined = list(zip(inputs, outputs))
    np.random.shuffle(combined)
    inputs, outputs = zip(*combined)
    inputs = list(inputs)
    outputs = list(outputs)

    # 作成したデータを学習用とテスト用に分割
    split_index = int(len(inputs) * train_data_percent)
    train_inputs = inputs[:split_index]
    train_outputs = outputs[:split_index]
    test_inputs = inputs[split_index:]
    test_outputs = outputs[split_index:]

    # numpy配列に変換
    train_inputs = np.array(train_inputs)
    test_inputs = np.array(test_inputs)
    train_outputs = np.array(train_outputs)
    test_outputs = np.array(test_outputs)

    return train_inputs, train_outputs, test_inputs, test_outputs


"""
モデルの定義
"""


def _get_model(
    model_no: int,
    input_features,
    output_units,
    time_steps,
):
    if model_no == 1:
        return _model1(output_units)

    if model_no == 2:
        return _model2(input_features, output_units, time_steps)


def _model1(output_units: int) -> tf.keras.Sequential:
    # モデルの作成
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                units=256,  # 出力空間の次元数
                activation="relu",  # 活性化関数
                kernel_regularizer=tf.keras.regularizers.l2(0.001),  # L2正則化を適用
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(
                rate=0.5,  # rate: 無効化する割合
            ),
            tf.keras.layers.Dense(
                units=128,  # 出力空間の次元数
                activation="relu",  # 活性化関数
                kernel_regularizer=tf.keras.regularizers.l2(0.001),  # L2正則化を適用
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(
                rate=0.5,  # rate: 無効化する割合
            ),
            tf.keras.layers.Dense(
                units=64,  # 出力空間の次元数
                activation="relu",  # 活性化関数
                kernel_regularizer=tf.keras.regularizers.l2(0.001),  # L2正則化を適用
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(
                rate=0.5,  # rate: 無効化する割合
            ),
            tf.keras.layers.Dense(units=output_units, activation="linear"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="huber",
        metrics=METRICS,
    )

    return model


def _model2(
    input_features: int,
    output_units: int,
    time_steps: int,
) -> tf.keras.Sequential:
    model = Sequential(
        [
            # 最初のLSTM層（return_sequences=True で次の層に系列情報を渡す）
            LSTM(
                128,
                activation="tanh",
                return_sequences=True,
                input_shape=(time_steps, input_features),
            ),
            Dropout(0.2),  # ドロップアウトで正則化
            # 2層目のLSTM
            LSTM(64, activation="tanh"),
            Dropout(0.2),
            # 全結合層
            Dense(64, activation="relu"),
            Dense(output_units),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=METRICS,
    )

    return model
