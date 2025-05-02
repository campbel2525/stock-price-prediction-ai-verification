import os
from typing import List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Dropout, Dense
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    Dropout,
    Add,
    Activation,
    Normalization,
)
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
    Add,
    Activation,
    Normalization,
)
from tensorflow.keras.models import Model

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
    input_units = len(train_inputs[0][0])
    output_units = len(train_outputs[0])
    time_steps = len(train_inputs[0])
    model = _get_model(model_no, input_units, output_units, time_steps)

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
    input_units: int,
    output_units: int,
    time_steps,
):
    if model_no == 1:
        return _model1(output_units)

    if model_no == 2:
        return _model2(input_units, output_units, time_steps)

    if model_no == 3:
        return _model3(input_units, output_units)

    if model_no == 4:
        return _model4(input_units, output_units)

    raise ValueError(f"model_no {model_no} is not supported.")


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
    input_units: int,
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
                input_shape=(time_steps, input_units),
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


def _model3(input_units: int, output_units: int) -> Model:
    """
    3次元入力 (batch, 1, input_units) を受け取る ResNet風 MLP
    """

    def residual_block(x, units, dropout_rate, l2_reg=1e-3):
        shortcut = Dense(units, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        out = Dense(units, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        out = BatchNormalization()(out)
        out = Activation("relu")(out)
        out = Dropout(dropout_rate)(out)
        out = Add()([shortcut, out])
        out = Activation("relu")(out)
        return out

    # 入力: (batch, 1, input_units)
    inputs = Input(shape=(1, input_units))
    # 1次元目を潰して (batch, input_units)
    x = Flatten()(inputs)
    # 正規化層
    norm = Normalization()
    # adapt は学習データを用意してから外部で呼び出す
    # 例: norm.adapt(train_inputs.reshape(-1, input_units))
    x = norm(x)

    # 残差ブロック

    x = residual_block(x, units=256, dropout_rate=0.4)
    x = residual_block(x, units=128, dropout_rate=0.3)
    x = residual_block(x, units=64, dropout_rate=0.2)

    # x = residual_block(x, units=1024, dropout_rate=0.8)
    # x = residual_block(x, units=512, dropout_rate=0.6)
    # x = residual_block(x, units=256, dropout_rate=0.4)
    # x = residual_block(x, units=128, dropout_rate=0.3)
    # x = residual_block(x, units=64, dropout_rate=0.2)
    # x = residual_block(x, units=32, dropout_rate=0.1)

    # 出力層
    outputs = Dense(output_units, activation="linear")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="huber",
        metrics=METRICS,
    )
    return model


def _model4(input_units: int, output_units: int) -> Model:
    """
    3次元入力 (batch, 1, input_units) を受け取り、深いResNet風MLPを実装
    """

    def residual_block(x, units, dropout_rate=0.3, l2_reg=1e-3):
        # ショートカット経路
        shortcut = Dense(units, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        # メイン経路
        out = Dense(units, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        out = BatchNormalization()(out)
        out = Activation("relu")(out)
        out = Dropout(dropout_rate)(out)
        # 合流と再活性化
        out = Add()([shortcut, out])
        out = Activation("relu")(out)
        return out

    # 入力: (batch, 1, input_units)
    inputs = Input(shape=(1, input_units))
    # Flattenして2次元に
    x = Flatten()(inputs)
    # 全体正規化
    norm = Normalization()
    x = norm(x)

    # 入力ドロップアウト
    x = Dropout(0.2)(x)

    # 残差ブロックを4段構築 (512→256→128→64)
    x = residual_block(x, units=512, dropout_rate=0.4, l2_reg=1e-3)
    x = residual_block(x, units=256, dropout_rate=0.35, l2_reg=1e-3)
    x = residual_block(x, units=128, dropout_rate=0.3, l2_reg=1e-4)
    x = residual_block(x, units=64, dropout_rate=0.25, l2_reg=1e-4)

    # グローバルスキップ接続 (入力を最後に足し合わせ)
    shortcut_final = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Add()([shortcut_final, x])
    x = Activation("relu")(x)

    # 出力層
    outputs = Dense(output_units, activation="linear")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="huber",
        metrics=METRICS,
    )
    return model
