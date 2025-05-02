FROM public.ecr.aws/docker/library/python:3.12-bullseye

# apt-getのアップデート
RUN apt-get update && apt-get upgrade -y

# playwright
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    python3-dev \
    default-libmysqlclient-dev \
    build-essential \
    libkrb5-dev \
    libsnappy-dev \
    libreoffice \
    libgl1-mesa-glx \
    poppler-utils

# デフォルトのディレクトリを設定
# イメージにディレクトリがないので作成される
WORKDIR /project

# project配下に.venvを作成する
ENV PIPENV_VENV_IN_PROJECT=1

# log出力をリアルタイムにする
ENV PYTHONUNBUFFERED=1

# キャッシュを作成しない
ENV PYTHONDONTWRITEBYTECODE=1

# パスを通す
ENV PYTHONPATH="/project"

# pipのアップデート
RUN pip install --upgrade pip

# pipenvのインストール
RUN pip install --upgrade setuptools pipenv

# # ライブラリのインストール
# COPY Pipfile /project/
# COPY Pipfile.lock /project/
# RUN pipenv install --dev
# RUN pipenv run playwright install

# # プロジェクトのファイルをコピー
# COPY . /project/
