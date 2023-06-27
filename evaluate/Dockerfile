FROM python:3.9-rc-slim-buster

RUN apt update && apt upgrade -y
RUN apt install -y curl

WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py
# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# COPY poetry.lock /app/poetry.lock
# COPY pyproject.toml /app/pyproject.toml



COPY . /app
WORKDIR /app/app
# # Install dependencies
RUN PATH="$PATH:/root/.local/bin" poetry config virtualenvs.create false
RUN PATH="$PATH:/root/.local/bin" poetry install

WORKDIR /app
