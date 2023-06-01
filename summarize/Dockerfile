from python:3.10.11-buster


# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Update pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy poetry files
COPY ./poetry.lock /poetry.lock
COPY ./pyproject.toml /pyproject.toml

# Install dependencies
RUN PATH="$PATH:/root/.local/bin" poetry config virtualenvs.create false
# RUN PATH="$PATH:/root/.local/bin" poetry install