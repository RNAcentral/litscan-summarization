from python:3.9

workdir /app
# Update pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

copy ./pyproject.toml /app/pyproject.toml
copy ./poetry.lock /app/poetry.lock

# Install dependencies
RUN PATH="$PATH:/root/.local/bin" poetry config virtualenvs.create false
RUN PATH="$PATH:/root/.local/bin" poetry install

copy search_demo.py /app/search_demo.py
