FROM python:3.8

COPY pyproject.toml poetry.lock ./

RUN pip install poetry && poetry config virtualenvs.in-project true && poetry install --no-dev

COPY src ./src
COPY models ./models

EXPOSE 8000

CMD ["poetry", "run", "flask", "--app", "./src/main", "run", "--host", "0.0.0.0"]
