FROM python:3.9
LABEL authors="TornikeAm"

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["python","main.py"]
