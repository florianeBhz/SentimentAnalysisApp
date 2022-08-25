# backend/Dockerfile

FROM python:3.8-slim

COPY requirements.txt app/requirements.txt

WORKDIR /app

RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080" , "--reload"]