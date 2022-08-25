# frontent/Dockerfile

FROM python:3.8-slim

COPY requirements.txt app/requirements.txt

WORKDIR /app

RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8501

ENTRYPOINT ["streamlit","run"]
CMD ["Home.py"]