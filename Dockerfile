FROM python:3

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r /app/requirements.txt
RUN mkdir plots

COPY final.py ./

CMD ["python", "final.py"]