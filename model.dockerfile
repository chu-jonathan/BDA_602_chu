FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.in

COPY hw_05_model.py .

ENV MYSQL_HOST=db
ENV MYSQL_USER=jchu
ENV MYSQL_PASSWORD=bda
ENV MYSQL_DB=baseball

CMD [ "python", "./hw_05_model.py" ]
