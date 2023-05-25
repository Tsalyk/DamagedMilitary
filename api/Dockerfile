FROM python:latest

COPY ./app.py ./requirements.txt ./models /opt/app/

RUN pip install -r /opt/app/requirements.txt

ENTRYPOINT ["python", "/opt/app/app.py"]
