FROM python:3.11

ADD main.py .

RUN pip install requests scipy

CMD ["python", "./main.py"]