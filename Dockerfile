FROM python:3.5
ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
ADD . /src
WORKDIR /src
CMD ["python", "loadData.py"]