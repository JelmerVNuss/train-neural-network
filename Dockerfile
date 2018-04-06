FROM nvidia/cuda:9.0-runtime
RUN apt-get -qq update &&\
    apt-get -qq -y install curl &&\
    apt-get -qq -y install libcudnn7=7.0.5.15-1+cuda9.1 &&\
    apt-get -qq install -y python3 &&\
    apt-get -qq install -y python3-pip
ADD requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
ADD . /src
WORKDIR /src
CMD ["python3", "start.py"]