FROM nvidia/cuda:11.4.0-base-ubuntu20.04

COPY ./ ./

RUN apt update
RUN apt-get install -y python3 python3-pip

RUN python3 -m pip install -U pip wheel

RUN python3 -m pip install -r ./requirements/base.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "2323", "--workers", "5"]