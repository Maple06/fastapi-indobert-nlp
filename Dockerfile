FROM --platform=linux/amd64 python:3.7.4

COPY ./ ./

RUN python -m pip install -U pip wheel

RUN python -m pip install -r ./requirements/base.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "2323", "--workers", "5"]