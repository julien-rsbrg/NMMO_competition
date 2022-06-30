FROM python:3.9.11

WORKDIR /tmp/install

# set the pip index-url you need
ARG NMMO_PYPI=https://mirrors.aliyun.com/pypi/simple/

COPY requirements_base.txt requirements_base.txt

RUN pip install --no-cache-dir -r requirements_base.txt -i ${NMMO_PYPI}

ARG NMMO_PYPI=https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements_nmmo.txt requirements_nmmo.txt

RUN pip install --no-cache-dir -r requirements_nmmo.txt -i ${NMMO_PYPI}
