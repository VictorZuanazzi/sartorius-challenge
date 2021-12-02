nFROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

RUN apt upgrade -y

RUN apt install wget -y vim -y

WORKDIR /sartorius

COPY requirements.txt /sartorius/requirements.txt

RUN pip install -r requirements.txt

COPY . /sartorius

EXPOSE 31000

