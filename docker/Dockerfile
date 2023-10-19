FROM ubuntu:20.04

ENV TZ=Europe/Belgrade
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    nano \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx unzip \
    curl cron ffmpeg \
    ssh openssh-client \
    vim

RUN pip3 install -U pip

# Install nodetracker
ADD requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Install YOLOX
RUN apt-get install -y git
RUN git clone https://github.com/Megvii-BaseDetection/YOLOX /YOLOX
WORKDIR /sudo apt-get install -y nvidia-container-toolkitYOLOX
RUN pip3 install -v -e .  # or  python3 setup.py develop

ENV PYTHONPATH="/node"

WORKDIR /node

CMD ["bash"]