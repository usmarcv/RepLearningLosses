# using pytorch with cuda 
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

# you can use only cpu python
#FROM python:3.12

ENV DEBIAN_FRONTEND noninteractive

# install essentials
RUN apt-get update -y && apt-get install -y \
	software-properties-common \
	build-essential \
	libblas-dev \
	libhdf5-serial-dev \
	git

# install some niceties to help development
RUN apt-get update -y && apt-get install -y zsh tmux htop vim
RUN pip3 install -U pip 
RUN pip3 install -U pipenv
RUN ln -s /usr/bin/python3 /usr/bin/python

# make terminals look pretty (setting a reasonable colour setting)
RUN touch /usr/share/locale/locale.alias
RUN apt-get -y install locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8
ENV TERM xterm-256color

# setting your work directory (you can update and add some files)
WORKDIR /root
COPY . /root

CMD ["bin/bash"]
