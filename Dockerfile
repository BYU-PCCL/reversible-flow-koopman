FROM nvcr.io/nvidia/pytorch:18.04-py3

RUN pip install --upgrade pip
RUN pip install Cython
RUN pip install py3nvml
RUN pip install colorama
RUN pip install tensorboardX
RUN pip install --upgrade torch
