FROM nvcr.io/nvidia/pytorch:18.09-py3

RUN pip install --upgrade pip
RUN pip install Cython
RUN pip install py3nvml

