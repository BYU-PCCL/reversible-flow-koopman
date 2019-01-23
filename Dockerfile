FROM nvcr.io/nvidia/pytorch:18.12-py3

RUN pip install --upgrade pip
RUN pip install Cython
RUN pip install py3nvml
RUN pip install colorama
RUN pip install line_profiler
RUN pip install tensorboardX
RUN pip install einops
RUN pip install opencv-python
RUN pip install pygame
RUN pip install pymunk

RUN echo 'PS1="🐋 \[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> /root/.bashrc

CMD ["/bin/bash"]
