FROM centos:8

ENV PYTHONDONTWRITEBYTECODE=true
ENV BASH_ENV ~/.bashrc
SHELL ["/bin/bash", "-c"]
ENV PATH=${PATH}:/miniconda3/bin

# Development tools including compilers
RUN yum groupinstall "Development Tools" -y --nogpgcheck && \
    yum install -y --nogpgcheck mesa-libGL libXt libXt-devel wget gcc-gfortran lapack vim tmux && \
    yum clean all

# Install miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    chmod +x /miniconda.sh && \
    /miniconda.sh -b -p /miniconda3/ && \
    rm /miniconda.sh && hash -r

ADD / /sharpy_dir/

# Update conda and make it run with no user interaction
# Cleanup conda installation
RUN conda init bash && \
    conda config --set always_yes yes --set changeps1 no && \
    conda update -q conda && \
    conda config --set auto_activate_base false && \
    conda env create -f /sharpy_dir/utils/environment_minimal.yml && conda clean -afy && \
    find /miniconda3/ -follow -type f -name '*.a' -delete && \
    find /miniconda3/ -follow -type f -name '*.pyc' -delete && \
    find /miniconda3/ -follow -type f -name '*.js.map' -delete

#COPY /utils/docker/* /root/
RUN ln -s /sharpy_dir/utils/docker/* /root/

RUN cd sharpy_dir && \
    conda activate sharpy_minimal && \
    mkdir build && \
    cd build && \
    CXX=g++ FC=gfortran cmake .. && make install -j 2 && \
    cd .. && \
    rm -rf build

ENTRYPOINT ["/bin/bash", "--init-file", "/root/bashrc"]

