FROM centos:8

ARG conda_env=sharpy_env

# Development tools including compilers
RUN yum groupinstall "Development Tools" -y
RUN yum install -y mesa-libGL
RUN yum install -y libXt libXt-devel
RUN yum install -y wget
RUN yum install -y gcc-gfortran
RUN yum install -y lapack
RUN yum clean all

ENV BASH_ENV ~/.bashrc
SHELL ["/bin/bash", "-c"]

# Install miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh
RUN chmod +x /miniconda.sh
RUN /miniconda.sh -b -p /miniconda3/
RUN rm /miniconda.sh
RUN hash -r

# Get SHARPy
RUN git clone https://github.com/imperialcollegelondon/sharpy --branch=dev_docker

ENV PATH=${PATH}:/miniconda3/bin
RUN conda init bash
# Update conda and make it run with no user interaction
RUN conda config --set always_yes yes --set changeps1 no
RUN conda update -q conda
RUN conda config --set auto_activate_base false

# Make the sharpy_env environment
RUN conda env create -f sharpy/utils/environment_linux.yml && conda clean -afy

COPY ./utils/docker/* /root/

RUN conda activate sharpy_env 

RUN echo "Building the libraries"
RUN git clone https://github.com/imperialcollegelondon/xbeam --branch=master
RUN conda activate sharpy_env && cd xbeam/ && sh run_make.sh && cd ..
RUN git clone https://github.com/imperialcollegelondon/uvlm --branch=master
RUN conda activate sharpy_env && cd uvlm/ && sh run_make.sh && cd ..
RUN rm -rf xbeam uvlm

ENTRYPOINT ["/bin/bash", "--init-file", "/root/bashrc"]

