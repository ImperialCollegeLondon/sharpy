FROM centos:8

ARG conda_env=sharpy_env

# Development tools including compilers
RUN yum groupinstall "Development Tools" -y
RUN yum install -y mesa-libGL
RUN yum install -y libXt libXt-devel
RUN yum install -y wget

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

ENV PATH="/miniconda3/bin/:${PATH}"
RUN conda init bash
# Update conda and make it run with no user interaction
RUN conda config --set always_yes yes --set changeps1 no
RUN conda update -q conda


# Set env variables
RUN source ~/.bashrc

# Make the sharpy_env environment
RUN conda env create -f sharpy/utils/environment_linux.yml

RUN yum install -y gcc-gfortran

RUN conda activate sharpy_env
ENV PATH="/miniconda3/envs/sharpy_env/bin:${PATH}"
RUN conda install -c conda-forge lapack

RUN git clone https://github.com/imperialcollegelondon/xbeam --branch=master
RUN conda activate sharpy_env && cd xbeam/ && sh run_make.sh && cd ..
RUN git clone https://github.com/imperialcollegelondon/uvlm --branch=master
RUN conda activate sharpy_env && cd uvlm/ && sh run_make.sh && cd ..

COPY ./utils/docker/* /root/

ENTRYPOINT ["/bin/bash", "--init-file", "/root/bashrc"]
#ENTRYPOINT ["/bin/bash"]

