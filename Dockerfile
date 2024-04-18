FROM centos:8

ENV PYTHONDONTWRITEBYTECODE=true
ENV BASH_ENV ~/.bashrc
SHELL ["/bin/bash", "-c"]
ENV PATH=${PATH}:/mamba/bin

# CENTOS 8 has reached end of life - Not yet an updated Docker base for CentOS stream
# Point to the CentOS 8 vault in order to download dependencies
RUN cd /etc/yum.repos.d/ && \
    sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-* && \
    sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-* && \
    cd /

# Development tools including compilers
RUN yum groupinstall "Development Tools" -y --nogpgcheck && \
    yum install -y --nogpgcheck mesa-libGL libXt libXt-devel wget gcc-gfortran lapack vim tmux && \
    yum clean all

# Install Mamba - swapped from Conda to Mamba due to Github runner memory constraint
RUN wget --no-check-certificate https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -O /mamba.sh && \
    chmod +x /mamba.sh && \
    /mamba.sh -b -p /mamba/ && \
    rm /mamba.sh && hash -r

ADD / /sharpy_dir/

# Initialise mamba installation
RUN mamba init bash && \
    mamba update -q conda && \
    mamba env create -f /sharpy_dir/utils/environment.yml && \
    find /mamba/ -follow -type f -name '*.a' -delete && \
    find /mamba/ -follow -type f -name '*.pyc' -delete && \
    find /mamba/ -follow -type f -name '*.js.map' -delete

RUN ln -s /sharpy_dir/utils/docker/* /root/

RUN cd sharpy_dir && \
    mamba activate sharpy && \
    git submodule update --init --recursive && \
    mkdir build && \
    cd build && \
    CXX=g++ FC=gfortran cmake .. && make install -j 4 && \
    cd .. && \
    pip install . && \
    rm -rf build
    
ENTRYPOINT ["/bin/bash", "--init-file", "/root/bashrc"]
