FROM centos:8

ENV PYTHONDONTWRITEBYTECODE=true
ENV BASH_ENV ~/.bashrc
SHELL ["/bin/bash", "-c"]

# CENTOS 8 has reached end of life - Not yet an updated Docker base for CentOS stream
# Point to the CentOS 8 vault in order to download dependencies
RUN cd /etc/yum.repos.d/ && \
    sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-* && \
    sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-* && \
    cd /

# Development tools including compilers
RUN yum groupinstall "Development Tools" -y --nogpgcheck && \
    yum install -y --nogpgcheck mesa-libGL libXt libXt-devel wget gcc-gfortran lapack vim tmux cmake && \
    yum clean all

RUN yum-config-manager  --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo && \
    yum install intel-mkl-2020.0-088

ADD / /sharpy_dir/

#COPY /utils/docker/* /root/
RUN ln -s /sharpy_dir/utils/docker/* /root/

RUN cd sharpy_dir && \
    git submodule update --init --recursive && \
    mkdir build && \
    cd build && \
    CXX=g++ FC=gfortran cmake .. && make install -j 2 && \
    cd .. && \
    pip install . && \
    rm -rf build
    
ENTRYPOINT ["/bin/bash", "--init-file", "/root/bashrc"]