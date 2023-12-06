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

# Install Mamba
# Swapped from Conda to Mamba due to Github runner memory constraint
RUN wget --no-check-certificate https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -O /mamba.sh && \
    chmod +x /mamba.sh && \
    /mamba.sh -b -p /mamba/ && \
    rm /mamba.sh && hash -r

ADD / /sharpy_dir/

# Update mamba and make it run with no user interaction
# Cleanup mamba installation
RUN mamba init bash
RUN mamba config --set always_yes yes --set changeps1 no
RUN mamba update -q conda
RUN mamba config --set auto_activate_base false
#RUN mamba env create -f /sharpy_dir/utils/environment.yml && mamba clean -afy && \
#    find /mamba/ -follow -type f -name '*.a' -delete && \
#    find /mamba/ -follow -type f -name '*.pyc' -delete && \
#    find /mamba/ -follow -type f -name '*.js.map' -delete
RUN mamba env create -f /sharpy_dir/utils/environment.yml
#RUN mamba clean -afy
RUN find /mamba/ -follow -type f -name '*.a' -delete
RUN find /mamba/ -follow -type f -name '*.pyc' -delete
RUN find /mamba/ -follow -type f -name '*.js.map' -delete

#COPY /utils/docker/* /root/
RUN ln -s /sharpy_dir/utils/docker/* /root/

RUN cd sharpy_dir && \
    mamba activate sharpy && \
    git submodule update --init --recursive && \
    mkdir build && \
    cd build && \
    CXX=g++ FC=gfortran cmake .. && make install -j 2 && \
    cd .. && \
    pip install . && \
    rm -rf build
    
ENTRYPOINT ["/bin/bash", "--init-file", "/root/bashrc"]

