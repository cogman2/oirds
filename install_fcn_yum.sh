#!/bin/bash
# Install FCN-Xs on CentOS
# forked from BVLC/caffe
#  Branch: future  caffe/docs/install_yum.md
# e4aed04  on Jul 28
# @shelhamer shelhamer [docs] fix lmdb fetch url and path
# 1 contributor
# RawBlameHistory     46 lines (33 sloc)  1.69 KB
# ---
# title: Installation: RHEL / Fedora / CentOS
# ---
# RHEL / Fedora / CentOS Installation
# General dependencies
sudo yum update
sudo yum install -y epel-release git emacs protobuf-devel \
    leveldb-devel snappy-devel opencv-devel boost-devel hdf5-devel \
    gflags-devel glog-devel lmdb-devel atlas-devel
#    python-devel 

# Python - see continuum io for the latest
wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda2-2.4.0-Linux-x86_64.sh
bash Anaconda2-2.4.0-Linux-x86_64.sh
conda install -y opencv
pip install easydict

# # Remaining dependencies, if not found
# # glog
# wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
# tar zxvf glog-0.3.3.tar.gz
# cd glog-0.3.3
# ./configure
# make && make install
# # gflags
# wget https://github.com/schuhschuh/gflags/archive/master.zip
# unzip master.zip
# cd gflags-master
# mkdir build && cd build
# export CXXFLAGS="-fPIC" && cmake .. && make VERBOSE=1
# make && make install
# # lmdb
# git clone https://github.com/LMDB/lmdb
# cd lmdb/libraries/liblmdb
# make && make install
# # Note that glog does not compile with the most recent gflags version (2.1), so before that is resolved you will need to build with glog first.

#Python (optional): if you use the default Python you will need to install the dev package to have the Python headers for building the pycaffe wrapper.

# Install CUDA
# http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Installation_Guide_Linux.pdf
# http://stackoverflow.com/questions/1911109/clone-a-specific-git-branch
git clone -b future https://github.com/longjon/caffe.git caffe

#CUDA: Install via the NVIDIA package instead of yum to be certain of the library and driver versions. Install the library and latest driver separately; the driver bundled with the library is usually out-of-date. + CentOS/RHEL/Fedora:
# BLAS: install ATLAS or install OpenBLAS or MKL for better CPU performance. For the Makefile build, uncomment and set BLAS_LIB accordingly as ATLAS is usually installed under /usr/lib[64]/atlas).
