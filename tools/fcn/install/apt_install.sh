#!/bin/bash
# Fork 3,896 longjon/caffe
# forked from BVLC/caffe
#  Branch: future  caffe/docs/install_apt.md
# e4aed04  on Jul 28
# @shelhamer shelhamer [docs] fix lmdb fetch url and path
# 4 contributors @shelhamer @lukeyeager @EricZeiberg @semitrivial
# RawBlameHistory     51 lines (36 sloc)  1.75 KB
# ---
# title: Installation: Ubuntu 14.04
# ---
# Ubuntu Installation
# General dependencies

sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
# CUDA: Install via the NVIDIA package instead of apt-get to be certain of the library and driver versions. Install the library and latest driver separately; the driver bundled with the library is usually out-of-date. This can be skipped for CPU-only installation.

# BLAS: install ATLAS or install OpenBLAS or MKL for better CPU performance.
sudo apt-get install libatlas-base-dev 

#Python (optional): if you use the default Python you will need to sudo apt-get install the python-dev package to have the Python headers for building the pycaffe interface.

# Remaining dependencies
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev

cd caffe
# http://stackoverflow.com/questions/28985551/caffe-installation-in-ubuntu-14-04
sed -i 's/using std::signbit/\/\/ using std::signbit/' include/caffe/util/math_functions.hpp
sed -i 's/signbit\(x\[/std::signbit\(x\[/' include/caffe/util/math_functions.hpp
make all -j8 # lscpu
make test
make runtest

# Download prototxts, cf. https://gist.github.com/shelhamer/91eece041c19ff8968ee
mkdir models
wget -P models https://gist.githubusercontent.com/shelhamer/91eece041c19ff8968ee/raw/829ca42202f21c884c13953dd0f1d484593f1b27/deploy.prototxt \
    http://dl.caffe.berkeleyvision.org/fcn-8s-pascalcontext.caffemodel \
    https://gist.githubusercontent.com/shelhamer/91eece041c19ff8968ee/raw/829ca42202f21c884c13953dd0f1d484593f1b27/solver.prototxt \
    https://gist.githubusercontent.com/shelhamer/91eece041c19ff8968ee/raw/829ca42202f21c884c13953dd0f1d484593f1b27/train_val.prototxt
wget -P python https://gist.githubusercontent.com/shelhamer/91eece041c19ff8968ee/raw/829ca42202f21c884c13953dd0f1d484593f1b27/solve.py

