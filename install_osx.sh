# OS X Installation with Anaconda 3.5
# http://stackoverflow.com/questions/24405561/how-to-install-2-anacondas-python-2-7-and-3-4-on-mac-os-10-9
conda create -n python2 python=2.7 anaconda
source activate python2

# **BLAS**: already installed as the Accelerate / vecLib Framework
# **Python** (optional): Anaconda is the preferred Python.
# **CUDA**: Install via the NVIDIA package that includes both 
# CUDA and the bundled driver. **CUDA 7 is strongly suggested.
# ** Older CUDA require `libstdc++` while clang++ is the default
#    compiler and `libc++` the default standard library on OS X 
#    10.9+. This disagreement makes it necessary to change the 
#    compilation settings for each of the dependencies. This is 
#    prone to error.
# **Library Path**: We find that everything compiles successfully 
#    if `$LD_LIBRARY_PATH` is not set at all, and
#    `$DYLD_FALLBACK_LIBRARY_PATH` is set to to provide CUDA, Python,
#    and other relevant libraries (e.g. 
#    `/usr/local/cuda/lib:$HOME/anaconda/lib:/usr/local/lib:/usr/lib`).
#   In other `ENV` settings, things may not work as expected.
brew install --fresh -vd snappy leveldb gflags glog szip lmdb imagemagick
brew tap homebrew/science
brew install -y cmake hdf5 opencv

# If using Anaconda Python, a modification to the OpenCV formula 
# might be needed.  Do `brew edit opencv` and change the lines 
# that look like the two lines below to exactly the two lines below.
brew edit opencv
#   -DPYTHON_LIBRARY=#{py_prefix}/lib/libpython2.7.dylib
#   -DPYTHON_INCLUDE_DIR=#{py_prefix}/include/python2.7

brew install --build-from-source --with-python --fresh -vd protobuf
brew install --build-from-source --fresh -vd boost boost-python

# Compile Caffe.
conda install cython
pip install easydict
git clone --recursive https://github.com/rbgirshick/fast-rcnn.git
cd fast-rcnn
echo "export FRCN_ROOT=$PWD" >> ~/.bashrc
source ~/.bashrc
cd caffe-fast-rcnn
sed 's/# WITH_PYTHON_LAYER := 1/WITH_PYTHON_LAYER := 1/' Makefile.config.example > Makefile.config
cd ../lib
make
cd ../caffe-fast-rcnn
make -j8 && make pycaffe
cd ..
./data/scripts/fetch_fast_rcnn_models.sh # optional

# Download the Overhead Imagery Research Dataset.
mkdir -p /data/oirds/png
wget http://sourceforge.net/projects/oirds/files/OIRDS%20-%20Vehicles/1.0/OIRDS_v1_0.zip
unzip OIRDS_v1_0.zip -d /data/oirds

# Use ImageMagick to convert from 
./convert_to_png.sh

# Crop.
python crop.py 40

# Create bounding boxes.
mkdir -p /data/object_proposals
python obj_proposals.py