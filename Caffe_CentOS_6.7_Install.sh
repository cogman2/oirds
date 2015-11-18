####NOTES on installing Caffe on CentOS 6.7####
# install rpmforge
wget http://pkgs.repoforge.org/rpmforge-release/rpmforge-release-0.5.3-1.el6.rf.x86_64.rpm
rpm -i rpmforge-release-0.5.3-1.el6.rf.x86_64.rpm
# modify /etc/yum.repos.d/rpmforge.repo to enable rpmforge-extras

yum install -y  protobuf-devel leveldb-devel snappy-devel hdf5-devel
yum install -y gflags-devel lmdb-devel
yum install -y gcc-c++ gcc atlas-devel
yum install -y cmake gtk2-devel pkgconfig
yum install -y qt3-devel
yum install -y libtiff-devel
yum install -y sqlite-devel
yum install -y openssl-devel
yum install -y doxygen
yum install -y bzip2-devel
yum install -y ImageMagick-devel
yum install -y freetype-devel
yum install -y libpng-devel

# install glog
wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
tar xvf glog-0.3.3.tar.gz
cd glog-0.3.3
./configure
make -j8
make install

# install python
yum install -y tk-devel readline-devel
curl -O https://www.python.org/ftp/python/2.7.10/Python-2.7.10.tgz
tar xvf Python-2.7.10.tgz
cd Python-2.7.10
./configure --enable-shared --prefix=/usr/local LDFLAGS="-Wl,--rpath=/usr/local/lib" --enable-unicode=ucs4
make -j8
make install
PATH=/usr/local/lib:$PATH
# echo "PATH=/usr/local/lib:$PATH" >> /home/$USER/.bashrc

# install easy-install and pip
wget --no-check-certificate http://pypi.python.org/packages/source/d/distribute/distribute-0.6.35.tar.gz
tar xvf distribute-0.6.35.tar.gz
cd distribute-0.6.35
python setup.py install
easy_install-2.7 pip
pip2.7 install numpy cython scikit-image

# install cuda (obtained rpm from NVIDIA)
#wget http://developer.download.nvidia.com/compute/cuda/repos/rhel6/x86_64/cuda-repo-rhel6-7.0-28.x86_64.rpm
# yum remove -y `cat cuda_remove`
wget -r --no-parent wget http://developer.download.nvidia.com/compute/cuda/repos/rhel6/x86_64
rpm -i cuda-repo-rhel6-7.0-28.x86_64.rpm
#wget http://developer.download.nvidia.com/compute/cuda/repos/rhel6/x86_64/cuda-repo-rhel6-7.5-18.x86_64.rpm
#rpm -i cuda-repo-rhel6-7.5-18.x86_64.rpm
yum clean all
yum -y install cuda

# install cudnn (need to sign up as an NVIDIA developer)
# https://developer.nvidia.com/rdp/cudnn-download
tar xvf cudnn-7.0-linux-x64-v3.0-rc.tar
cd cuda
cp include/cudnn.h /usr/local/cuda/include/
cp lib64/libcudnn* /usr/local/cuda-7.0/lib64/

# install boost (from boost.org)
# wget http://sourceforge.net/projects/boost/files/boost/1.59.0/boost_1_59_0.tar.gz
tar xf boost_1_59_0.tar.gz
cd boost_1_59_0/
./bootstrap.sh
./b2 -j 8
./b2 install

# compile FFmpeg (see note below in OpenCV)
yum install -y yasm-devel libva-devel libass-devel libkate-devel \
    libbluray-devel libdvdnav-devel libcddb-devel libmodplug-devel \
    a52dec-devel libmpeg2-devel
git clone https://github.com/FFmpeg/FFmpeg.git
cd FFmpeg
git checkout n1.0
mkdir build
cd build/
../configure --enable-shared --prefix=/opt/ffmpeg
make -j8
make install

# install OpenCV
#yum -y install lynx
#lynx https://codeload.github.com/Itseez/opencv/zip/2.4.11
wget https://codeload.github.com/Itseez/opencv/zip/2.4.11
# save download file
#unzip opencv-2.4.11.zip
unzip 2.4.11
cd opencv-2.4.11
# apparently, FFMPEG isn't needed ... so you can also just pass the -D WITH_FFMPEG=OFF flag to cmake as well
cmake -D CUDA_GENERATION=Kepler .
make -j8
make install

# install caffe
git clone https://github.com/BVLC/caffe.git
cd caffe
sed -i "s|Atlas_LAPACK_LIBRARY NAMES|Atlas_LAPACK_LIBRARY NAMES lapack|g" cmake/Modules/FindAtlas.cmake
cmake -DCMAKE_INSTALL_PREFIX:PATH=/opt/caffe .
make -j8
make install
yum groupinstall -y "Development Tools" | tee yum_dev_tools.log
yum install -y kernel-devel | tee yum_kernel-devel.log

# Download the Overhead Imagery Research Dataset.
mkdir -p /data/oirds/png
wget http://sourceforge.net/projects/oirds/files/OIRDS%20-%20Vehicles/1.0/OIRDS_v1_0.zip
unzip OIRDS_v1_0.zip -d /data/oirds

# Use ImageMagick to convert from tif to png.
./convert_to_png.sh

# Crop.
python crop.py 40

# Get the ImageNet pre-trained model for fine-tuning.
wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
mv bvlc_reference_caffenet.caffemodel /opt/caffe/models/finetune_flickr_style