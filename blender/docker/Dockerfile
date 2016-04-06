FROM centos:7.2.1511

MAINTAINER Eric Robertson <rwgdrummer@gmail.com>:

ENV BLENDER_MAJOR 2.76
ENV BLENDER_VERSION 2.76b
ENV BLENDER_BZ2_URL http://download.blender.org/source/blender-2.77.tar.gz

cd /home/
groupadd blender
useradd -m -g blender -d /home/blender blender
yum install -y sudo
yum -y install wget

cd blender
wget $BLENDER_BZ2_URL
tar xf blender-2.77.tar.gz
cd blender-2.77/build_files/build_environment


su -c 'rpm -Uvh http://download.fedoraproject.org/pub/epel/7/x86_64/e/epel-release-7-5.noarch.rpm'
su -c 'yum install foo'
wget http://download1.rpmfusion.org/free/el/updates/6/i386/rpmfusion-free-release-6-1.noarch.rpm
su -c 'rpm -Uvh rpmfusion-free-release-6-1.noarch.rpm'
mkdir ~/src/blender-deps
pushd ~/src/blender-deps/
tar xvf ffmpeg-2.8.4.tar.gz
popd

#wget http://llvm.org/releases/3.6.2/clang+llvm-3.6.2-x86_64-fedora21.tar.xz

printf 'y\n' | ./install_deps.sh --with-all --force-llvm --build_oiio --build-llvm --skip-osl --build-foo --build-ffmpeg

cd ../../
make

rm blender-277.tar.gz

VOLUME /media
ENTRYPOINT ["/usr/local/blender/blender", "-b"]