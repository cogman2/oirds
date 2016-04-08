FROM centos:7.2.1511

MAINTAINER Eric Robertson <rwgdrummer@gmail.com>:

ENV BLENDER_MAJOR 2.76
ENV BLENDER_VERSION 2.76b
ENV BLENDER_BZ2_URL http://download.blender.org/source/blender-2.77.tar.gz

RUN groupadd blender
RUN useradd -m -g blender -d /home/blender blender
RUN yum install -y sudo
RUN yum -y install wget

# Edit sudoers file # To avoid error: sudo: sorry, you must have a tty to run sudo
RUN sed -i -e "s/Defaults    requiretty.*/ #Defaults    requiretty/g" /etc/sudoers

RUN rpm -Uvh http://download.fedoraproject.org/pub/epel/7/x86_64/e/epel-release-7-5.noarch.rpm
RUN rpm -Uvh http://download1.rpmfusion.org/free/el/updates/6/i386/rpmfusion-free-release-6-1.noarch.rpm

RUN mkdir -p /home/blender/src/blender-deps
WORKDIR /home/blender/src/blender-deps/
RUN wget http://ffmpeg.org/releases/ffmpeg-2.8.4.tar.gz
RUN tar xvf ffmpeg-2.8.4.tar.gz

WORKDIR /home/blender
RUN wget $BLENDER_BZ2_URL
RUN tar xf blender-2.77.tar.gz
WORKDIR /home/blender/blender-2.77/build_files/build_environment

RUN mkdir -p /root/src
RUN ln -s /home/blender/src/blender-deps /root/src/blender-deps
#wget http://llvm.org/releases/3.6.2/clang+llvm-3.6.2-x86_64-fedora21.tar.xz
RUN printf 'y\n' | ./install_deps.sh --with-all --force-llvm --build-oiio --build-llvm --skip-osl --build-foo --build-ffmpeg

WORKDIR /home/blender/blender-2.77
RUN make
WORKDIR /home/blender
RUN rm blender-2.77.tar.gz && mkdir tmp && chown -R blender:blender * 

USER blender
ENV PYTHONHOME /home/blender/build_linux/bin/2.77/python/lib
ENV TMP /home/blender/tmp
ENV PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/blender/build_linux/bin

VOLUME /home/blender
#ENTRYPOINT ["/usr/local/blender/blender", "-b"]
CMD ["/bin/bash"]
