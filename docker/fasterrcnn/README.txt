These are docker build files to create and run Faster RCNN on caffe.
There are two basic versions, a gpu version and a cpu version. Each is in its respective directory oirds/docker/gpu and oirds/docker/cpu.

To build the full docker image go to that directory and type the command:
docker build -t rsneddon/cudapython -f cudaDockerfile .
(Note that this build includes jupyter and tini; you can comment them out if you don't need them.)
This will build the cuda+anaconda image. Then build the Faster RCNN:
docker build -t rsneddon/fasterrcnn -f fasterRcnnDockerfile .

If you want to run the basic data pull scripts and correct a few bugs go to the directory,
oirds/docker and type
docker build -t rsneddon/lightvoccudapython -f VOCLightDockerfile .

You can also do a larger build which adds all of the VOC2007 images. However this build will crash unless you allocate more memory to docker, say 20G. Then type:
docker build -t rsneddon/VOCcudapython -f VOCDockerfile .

Please let me know of any bugs you find, sneddon_robert@bah.com
