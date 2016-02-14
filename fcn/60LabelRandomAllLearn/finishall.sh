GT=gt2
GTNEXT=gt3

mkdir $GT
mv run.out $GT/run.out

cp test_mean.binaryproto  $GT
cp train_mean.binaryproto $GT
cp test_label.json $GT


mkdir $GT/conv1to7
mkdir $GT/conv2to7
mkdir $GT/conv3to7
mkdir $GT/conv4to7
mkdir $GT/conv5to7
mkdir $GT/conv6to7
mkdir $GT/conv7
mkdir $GT/convup

mv conv1to7/train_iter_15000.caffemodel  $GT/conv1to7
mv conv2to7/train_iter_15000.caffemodel  $GT/conv2to7
mv conv3to7/train_iter_15000.caffemodel  $GT/conv3to7
mv conv4to7/train_iter_15000.caffemodel  $GT/conv4to7
mv conv5to7/train_iter_15000.caffemodel  $GT/conv5to7
mv conv6to7/train_iter_15000.caffemodel  $GT/conv6to7
mv conv7/train_iter_15000.caffemodel  $GT/conv7
mv convup/train_iter_15000.caffemodel  $GT/convup

cp conv1to7/deploy.prototxt  $GT/conv1to7
cp conv2to7/deploy.prototxt  $GT/conv2to7
cp conv3to7/deploy.prototxt  $GT/conv3to7
cp conv4to7/deploy.prototxt  $GT/conv4to7
cp conv5to7/deploy.prototxt  $GT/conv5to7
cp conv6to7/deploy.prototxt  $GT/conv6to7
cp conv7/deploy.prototxt  $GT/conv7
cp convup/deploy.prototxt  $GT/convup

mv conv1to7/train_iter_20000.caffemodel  $GT/conv1to7
mv conv2to7/train_iter_20000.caffemodel  $GT/conv2to7
mv conv3to7/train_iter_20000.caffemodel  $GT/conv3to7
mv conv4to7/train_iter_20000.caffemodel  $GT/conv4to7
mv conv5to7/train_iter_20000.caffemodel  $GT/conv5to7
mv conv6to7/train_iter_20000.caffemodel  $GT/conv6to7
mv conv7/train_iter_20000.caffemodel  $GT/conv7
mv convup/train_iter_20000.caffemodel  $GT/convup

find conv* -name "*.caffemodel" -print -exec rm {} \;
find conv* -name "*.solverstate" -print -exec rm {} \;
find conv* -name "run.out" -print -exec rm {} \;
find conv* -name "nohup.out" -print -exec rm {} \;
find conv* -name "*.csv" -print -exec rm {} \;

rm -rf raw_t* groundtruth_t* test_mean.binaryproto  train_mean.binaryproto 
cp -r ~/dd/$GTNEXT/* .

cd $GT
source ~/.bashrc
#(nohup python ~/oirds/tools/fcn/python/test_all_label.py test_label.json > stats.out) & 
