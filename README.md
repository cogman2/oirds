
(1) For python scrips to run:
Changed import order/statement prior to import of cafee:
  import matplotlib
  matplotlib.use('Agg')

(2) File layout:
    1.  Load the metadata from .xls files.
        - gen_file_lists.py
    2.  Correct for ground sample distance variation and modify the centroid and polygon coordinates.
        - tools/prep.py; tools/img_manip.py
    3.  Remove vehicles with shadows.
        - tools/prep.py
    4.  Rotate the images in 15 degree increments and modify the centroid and polygon coordinates.
        - tools/prep.py; img_manip.py
    5.  Select patches of a given size around vehicles in the images.
        - tool/prep.py
    6.  Crop a 'no vehicle' patches from images with only one vehicle.
        - tools/prep.py
    7.  Create the corresponding training and validation set .txt files with a user defined train-test ratio.
        - tools/prep.py
    8.  Label OIRDS images by pixel (and plot the training images), as required for Fully Convolutional Networks for Semantic Segmentation.
        - tools/label_image.py
    9.  Convert .tif files to .png using ImageMagick.
        - tools/convert_to_png.sh
    10. Print output from the models via google logs.
        - tools/glogs.py
    11. Generate the Lightning Memory-mapped Databases (train and test) and training image mean.
        - tools/warm_up.py
    12. Achieve sweet accuracy with CNN on cropped images.
        - 'caffe train -solver models/small_solver.prototxt -gpu 0'
    13. Fine-tune a CNN on OIRDS.
        - tools/finetuning/oirds-fine-tuning.py
        
    *0. Install instructions and sample prototxts for Fully Convolutional Networks for Semantic Segmentation.
        - tools/fcn/
    *1. Tools for Region-based (sliding window) CNNs
        - tools/rcnn/
    *2. Describe the environment of a successful installation.
        - describe.sh; description.txt


F-CNN Work:
     1. Convert images to png files.  Place in png directory with top level Data Set.
     2. python  label_image.py /data/oirds
       -- creates label and data LMDB files (raw_train, raw_test, groundtruth_train, groundtruth_test)
     3. /opt/fcn/caffe/build/tools/compute_image_mean raw_train train_mean.binaryproto
     4. /opt/fcn/caffe/build/tools/compute_image_mean raw_test test_mean.binaryproto
     5. /opt/fcn/caffe/build/tools/caffe train -solver fcn8_solver.prototxt -weights fcn-8s-pascalcontext.caffemodel -gpu 0
       -- the model comes from https://gist.github.com/longjon/1bf3aa1e0b8e788d7e1d#file-readme-md     
     6. python test_label.py /data/oirds/ /data/oird_fcn 0.05
       -- tests a random sample of images from the /data/oirds data set. The images are assumed to be in a directory called 'png'.  The second argument is the data directory containing the deploy.prototxt, the train_itet_8000.caffemodel (from step 5), the train_mean.binaryproto file.
       -- This python routine dumps a stats.txt file containing tuples for each image: (x dim, y dim, false positive, false negative, true positive, true negative, wrong label, precision, recall, accuracy, f1).
       -- This python routine also dumps out a large amount of images for convolution weights, and an image for {image_name}_output.png showing the classification.