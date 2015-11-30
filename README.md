
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