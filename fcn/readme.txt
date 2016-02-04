Steps for each of these training configuration sets:
(1) Create Traing DB
  python oirds/tools/fcn/python/create_train_dbs.py create_db.json
  *Recall that the training data must be in a directory with the Excel files and a png sub-directory with the images.
  * There are a host of data directories used to configure the ten fold cross validation.
(2) Compute the means
  /opt/fcn/bin/compute_image_mean.bin raw_train train_mean.binaryproto
  /opt/fcn/bin/compute_image_mean.bin raw_test test_mean.binaryproto
(3) Run the solver for each configuration
  cd convup
  nohup python oirds/tools/fcn/python/solver_with_netsurgery.py solver.json &
  *Check the solver.json for the location of the model and check the gpuID.
(4) Test
   (a) Individually (run from each configuration directory)
    python oirds/tools/fcn/python/test_label.py ../test_label.json
   (b) All (run from the parent)
    python oirds/tools/fcn/python/test_all_label.py ../test_label.json
