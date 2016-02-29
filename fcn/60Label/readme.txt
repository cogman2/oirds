Each of these sub-directories contain the training prototxt and solver configurations.   Each directory represents a cut-off of learning the lrate.
Learning across the entire network is conv1to7.  IN ALL CASES, the network is initialized with the pre-trained model.
convup - learning only the last set of layers after fc7
conv7 - learning from fc7 to the end 
con6to7 - learning from fc6 to the end
etc.
