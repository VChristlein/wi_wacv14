# wi_wacv14
Writer Identification used for the WACV 2014 paper

Please cite: 
Christlein, V.; Bernecker, D.; Honig, F.; Angelopoulou, E.,
"Writer identification and verification using GMM supervectors," in
Applications of Computer Vision (WACV), 2014 IEEE Winter Conference on , vol.,
no., pp.998-1005, 24-26 March 2014 doi: 10.1109/WACV.2014.6835995

Required Packages: progressbar, OpenCV Version 2.4.x

The identification uses 3 steps
1. Feature Extraction (feat_ex.py) of test and train data
2. Clustering (clustering.py) of the train data
3. Encoding (ubm_adaptation.py) uses cluster of the train data and encodes test
   data

In advance you need to create a label-file for your data, which contains
in each row the name of the image-file and the label (i.e. the writer id).

