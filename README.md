# wi_wacv14
Writer Identification used for the WACV 2014 paper.

Please cite: 
Christlein, V.; Bernecker, D.; Honig, F.; Angelopoulou, E.,
"Writer identification and verification using GMM supervectors," in
Applications of Computer Vision (WACV), 2014 IEEE Winter Conference on , vol.,
no., pp.998-1005, 24-26 March 2014 doi: 10.1109/WACV.2014.6835995

## Requirements
Required Python-Packages: progressbar, OpenCV Version 2.4.x

## Workflow 
The identification uses 3 steps.  In advance you need to create a label-file for your data, which contains
in each row the name of the image-file and the label (i.e. the writer id). (*Update* I provided label-files, see \*.txt files.)

* 1. Feature Extraction (feat_ex.py) of test and train data

python2 feat_ex.py -i /path/to/train -l train_label.txt -o /path/to/outtrain

python2 feat_ex.py -i /path/to/test -l test_label.txt -o /path/to/outtest

* 2. Clustering (clustering.py) of the train data

python2 clustering.py -l train_label.txt --suffix _SIFT_SIFT.pkl.gz -i /path/to/outtrain -o /path/to/outvoc 

* 3. Encoding (ubm_adaptation.py) uses cluster of the train data and encodes test
   data

python2 ubm_adaptation.py -o /path/to/test_encoding -l test_label.txt -i /path/to/test --suffix _SIFT_SIFT.pkl.gz --load_ubm /path/to/outtrain/ubm.pkl.gz --encoding supervector --normalize ssr l2g 


