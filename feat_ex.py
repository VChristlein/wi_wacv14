import progressbar
import os
import sys
import puhma_common as pc
import argparse
import cPickle
import gzip
import cv2

"""
extract features for images
"""

class FeatureEx(object):
    """
    wrapper class, currently only around cv2 functionality, but in this
    way easily extendable for other features
    """

    def __init__(self, detector_name, feat_type):
        self.feat_type = feat_type        
        self.detector = cv2.FeatureDetector_create(detector_name)
        self.descriptor_ex = cv2.DescriptorExtractor_create(feat_type)

    def detect(self, img):
       kpts = self.detector.detect( img )
       return kpts

    def extract(self, img, keypoints):
        keypoints, img_features = self.descriptor_ex.compute(img, keypoints)
        return keypoints, img_features

def parseArgs(parser):
    feat_group = parser.add_argument_group('feature options')
    feat_group.add_argument('--detector', default='SIFT',\
                            help='detector type')
    feat_group.add_argument('--feature', '--descriptor', 
                            default='SIFT',\
                            help='feature type')
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Some Feature Extractionmethods")
    parser = pc.commonArguments(parser)
    parser = parseArgs(parser)
    args = parser.parse_args()

    if not os.path.exists(args.outputfolder):
        pc.mkdir_p(args.outputfolder)
        
    files,_ = pc.getFiles(args.inputfolder, args.suffix, args.labelfile)
    if not files or len(files) == 0:
        print 'getFiles() returned no images'
        sys.exit(1)

    all_features = []

    fe = FeatureEx(args.detector, args.feature)

    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
               progressbar.ETA()]
    progress = progressbar.ProgressBar(widgets=widgets,
                                      maxval=len(files))

    progress.start()
    def compute(i):
        img_file = files[i]
        img = cv2.imread(img_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        if img is None:
            print 'img {} is None, path correct? --> skip'.format(img_file)
            return

        kpts = fe.detect(img)
        kpts, descriptors = fe.extract(img, kpts)

        # output
        new_basename = os.path.join(args.outputfolder,
                                    os.path.basename(os.path.splitext(img_file)[0]))
        feat_filename = new_basename + '_' + args.detector \
                        + '_' + args.feature + '.pkl.gz'
        with gzip.open(feat_filename, 'wb') as f:
            cPickle.dump(descriptors, f, -1)

        progress.update(i+1)

    if args.parallel:
        pc.parmap(compute, range(len(files)))
    else:
        map(compute, range(len(files)))
    progress.finish()
