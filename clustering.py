import os
import gzip
import cPickle
import argparse
import sys
from sklearn import mixture
import puhma_common as pc

def fitGMM(data, num_clusters, iterations, update, covar_type):
    print ('fit gmm with num_clusters {}, params {}, n_iter {}, covar_type '
        '{}'.format(num_clusters, update, iterations, covar_type))
    gmm = mixture.GMM(num_clusters, n_iter=iterations,
                      params=update, covariance_type=covar_type)
    gmm.fit(data)
    return gmm

def computeVocabulary(descriptors, method, num_clusters, iterations, update,
                      covar_type):
    print 'compute now vocabulary' 
    if 'gmm' == method:
        gmm = fitGMM(descriptors, num_clusters,
                         iterations, update, covar_type)
        return gmm
   
def parserArguments(parser):	
    parser.add_argument('--max_descriptors', nargs='*',
                        type=int, default=[150000],
                        help='load maximum descriptors')
    parser.add_argument('--num_clusters', type=int, default=500,\
                        help='number of cluster-centers = size of vocabulary')
    parser.add_argument('--vocabulary_filename', default='ubm',\
                        help='write vocabulary to this file')
    parser.add_argument('--method', default='gmm',\
                        choices=['gmm'],
                        help=('method for clustering'))
    parser.add_argument('--iterations', type=int, default=100,\
                        help=' number of iterations (if gmm, this is the gmm '
                        'part, not the kmeans-initialization part)')
    parser.add_argument('--update', default='wmc',\
                        help='what to update w. GMM, w:weights, m:means, c:covars')
    parser.add_argument('--covar_type', default='diag',
                        choices=['full','diag'],
                        help='covariance type for gmm')
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clustering - Create vocabulary")
    parser = pc.commonArguments(parser)
    parser = parserArguments(parser)
    args = parser.parse_args()

    if not os.path.exists(args.outputfolder):
        pc.mkdir_p(args.outputfolder)

    files,_ = pc.getFiles(args.inputfolder, args.suffix, args.labelfile,
                        exact=True)
    if not files or len(files) == 0:
        print 'getFiles() returned no images'
        sys.exit(1)

    # load features to train a universal background gmm
    print 'load features for training ubm from {} files'.format(len(files))

    descriptors = pc.loadDescriptors(files,\
                                     max_descs=args.max_descriptors[0],
                                     max_descs_per_file=max(int(args.max_descriptors[0]/len(files)),\
                                                            1), 
                                     rand=True, \
                                     hellinger=args.hellinger) 
    print 'got {} features'.format(len(descriptors))
    print 'features.shape', descriptors.shape

    vocabulary = computeVocabulary(descriptors, args.method, args.num_clusters, 
                                   args.iterations, args.update,
                                   args.covar_type) 

    # save gmm
    voc_filepath = os.path.join(args.outputfolder, args.vocabulary_filename +\
                                '.pkl.gz')
    with gzip.open(voc_filepath, 'wb') as f:
        cPickle.dump(vocabulary, f, -1)
        print 'saved vocabulary at', voc_filepath

