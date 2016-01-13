import numpy as np
import scipy.spatial.distance as spdistance
import puhma_common as pc
import sys
import argparse


def computeDistance(X, Y, method):
    if method == 'cosine':
        dist = spdistance.cosine(X,Y)
    
    if dist < 0:
        print ('WARNING: distance between X {} and Y {} = {} < 0, method: '
                         '{}'.format(X, Y, dist, method))

    return dist

def computeDistances(descriptors, method, parallel=True,
                     distance_func=None):
    num_desc = len(descriptors)

    indices = [(y,x) for y in range(num_desc-1) for x in range(y+1, num_desc)]
    splits = np.array_split(np.array(indices), 8)
    def loop(inds): 
        dists = []
        for ind in inds:
            if distance_func == None:
                try:
                    dist = computeDistance(descriptors[ ind[0] ],descriptors[ ind[1] ], method)
                except:
                    print 'method {} failed'.format(method)
                    raise
            else: 
                dist = distance_func( descriptors[ ind[0] ],descriptors[ ind[1] ] )
            dists.append(dist)
        return dists

    if parallel:
        dists = pc.parmap(loop, splits)
    else:
        dists = map(loop, splits) 
  
    # convert densed vector-form to matrix
    dense_vector = np.concatenate( dists )
    if spdistance.is_valid_y(dense_vector, warning=True):
        dist_matrix = spdistance.squareform( dense_vector )
    else:
        print 'ERROR: not a valid condensed distance matrix!'
        n = dense_vector.shape[0]
        d = int(np.ceil(np.sqrt(n * 2)))
        should = d * (d - 1) / 2
        print '{} != {}, num: {}'.format(should, n, num_desc)
        sys.exit(1)
        
    # fill diagonal elements with max
    np.fill_diagonal(dist_matrix, np.finfo(float).max)
    return dist_matrix 

def computeStats( name, dist_matrix, labels,  
                 parallel=True ):
    """ 
    compute TOP1 and mAP of dist_matrix via given labels
    """

    num_descr = dist_matrix.shape[0]
    if parallel:
        def sortdist(split):
            return split.argsort()
        splits = np.array_split(dist_matrix, 8) # todo assume 8 threads
        indices = pc.parmap(sortdist, splits)
        indices = np.concatenate(indices, axis=0)
    else:
        indices = dist_matrix.argsort()

    def loop_descr(r):
        # compute TOP-1 accuracy (AP)
        correct = 0
        for k in xrange(1):
            if labels[ indices[r,k] ] == labels[ r ] :
                correct += 1

        # compute mAP
        rel = 0
        avg_precision = []
        for k in range(0,num_descr-1): # don't take last one, since this is the
                                       # element itself
            if labels[ indices[r,k] ] == labels[ r ]:
                rel += 1
                avg_precision.append( rel / float(k+1) )
        return correct, np.mean(np.array(avg_precision))

    if parallel:
        top1_correct, query_precisions = zip( *pc.parmap(loop_descr, range(num_descr)) )
    else:
        top1_correct, query_precisions = zip( *map(loop_descr, range(num_descr)) )

    top1 = float(np.array(top1_correct).sum()) / float(num_descr)
    mAP = np.mean(np.array(query_precisions))
    print "NN {:10} TOP-1: {:7}  mAP: {:12}".format(name, top1, mAP)
    
    return top1, mAP

def runNN(descriptors, labels, parallel):
    """
    compute nearest neighbor from specific descriptors, given labels
    """

    distance_method = { "cosine": 'cosine' }
    ret_matrix = None
    for name, method in distance_method.iteritems():
        dist_matrix = computeDistances(descriptors, method, 
                                           parallel)

        computeStats(name, dist_matrix, labels, parallel)
        ret_matrix = dist_matrix

    return ret_matrix 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate stuff")
    parser = pc.commonArguments(parser)
    args = parser.parse_args()

    descr_files, labels = pc.getFiles(args.inputfolder, args.suffix, args.labelfile,
                                 exact=True)
    descriptors = pc.loadDescriptors(descr_files)

    ret_matrix = runNN( descriptors, labels, args.parallel )
