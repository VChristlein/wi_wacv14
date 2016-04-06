import os
import gzip
import cPickle
import argparse
import numpy as np
import sys
import progressbar
#import pyvole
import puhma_common as pc
from encoding import *
import evaluate

def parserArguments(parser):	
    ubm_group = parser.add_argument_group('adaptation options')
    ubm_group.add_argument('--load_ubm',\
                        help='filepath to pkl.gz file which contains the ubm-gmm')
    ubm_group.add_argument('--load_scores',\
                        help='load score-file (pkl.gz)')
    ubm_group.add_argument('--encoding',choices =['fisher', 
                                               'supervector',
                                               'vlad'],\
                        help='different encoding schemes of the gmm')
    ubm_group.add_argument('--normalize', nargs='*', default=[],
                        choices=['l2g','l2c','ssr'],
                        help='normalization options')
    ubm_group.add_argument('-r', '--relevance',type=int, default=16,\
                        help=('relevance factor for mixing feature vectors'\
                              ' with UBM'))
    parser.add_argument('--update', default='wmc',\
                        help='what to update w. GMM, w:weights, m:means, c:covars')
    parser.add_argument('--no_eval', action='store_true', 
                        help='dont evaluate skip evaluation and just write'
                        ' encodings ') 
    return parser

def loadUBM(ubm_file):
    with gzip.open(ubm_file, 'rb') as f:
        ubm_gmm = cPickle.load(f)
        if hasattr(ubm_gmm, 'weights_') and ubm_gmm.weights_.ndim > 1:
            ubm_gmm.weights_ = ubm_gmm.weights_.flatten()

    return ubm_gmm

if __name__ == '__main__':
#    import stacktracer
#    stacktracer.trace_start("trace.html",interval=5,auto=True)

    parser = argparse.ArgumentParser(description="UBM Adaption")
    parser = pc.commonArguments(parser)
    parser = parserArguments(parser)
    args = parser.parse_args()

    if not args.labelfile or not args.inputfolder or not args.outputfolder:
        print('WARNING: no labelfile or no inputfolder'
                                         ' or no outputfolder specified')
    if args.outputfolder and not os.path.exists(args.outputfolder):
        pc.mkdir_p(args.outputfolder)
    if not args.load_ubm:
        raise argparse.ArgumentTypeError('no gmm to load')

    #####
    # UBM-creation / loading
    print 'load gmm from', args.load_ubm
    ubm_gmm = loadUBM(args.load_ubm)

    #####
    # Enrollment
    # now for each feature-set adapt a gmm
    #####
    descriptor_files, labels = pc.getFiles(args.inputfolder, args.suffix,
                                               args.labelfile)

    if len(descriptor_files) == 0:
        print 'no descriptor_files'
        sys.exit(1)
    elif labels:
        num_scribes = len(list(set(labels)))
    else:
        num_scribes = 'unknown'

    num_descr = len(descriptor_files)
    print 'number of classes:', num_scribes
    print 'number of descriptor_files:', num_descr
    print 'adapt traing-features to create individual scribe-gmms (or load saved ones)'
    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
               progressbar.ETA()]
    progress = progressbar.ProgressBar(widgets=widgets,
                                       maxval=len(descriptor_files))

    if args.encoding == 'supervector':
        identifier = '_sv'
    elif 'fisher' in args.encoding:
        identifier = '_fv' 
    else: #vlad
        identifier = '_vlad'


    def encode(i):
        if isinstance(descriptor_files[i], basestring):            
            base = os.path.basename(os.path.splitext(descriptor_files[i])[0])
        else:
            base = os.path.basename(os.path.commonprefix(descriptor_files[i]))

        gmm_name = base + '_gmm.pkl.gz'         
        gmm = ubm_gmm
        
        # load encoding
        if args.load_scores:
            filepath = os.path.join(args.load_scores, base + identifier + '.pkl.gz')
            if os.path.exists(filepath):
                with gzip.open(filepath, 'rb') as f:
                    enc = cPickle.load(f)
                    return enc

        # load data and preprocess
        features = pc.loadDescriptors( descriptor_files[i],
                                      hellinger=args.hellinger,
                                      min_descs_per_file=args.min_descs,
                                      show_progress= True)
        if features is None:
            print 'WARNING: features==None ?!'
            progress.update(i+1)
            return 0.0
        
        # make the actual encoding step
        enc = encodeGMM(args.encoding, gmm, features, 
                             normalize=args.normalize, 
                             update=args.update, relevance=args.relevance )

        # save encoding
        filepath = os.path.join(args.outputfolder, base + identifier + '.pkl.gz')
        with gzip.open(filepath, 'w') as f:
            cPickle.dump(enc, f, -1)

        progress.update(i+1)
        
        if args.no_eval: # save some memory
            return None
        return enc

    progress.start()
    if args.parallel:
        all_enc = zip( *pc.parmap( encode, range(num_descr), args.nprocs ) )
    else:
        all_enc = zip( *map( encode, range(num_descr) ) )
    
    progress.finish()

    print 'got {} encodings'.format(len(all_enc))

    if args.no_eval:
        sys.exit(1)

    all_enc = np.concatenate(all_enc, axis=0).astype(np.float32)
    
    print 'Evaluation:'
    ret_matrix = evaluate.runNN( all_enc , labels, parallel=args.parallel,
                                nprocs=args.nprocs )    
    if ret_matrix is not None:
        fpath = os.path.join(args.outputfolder, 'dist' + identifier + '.cvs')
        np.savetxt(fpath, ret_matrix, delimiter=',')

