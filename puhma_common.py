import gzip
import cPickle
import os
import glob
import multiprocessing 
import random
import numpy as np
import argparse
import sys
import progressbar

verbose = False
if verbose:
    def verboseprint(*args):
        # Print each argument separately so caller doesn't need to
        # stuff everything to be printed into a single string
        for arg in args:
           print arg,
        print
else:   
    verboseprint = lambda *a: None      # do-nothing function

def enable_nice_traceback():
    # also look for "ultraTB" in "IPython.CrashHandler"
    import IPython.core.ultratb as utb
    sys.excepthook = utb.VerboseTB(include_vars=0) #VerboseTB, ColorTB, AutoFormattedTB, FormattedTB

def hellingerNormalization(features):
    # L1 normalization
    features += np.finfo(np.float32).eps
    features /= np.sum(features, axis=1)[:,np.newaxis]
    # square root
    features = np.sqrt(features)
    return features

def commonArguments(parser):
    io_group = parser.add_argument_group('input output options')
    io_group.add_argument('-l', '--labelfile',\
                        help='label-file containing the images to load + labels')
    io_group.add_argument('--min_descs',  default=0, type=int,\
                          help='minimum features per file')
    io_group.add_argument('-i', '--inputfolder',\
                        help='the input folder of the images / features')
    io_group.add_argument('--suffix', default='',\
                        help='only chose those images with a specific suffix')
    io_group.add_argument('-o', '--outputfolder', default='.',\
                        help='the output folder for the descriptors')
    io_group.add_argument('--overwrite', type=bool, default=True,
                          help='overwrite output-file (default=True)')

    preprocess = parser.add_argument_group('preprocess features options')
    preprocess.add_argument('--hellinger', action='store_true',\
                        help='normalize feature vector (sqrt(F./sum(F)))')

    general = parser.add_argument_group('general options')
    general.add_argument('--parallel', action='store_true',\
                        help='some parts are parallelized')
    general.add_argument('--nprocs', type=int,
                         default=multiprocessing.cpu_count(),
                         help='number of parallel instances')
    return parser


def getFiles(folder, pattern, labelfile=None, exact=True, concat=False):

    if not os.path.exists(folder):
        print 'WARNING: file or folder {} doesnt exist'.format(folder)

    if not os.path.isdir(folder):
        return  [ folder ], None

    if labelfile:
        labels = []
        with open(labelfile, 'r') as f:
            all_lines = f.readlines()
        all_files = []
        for  line in all_lines:
            img_name, class_id = line.split()
            if exact:
                file_name = os.path.join(folder, os.path.splitext(img_name)[0] + pattern )
                all_files.append(file_name)
            else:
                search_pattern = os.path.join(folder, 
                                              os.path.splitext(img_name)[0] 
                                              + '*' + pattern)
                files = glob.glob(search_pattern)
                if concat:
                    all_files.append(files)
                else:
                    all_files.extend(files)
            labels.append(class_id)               

        return all_files, labels

    return glob.glob(os.path.join(folder, '*' + pattern)), None


def loadDescriptors(files, reshape=False, max_descs=0, max_descs_per_file=0,
                    rand=True, ax=0, hellinger=False,
                    min_descs_per_file=0, 
                    show_progress=True): 
    if len(files) == 0:
        print 'WARNING: laodDescriptor() called with no files'
        return
    if isinstance(files, basestring):
        files = [files]

    descriptors = []
    desc_length = 0
    if len(files) > 100 and show_progress:
        widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
               progressbar.ETA()]
        progress = progressbar.ProgressBar(widgets=widgets)
    else:
        def progress(x):
            return x

    for i in progress(range(len(files))):
        f = files[i]
        try:
            if f.endswith('pkl.gz'):
                with gzip.open(f, 'rb') as ff:
                    desc = cPickle.load(ff)               
                    if desc.dtype != np.float32 and desc.dtype != np.float64:
                        print ('WARNING: desc.dtype ({}) != np.float32 ->'
                        ' convert it'.format(desc.dtype))
                        desc = desc.astype(np.float32)
            else:
                desc = np.loadtxt(f, delimiter=',',ndmin=2, dtype=np.float32)
        except ValueError:
            print 'Error at file', f
            raise


        if len(desc) == 0:
            print 'no descriptors of file {}?'.format(f)
            continue
        
        # skip if too few descriptors per file
        if min_descs_per_file > 0 and len(desc) <= min_descs_per_file:
            continue
        # reshape the descriptor
        if reshape:
            desc = desc.reshape(1,-1)
        # pick max_descs_per_file, either random or the first ones
        if max_descs_per_file != 0:
            if rand:
                desc = desc[ np.random.choice(len(desc), 
                                              min(len(desc),
                                              max_descs_per_file))]
                # this is probably slower
#                desc = np.array( random.sample(desc, min( len(desc), max_descs_per_file) ) )                
            else:
                desc = desc[:max_descs_per_file]

        descriptors.append(desc)

        desc_length += len(desc)
        if max_descs != 0 and desc_length > max_descs:
            break
    
    if len(descriptors) == 0:
        if min_descs_per_file == 0:
            print 'couldnt load ', ' '.join(files)
        return None 

    descriptors = np.concatenate(descriptors, axis=ax)

    if hellinger:
        descriptors = hellingerNormalization(descriptors)   
    
    return descriptors

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        import errno
        if exc.errno == errno.EEXIST:
            pass
        else: 
            raise

def spawn(f):
    def fun(q_in,q_out):
        while True:
            i,x = q_in.get()
            if i == None:
                break
            q_out.put((i,f(x)))
    return fun

def parmap(f, X, nprocs = multiprocessing.cpu_count()):
    q_in   = multiprocessing.Queue(1)
    q_out  = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=spawn(f),args=(q_in,q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i,x)) for i,x in enumerate(X)]
    [q_in.put((None,None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i,x in sorted(res)]


