import numpy as np
from sklearn import preprocessing
from sklearn import mixture
import math
import copy

class Encoding(object):
    """ 
    wrapper class around encoding methods
    """
    def __init__(self, method, ubm, 
                 normalize=['l2g','l2c'], 
                 update='wmc', 
                 relevance=28):
        self.method = method
        self.ubm = ubm
        self.normalize = normalize
        self.update = update
        self.relevance = relevance

    def encode(self, features, posteriors=None):
        enc = encodeGMM(self.method, self.ubm, features, 
                  normalize=self.normalize, relevance=self.relevance,
                  update=self.update, posteriors=posteriors)
        return enc

def normalizeEnc(enc, method):
    """
    normalize encoding w. global normalization scheme(s)
    
    parameters:
        enc: the encoding vector to normalize
        method:
            'ssr': signed square root
            'l2g': global l2 normalization
    """
    # ssr-normalization (kinda hellinger-normalization)
    if 'ssr' in method:
        enc = np.sign(enc) * np.sqrt(np.abs(enc))

    if 'l2g' in method:
        enc = preprocessing.normalize(enc)

    return enc


def getPosteriors(gmm, data):
    """
    compute the posterior probability (assignment) for each sample

    parameters:
        gmm: scikit-learn computed gmm
        data: feature-vectors row-wise
    """

    posteriors = gmm.predict_proba(data)

    return posteriors

def encodeGMM(method, gmm, data,  
              normalize=['ssr', 'l2g'], relevance=28, update='wmc',
              posteriors=None):
    """ 
        Encoding scheme adapting data from a GMM

        parameters:
            method: an encoding method to call, currently acceptable
                'supervector', 'fisher', 'vlad', 
            gmm: the background gmm = vocabulary = universal background model
            data: typiclly the features, row wise order
            normalize: global and local normalization schemes
            relevance: (gmm-supervector-specific): relevance factor for mixing 
            update: (gmm-supervector-specific): which parts to use for mixing 
            posteriors: if given, they won't be computed again
        returns: encoded vector
    """
    if posteriors == None:
        # compute posteriors from gmm
        posteriors = getPosteriors(gmm, data)

    if 'supervector' in method:
        enc = supervector(gmm, data, posteriors, relevance=relevance,
                          update=update)
    elif 'fisher' in method:
        enc = fisherEncode(gmm, data, 
                           posteriors, normalize, 
                           fv_components=update)
    elif 'vlad' in method:
        enc = vlad(data, gmm.means_, posteriors, gmm.means_.shape[0],
                   normalize ) 
    else:
        raise ValueError('unknown encoding method {}'.format(method)) 
    
    norm_enc = normalizeEnc(enc, normalize)

    return norm_enc


def adaptMAP(data, gmm, posteriors, relevance=16, update='wmc'):
    """
    Adapt new data to a given gmm with a specific
    relevance factor.
    Reference: D.A.Reynolds et al. 'Speaker Verification Using Adapted Gaussian
    Mixture Models'
    """

    sum_post = np.sum(posteriors, axis=0) # (N_component x ,)
    
    nd = len(gmm.weights_) # number of components / gaussians
    fd = data.shape[1] # feature dimension

    data_square = data * data

    def loop(i):
        means_ = posteriors[:,i].reshape(1,-1).dot(data)
        covs_ = posteriors[:,i].reshape(1,-1).dot(data_square)
        return means_, covs_

    means, covs = zip( *map(loop, range(nd)))

    means = np.array(means).reshape(nd, fd)
    covs = np.array(covs).reshape(nd,fd)

    # add some small number
    means += np.finfo(float).eps
    covs += np.finfo(float).eps

    # normalize them
    means /= sum_post.reshape(-1,1) + np.finfo(float).eps
    covs /= sum_post.reshape(-1,1) + np.finfo(float).eps

    # now combine the two estimates using the relevance factor
    # i.e. interpolation controlled by relevance factor
    def combine(i):
        alpha = sum_post[i] / (sum_post[i] + relevance)

        # update priors
        if 'w' in update:
            weights_ = ( (alpha * sum_post[i]) / float(len(data)) ) \
                    + ( (1.0 - alpha) * gmm.weights_[i] )
        else:
            weights_ = copy.deepcopy(gmm.weights_[i])
    
        # update means
        if 'm' in update:
            means_ = alpha * means[i] \
                    + ( (1.0 -alpha) * gmm.means_[i] )
        else:
            means_ = copy.deepcopy(gmm.means_[i])
        # update covariance matrix
        if 'c' in update:
            covs_ = alpha * covs[i] \
                    + (1.0 - alpha) * (gmm.covars_[i] + \
                                       gmm.means_[i] * gmm.means_[i])\
                    - (means_ * means_) # careful, this is means_ not means_[i], 
                                        # since we are in that specific
                                        # component computation already!
        else:
            covs_ = copy.deepcopy(gmm.covars_[i])

        return weights_, means_, covs_

    weights, means, covs = zip( *map(combine, range(nd)) )

    weights = np.array(weights)
    means = np.array(means)
    covs = np.array(covs)

    # let weights sum to 1
    if 'w' in update:
        weights /= weights.sum() + np.finfo(float).eps

    # create new mixture  
    adapted_gmm = mixture.GMM(nd)
    # and assign mean, cov, priors to it
    adapted_gmm.weights_ = weights
    adapted_gmm.means_ = means
    adapted_gmm.covars_ = covs
#    adapted_gmm._set_covars( covs ) # this variant checks the covariances 

    return adapted_gmm

def supervector(gmm, data, posteriors, 
                normalize=[], update='wmc', 
                relevance=16):

    scribe_gmm = adaptMAP(data, gmm, posteriors, relevance,                    
                          update)

    return supervectorStacking(scribe_gmm, update, normalize)

def supervectorStacking(gmm, update='wmc', normalize=[]):
    """
    form supervector, optionally normalize each component
    """
    enc = []
    if 'l2c' in normalize:
        for i in range(len(gmm.means)):
            c_enc = []
            if 'm' in update:
                enc_m = gmm.means_[i]
                c_enc.append(enc_m)

            if 'c' in update:
                enc_c = gmm.covars_[i]
                c_enc.append(enc_c)

            if 'w' in update:
                enc_w = gmm.weights_[i]
                c_enc.append(enc_w)
            c_enc = np.concatenate(enc, axis=1)
            c_enc = preprocessing.normalize(enc)
        enc.append(c_enc)
    else:
        if 'm' in update:
            enc_m = gmm.means_
            enc.append(enc_m.reshape(1,-1))

        if 'c' in update:
            enc_c = gmm.covars_
            enc.append(enc_c.reshape(1,-1))

        if 'w' in update:
            enc_w = gmm.weights_
            enc.append(enc_w.reshape(1,-1))

    enc = np.concatenate(enc, axis=1)

    return enc

def fisher(data, means, weights, posteriors, 
           inv_sqrt_cov): 
    
    components, fd = means.shape

    def encode(i):
        if weights[i] < 1e-6:
            return np.zeros( (fd), dtype=means.dtype),\
                   np.zeros( (fd), dtype=means.dtype) 

        #diff = data * inv_sqrt_cov[i]
        diff = (data - means[i]) * inv_sqrt_cov[i]
        weights_ = np.sum(posteriors[:,i] - weights[i])
        means_ = posteriors[:,i].T.dot( diff )
        covs_ = posteriors[:,i].T.dot( diff*diff - 1 )

        weights_ /= ( len(data) * math.sqrt(weights[i]) )
        means_ /= ( len(data) * math.sqrt(weights[i]) )
        covs_ /= ( len(data) * math.sqrt(2.0*weights[i]) )
   
        return weights_, means_, covs_

    wk_, uk_, vk_ = zip( map(encode, range(components)) )
    
    return wk_, uk_, vk_

def fisherEncode( gmm, data, posteriors, 
                  normalize=[],
                  fv_components='wmc'):

    inv_sqrt_cov = np.sqrt(1.0 / (gmm.covars_ + np.finfo(np.float32).eps))
    wk_, uk_, vk_ = fisher( data, gmm.means_, gmm.weights_,
                              posteriors, inv_sqrt_cov)

    wk_ = np.array(wk_)
    uk_ = np.array(uk_)
    vk_ = np.array(vk_)

    components, fd = gmm.means_.shape
    # component-wise normalization
    if 'l2c' in normalize: 
        wk_ = preprocessing.normalize(wk_)
        uk_ = preprocessing.normalize(uk_.reshape(components, fd))
        vk_ = preprocessing.normalize(vk_.reshape(components, fd))
    
    # stacking
    enc = []
    if 'w' in fv_components: 
        enc.append(wk_.reshape(1,-1))
    if 'm' in fv_components:
        enc.append(uk_.reshape(1,-1))
    if 'c' in fv_components:
        enc.append(vk_.reshape(1,-1))

    fv = np.concatenate(enc, axis=1) 

    return fv


def vlad(data, means, assignments, components, 
               normalize=['l2c']):
    """
    compute 'vector of locally aggregated descriptors'
    """
    def encode(k):
        uk_ = assignments[:,k].T.dot(data)        

        clustermass = assignments[:,k].sum()
        if clustermass > 0:
            uk_ -= clustermass * means[k]

        if 'l2c' in normalize:
            n = max(math.sqrt(np.sum(uk_ * uk_)), 1e-12)
            uk_ /= n

        return uk_

    uk = map(encode, range(components))

    uk = np.concatenate(uk, axis=0).reshape(1,-1)

    return uk

