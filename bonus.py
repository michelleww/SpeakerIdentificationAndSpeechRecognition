from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dataDir = "/u/cs401/A3/data/"


class theta:
    def __init__(self, name, M=8, d=13):
        """Class holding model parameters.
        Use the `reset_parameter` functions below to
        initialize and update model parameters during training.
        """
        self.name = name
        self._M = M
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))

    def precomputedForM(self, m):
        """Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """
        mu_squared = np.square(self.mu[m])
        sigma_squared = self.Sigma[m]
        term1 = np.sum(np.divide(mu_squared, 2 * sigma_squared, where=(sigma_squared != 0)))
        
        term2 = (self._d / 2) * np.log(2 * np.pi)

        prod = np.prod(sigma_squared)
        term3 = np.log(prod, where=(prod != 0)) / 2

        return term1 + term2 + term3

    def reset_omega(self, omega):
        """Pass in `omega` of shape [M, 1] or [M]
        """
        omega = np.asarray(omega)
        assert omega.size == self._M, "`omega` must contain M elements"
        self.omega = omega.reshape(self._M, 1)

    def reset_mu(self, mu):
        """Pass in `mu` of shape [M, d]
        """
        mu = np.asarray(mu)
        shape = mu.shape
        assert shape == (self._M, self._d), "`mu` must be of size (M,d)"
        self.mu = mu

    def reset_Sigma(self, Sigma):
        """Pass in `sigma` of shape [M, d]
        """
        Sigma = np.asarray(Sigma)
        shape = Sigma.shape
        assert shape == (self._M, self._d), "`Sigma` must be of size (M,d)"
        self.Sigma = Sigma


def log_b_m_x(m, x, myTheta):
    """ Returns the log probability of d-dimensional vector x using only
        component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `myTheta.preComputedForM(m)` for this.

    Return shape:
        (single row) if x.shape == [d], then return value is float (or equivalent)
        (vectorized) if x.shape == [T, d], then return shape is [T]

    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.
    """
    sigma_squared = myTheta.Sigma[m]
    sub_1 = np.divide(np.square(x), sigma_squared, where=(sigma_squared != 0)) / 2
    sub_2 = np.divide(myTheta.mu[m] * x, sigma_squared, where=(sigma_squared != 0))
    sub = sub_1 - sub_2

    if len(x.shape) == 1:
        term1 = np.sum(sub)
    else:
        term1 = np.sum(sub, axis = 1)
    
    return -term1 - myTheta.precomputedForM(m)

# helper function from the tut 
def stable_logsumexp(array_like, axis=-1):
    """Compute the stable logsumexp of `vector_like` along `axis`
    This `axis` is used primarily for vectorized calculations.
    """
    array = np.asarray(array_like)
    # keepdims should be True to allow for broadcasting
    m = np.max(array, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(array - m), axis=axis))

def log_p_m_x(log_Bs, myTheta):
    """ Returns the matrix of log probabilities i.e. log of p(m|X;theta)

    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout

    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below
    """
    log_Ws = np.log(myTheta.omega, where=(myTheta.omega != 0))
    term3 = stable_logsumexp((log_Ws+log_Bs), axis = 0)
    return log_Ws + log_Bs - term3


def logLik(log_Bs, myTheta):
    """ Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    """
    log_W = np.log(myTheta.omega, where=(myTheta.omega != 0))
    return np.sum(stable_logsumexp((log_W + log_Bs), axis = 0))

def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker. Returns the theta (omega, mu, sigma)"""
    T = X.shape[0]
    d = X.shape[1]
    myTheta = theta(speaker, M, d)
    # initialize to a random verctor from the data X, see tut slides P28
    samples = random.sample(range(T), M)
    myTheta.reset_mu(X[samples])
    # initialize sigma to a random diagonal matrix
    myTheta.reset_Sigma(np.ones((M,d)))
    # initialize omega to be 1/m
    myTheta.reset_omega(np.ones((M, 1)) / M)

    i = 0
    previous_L = - np.inf
    improvement = np.inf
    while i <= maxIter and improvement >= epsilon:
        log_Bs = np.zeros((M, T))
        for m in range(M):
            log_Bs[m] = log_b_m_x(m, X, myTheta)
        log_Ps = log_p_m_x(log_Bs, myTheta)

        # compute likelihood
        L = logLik(log_Bs, myTheta)

        # calculate and update parameters for theta
        Ps = np.exp(log_Ps)
        Ps_sum_over_T = np.sum(Ps, axis=1)
        Ps_sum_over_T = Ps_sum_over_T.reshape(Ps_sum_over_T.shape[0], 1)

        omegas = Ps_sum_over_T / T
        myTheta.reset_omega(omegas)

        mu_numerator = np.dot(Ps, X)
        mu = np.divide(mu_numerator, Ps_sum_over_T, where=(Ps_sum_over_T != 0))
        myTheta.reset_mu(mu)

        Sigma_numerator = np.dot(Ps, np.square(X))
        Sigma = np.divide(Sigma_numerator, Ps_sum_over_T, where=(Ps_sum_over_T != 0)) - np.square(mu)
        myTheta.reset_Sigma(Sigma)

        # updating improvement, i and previous_L for next iterations
        improvement = L - previous_L
        previous_L = L
        i += 1
    return myTheta


def test(mfcc, correctID, models, k=5):
    """ Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    """
    if k <= 0:
        return 0
    bestModel = -1
    likelihoods = []
    T = mfcc.shape[0]
    for model in models:
        log_Bs = np.zeros((M, T))
        for m in range(M):
            log_Bs[m] = log_b_m_x(m, mfcc, model)
        # compute likelihood
        likelihoods.append(logLik(log_Bs, model))
    sorted_likli = np.flipud(np.argsort(likelihoods))
    top_k = sorted_likli[:k]
    bestModel = top_k[0]

    return 1 if (bestModel == correctID) else 0

def pca(new_dimension, speakers, trainMFCCs, testMFCCs):

    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20

    # standardarize the data
    scaler = StandardScaler()

    # Fit on training set only.
    X = np.vstack(trainMFCCs)
    scaler.fit(X)
    # # Apply transform to both the training set and the test set.
    X = scaler.transform(X)

    pca = PCA(new_dimension)
    pca.fit(X)
    X = pca.transform(X)

    trainThetas = []

    start = 0
    end = 0
    for i in range(len(trainMFCCs)):
        start = end
        length = trainMFCCs[i].shape[0]
        end = start + length
        trainThetas.append(train(speakers[i], X[start:end], M, epsilon, maxIter))

    numCorrect = 0
    for i in range(0, len(testMFCCs)):
        numCorrect += test(pca.transform(scaler.transform(testMFCCs[i])), i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
    return accuracy

if __name__ == "__main__":
    testMFCCs = []
    trainMFCCs = []
    speakers = []
    d = 13
    M = 8
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)
            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))

            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)
            trainMFCCs.append(X)
        
            speakers.append(speaker)
    with open('bonus_pca.txt', 'a+') as output_f:
        for i in range(13, 0, -2):
            accuracy = pca(i, speakers, trainMFCCs, testMFCCs)
            output_f.write( "Dimension is {:2}, accuracy is {:.6f}\n".format(i, accuracy))
