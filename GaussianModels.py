import numpy as np

"""
This script contains Univariate, Multivariate and Mixture Gaussian Models built using the numpy library. For a 
custom model to work within the algorithms implementation is required a sample() and __call__() functions that take no
arguments and a dataset of d-dimensional datasets.
"""

def normpdf(xset, mean, covar):
    """
    Computes the Gaussian Multivariate Probability Density Functions for a dataset of inputs.
    :param xset: Array of d-dimensional arrays
    :param mean: d-dimensional array
    :param covar: dxd dimensional matrix
    :return: Gaussian Multivariate PDF
    """
    k = len(xset[0])
    norms = []
    for x in xset:
        part1 = 1 / (((2 * np.pi)**(k/2)) * np.linalg.det(covar)**0.5)
        part2 = np.exp(-0.5 * np.dot(np.dot((x - mean).T,np.linalg.inv(covar)), x-mean))
        norms.append(part1 * part2)

    return np.asarray(norms)


class GaussianMixtureModel:
    """
    Gaussian Mixture Model whose means, covariances and model weights are computed using the EM optimization method.
    """
    def __init__(self, n_components=1, em_iterations=5, tol = 0.1):
        self.n_components = n_components
        self.mus =  None
        self.vars = None
        self.kprobs = None
        self.emits = em_iterations
        self.tol = tol

    def sample(self):
        c = np.random.choice(range(self.n_components))
        return np.random.multivariate_normal(self.means[c], self.covs[c])


    def __call__(self, X):
        # data's dimensionality and responsibility vector
        n_row, n_col = X.shape
        self.resp = np.zeros((n_row, self.n_components))

        # initialize parameters
        chosen = np.random.choice(n_row, self.n_components, replace=False)
        self.means = X[chosen]
        self.weights = np.full(self.n_components, 1 / self.n_components)

        # for np.cov, rowvar = False,
        # indicates that the rows represents obervation
        shape = self.n_components, n_col, n_col
        self.covs = np.full(shape, np.cov(X, rowvar=False))

        log_likelihood = 0
        self.converged = False
        self.log_likelihood_trace = []

        for i in range(self.emits):
            log_likelihood_new = self._do_estep(X)
            self._do_mstep(X)

            if abs(log_likelihood_new - log_likelihood) <= self.tol:
                self.converged = True
                break

            log_likelihood = log_likelihood_new
            self.log_likelihood_trace.append(log_likelihood)

        return self

    def _do_estep(self, X):
        """
        E-step: compute responsibilities,
        update resp matrix so that resp[j, k] is the responsibility of cluster k for data point j,
        to compute likelihood of seeing data point j given cluster k, use multivariate_normal.pdf
        """
        self._compute_log_likelihood(X)
        log_likelihood = np.sum(np.log(np.sum(self.resp, axis=1)))

        # normalize over all possible cluster assignments
        self.resp = self.resp / self.resp.sum(axis=1, keepdims=1)
        return log_likelihood

    def _compute_log_likelihood(self, X):
        for k in range(self.n_components):
            prior = self.weights[k]
            likelihood = normpdf(X, self.means[k], self.covs[k]+np.eye(len(self.covs[k]))*1e-1)
            self.resp[:, k] = prior * likelihood

        return self

    def _do_mstep(self, X):
        """M-step, update parameters"""

        # total responsibility assigned to each cluster, N^{soft}
        resp_weights = self.resp.sum(axis=0)

        # weights
        self.weights = resp_weights / X.shape[0]

        # means
        weighted_sum = np.dot(self.resp.T, X)
        self.means = weighted_sum / resp_weights.reshape(-1, 1)
        # covariance
        for k in range(self.n_components):
            diff = (X - self.means[k]).T
            weighted_sum = np.dot(self.resp[:, k] * diff, diff.T)
            self.covs[k] = weighted_sum / resp_weights[k]
        return self

class UnivariateNormalModel:
    """
    Class is a univariate normal model, whose mean and variance parameters are computed
    using the method of maximum estimation.
    """
    def __init__(self):
        self.mus = None
        self.sigmas = None
        self.dimension = None


    def __call__(self, dataset):
        self.mus: float = [0] * dataset.shape[1]
        self.sigmas: float = [0] * dataset.shape[1]
        self.dimension: float = dataset.shape[1]

        for d in range(dataset.shape[1]):
            dimension_sum = sum(xi[d] for xi in dataset)
            self.mus[d] = dimension_sum / dataset.shape[0]
            self.sigmas[d] =  sum((xi[d] - self.mus[d]) ** 2 for xi in dataset) / dataset.shape[0]


    def sample(self):
        s = np.zeros(shape=(self.dimension,))
        for i in range(self.dimension):
            s[i] = np.random.normal(self.mus[i], self.sigmas[i])

        return s

class MultivariateNormalModel:
    """
    Class is a univariate normal model, whose mean and variance parameters are computed
    using the method of maximum estimation.
    """
    def __init__(self):
        self.mu = None
        self.sigma = None
        self.dimension = None


    def __call__(self, dataset):
        self.dimension = dataset.shape[1]
        self.mu = sum(d for d in dataset) / len(dataset)
        self.sigma = np.zeros(shape=(dataset.shape[1],dataset.shape[1]))

        for i in dataset:
            self.sigma += np.dot(i-self.mu, (i-self.mu).T)
        self.sigma = self.sigma/len(dataset)

    def sample(self):
        return np.random.multivariate_normal(self.mu,self.sigma)