import warnings
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.manifold._utils import _binary_search_perplexity
from sklearn.base import BaseEstimator, TransformerMixin

MACHINE_EPSILON = np.finfo(np.double).eps


def _joint_probabilities_nn(distances, desired_perplexity):
    """Compute joint probabilities p_ij from distances using just nearest
    neighbors.

    This method is approximately equal to _joint_probabilities. The latter
    is O(N), but limiting the joint probability to nearest neighbors improves
    this substantially to O(uN).

    Parameters
    ----------
    distances : sparse matrix of shape (n_samples, n_samples)
        Distances of samples to its n_neighbors nearest neighbors. All other
        distances are left to zero (and are not materialized in memory).
        Matrix should be of CSR format.
    desired_perplexity : float
        Desired perplexity of the joint probability distributions.
    verbose : int
        Verbosity level.

    Returns
    -------
    P : sparse matrix of shape (n_samples, n_samples)
        Condensed joint probability matrix with only nearest neighbors. Matrix
        will be of CSR format.
    """

    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances.sort_indices()
    n_samples = distances.shape[0]
    distances_data = distances.data.reshape(n_samples, -1)
    distances_data = distances_data.astype(np.float32, copy=False)

    conditional_P = _binary_search_perplexity(
        distances_data, desired_perplexity, verbose=False
    )

    assert np.all(np.isfinite(conditional_P)), "All probabilities should be finite"

    # Symmetrize the joint probability distribution using sparse operations
    P = csr_matrix(
        (conditional_P.ravel(), distances.indices, distances.indptr),
        shape=(n_samples, n_samples)
    )
    P = P + P.T

    # Normalize the joint probability distribution
    sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
    P /= sum_P

    assert np.all(np.abs(P.data) <= 1.0)

    return P


class TsneAffinityKernel(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        perplexity=30.0,
        square_distances=True,
    ):
        self.perplexity = perplexity
        self.square_distances = square_distances

    def fit(self, knn_graph):

        knn_indices, knn_dists = knn_graph

        n = knn_indices.shape[0]
        n_neighbors = knn_indices.shape[1]

        # LvdM uses 3 * perplexity as the number of neighbors.
        if n_neighbors < 3 * self.perplexity:
            warnings.warn(
                "It is recommended to use at least 3 * perplexity # of neighbors "
                "to calculate affinities with a given perplexity."
            )

        # construct CSR matrix representation of the k-NN graph
        nnz = n * n_neighbors
        indptr = np.arange(0, nnz + 1, n_neighbors)
        knn_graph = csr_matrix(
            (knn_dists.ravel(), knn_indices.ravel(), indptr),
            shape=(n, n)
        )

        if self.square_distances:
            # knn return the euclidean distance but we need it squared
            # to be consistent with the 'exact' method. Note that the
            # the method was derived using the euclidean method as in the
            # input space. Not sure of the implication of using a different
            # metric.
            knn_graph.data **= 2

        # compute the joint probability distribution for the input space
        self.P_ = _joint_probabilities_nn(
            knn_graph, self.perplexity, self.verbose
        )

        return self

    def transform(self, knn_graph):
        return self._P
