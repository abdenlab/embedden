import locale
from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

locale.setlocale(locale.LC_NUMERIC, "C")

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf


def _smooth_knn_dist(
    distances,
    k,
    n_iter=64,
    local_connectivity=1.0,
    bandwidth=1.0
):
    target = np.log2(k) * bandwidth

    # distance to each point's "kth nearest neighbor" where k may not be integral
    # i.e. radius of a ball at which the fuzzy neighbor membership reaches k
    sigmas = np.zeros(distances.shape[0], dtype=np.float32)

    # distance to each point's 1st nearest neighbor
    rhos = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rhos[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rhos[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rhos[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rhos[i] = np.max(non_zero_dists)

        for _ in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rhos[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        sigmas[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rhos[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if sigmas[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                sigmas[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if sigmas[i] < MIN_K_DIST_SCALE * mean_distances:
                sigmas[i] = MIN_K_DIST_SCALE * mean_distances

    return sigmas, rhos


def _compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos):
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals


def make_membership_graph(
    knn_indices, knn_dists, local_connectivity, set_op_mix_ratio
):
    # 1. Use the knn graph to get a kernel width (sigma) for each point
    # and the distance to the closest neighbor (rho).
    # The kernel width is the ball radius at which the fuzzy membership reaches
    # n_neighbors.
    n = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]
    knn_dists = knn_dists.astype(np.float32)
    sigmas, rhos = _smooth_knn_dist(
        knn_dists,
        float(n_neighbors),
        local_connectivity=float(local_connectivity),
    )

    # 2. Construct a "fuzzy simplicial set", (symmetrized membership graph)
    rows, cols, vals = _compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos
    )
    graph = coo_matrix((vals, (rows, cols)), shape=(n, n))
    graph.eliminate_zeros()
    transpose = graph.transpose()
    prod_matrix = graph.multiply(transpose)
    graph = (
        set_op_mix_ratio * (graph + transpose - prod_matrix)
        + (1.0 - set_op_mix_ratio) * prod_matrix
    )
    graph.sum_duplicates()
    graph.eliminate_zeros()

    return graph


class UmapAffinityKernel(BaseEstimator, TransformerMixin):

    def __init__(self, local_connectivity, set_op_mix_ratio):
        self.local_connectivity = local_connectivity
        self.set_op_mix_ratio = set_op_mix_ratio

    def fit(self, knn_graph):
        knn_indices, knn_dists = knn_graph
        self.affinities_ = make_membership_graph(
            knn_indices,
            knn_dists,
            self.local_connectivity,
            self.set_op_mix_ratio
        )
        return self

    def transform(self, knn_graph):
        return self.affinities_
