import numpy as np

# import umap.distances as dist
# from umap.utils import tau_rand_int
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, TransformerMixin
from . import _umap
from ._umap import tau_rand_int

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


def clip(val):
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val


def rdist(x, y):
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result


def _optimize_layout_euclidean_single_epoch(
    head_embedding,
    tail_embedding,
    head,
    tail,
    a,
    b,
    alpha,
    gamma,
    rng_state,
    move_other,
    epochs_per_edge,
    epochs_per_negative_sample,
    epoch_of_next_edge,
    epoch_of_next_negative_sample,
    current_epoch,
):
    n_vertices = head_embedding.shape[0]
    dim = head_embedding.shape[1]
    n_edges = head.shape[0]

    for i in range(n_edges):
        if epoch_of_next_edge[i] <= current_epoch:
            # 1. Sample this edge
            j = head[i]
            k = tail[i]
            current = head_embedding[j]
            other = tail_embedding[k]
            dist_squared = rdist(current, other)

            # move the vertices towards each other
            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0
            for d in range(dim):
                grad_d = clip(grad_coeff * (current[d] - other[d]))
                current[d] += grad_d * alpha
                if move_other:
                    other[d] += -grad_d * alpha

            epoch_of_next_edge[i] += epochs_per_edge[i]

            # 2. Sample random edges from same head node
            n_neg_samples = int(
                (current_epoch - epoch_of_next_negative_sample[i])
                / epochs_per_negative_sample[i]
            )
            for _ in range(n_neg_samples):
                # samples a random int32 value and modified the rng_state
                k = tau_rand_int(rng_state) % n_vertices
                other = tail_embedding[k]
                dist_squared = rdist(current, other)

                # move the head vertex away from the tail
                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1
                    )
                elif j == k:
                    continue
                else:
                    grad_coeff = 0.0
                for d in range(dim):
                    if grad_coeff > 0.0:
                        grad_d = clip(grad_coeff * (current[d] - other[d]))
                    else:
                        grad_d = 4.0
                    current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )


def _optimize_layout_vectorized(
    head_embedding,
    tail_embedding,
    head,
    tail,
    a,
    b,
    alpha,
    gamma,
    rng_state,
    move_other,
    epochs_per_edge,
    epochs_per_negative_sample,
    epoch_of_next_edge,
    epoch_of_next_negative_sample,
    current_epoch,
):

    dim = head_embedding.shape[1]
    n_vertices = head_embedding.shape[0]
    n_edges = head.shape[0]

    # 1. Sample each edge that is due for an update
    is_used = epoch_of_next_edge <= current_epoch
    h = head[is_used]
    t = tail[is_used]

    dist_squared = rdist(h, t)
    grad_coeff = (-2.0 * a * b * pow(dist_squared, b - 1.0)) / (
        a * pow(dist_squared, b) + 1.0
    )
    grad_d = np.clip(grad_coeff * (h - t), -4.0, 4.0)
    head_embedding[is_used] += grad_d * alpha
    if move_other:
        tail_embedding[is_used] += -grad_d * alpha

    epoch_of_next_edge[is_used] += epochs_per_edge[is_used]

    # 2. Sample random edges from same head node
    n_neg_samples = (
        (current_epoch - epoch_of_next_negative_sample[is_used])
        / epochs_per_negative_sample
    ).astype(np.int)
    idx = np.random.randint(0, n_vertices, n_neg_samples.sum())

    h = np.zeros(idx.shape[0])
    for i in range(n_neg_samples.shape[0]):
        h[c : c + n_neg_samples[i]] = head_embedding[i]

    t = tail_embedding[idx]

    dist_squared = rdist(h, t)
    grad_coeff = (2.0 * gamma * b) / (
        (0.001 + dist_squared) * (a * pow(dist_squared, b) + 1.0)
    )
    grad_coeff[h == t] = 0
    grad_d = np.clip(grad_coeff * (h - t), -4.0, 4.0)
    head_embedding += grad_d * alpha

    epoch_of_next_negative_sample[is_used] += (
        n_neg_samples * epochs_per_negative_sample[is_used]
    )


def _optimize_layout_general_single_epoch(
    head_embedding,
    tail_embedding,
    head,
    tail,
    a,
    b,
    alpha,
    gamma,
    rng_state,
    move_other,
    output_metric,
    output_metric_kwds,
    epochs_per_edge,
    epochs_per_negative_sample,
    epoch_of_next_edge,
    epoch_of_next_negative_sample,
    current_epoch,
):
    n_vertices = head_embedding.shape[0]
    dim = head_embedding.shape[1]
    n_edges = head.shape[0]

    for i in range(n_edges):
        if epoch_of_next_edge[i] <= current_epoch:
            # 1. Sample this edge
            j = head[i]
            k = tail[i]
            current = head_embedding[j]
            other = tail_embedding[k]
            dist_output, grad_dist_output = output_metric(
                current, other, *output_metric_kwds
            )
            _, rev_grad_dist_output = output_metric(other, current, *output_metric_kwds)
            if dist_output > 0.0:
                w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
            else:
                w_l = 1.0
            grad_coeff = 2 * b * (w_l - 1) / (dist_output + 1e-6)

            # move the vertices towards each other
            for d in range(dim):
                grad_d = clip(grad_coeff * grad_dist_output[d])

                current[d] += grad_d * alpha
                if move_other:
                    grad_d = clip(grad_coeff * rev_grad_dist_output[d])
                    other[d] += grad_d * alpha

            epoch_of_next_edge[i] += epochs_per_edge[i]

            # 2. Sample random edges from same head node
            n_neg_samples = int(
                (current_epoch - epoch_of_next_negative_sample[i])
                / epochs_per_negative_sample[i]
            )
            for _ in range(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices
                other = tail_embedding[k]
                dist_output, grad_dist_output = output_metric(
                    current, other, *output_metric_kwds
                )
                if dist_output > 0.0:
                    w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
                elif j == k:
                    continue
                else:
                    w_l = 1.0
                grad_coeff = gamma * 2 * b * w_l / (dist_output + 1e-6)

                # move the head vertex away from the tail
                for d in range(dim):
                    grad_d = clip(grad_coeff * grad_dist_output[d])
                    current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )


def _optimize_layout_inverse_single_epoch(
    head_embedding,
    tail_embedding,
    head,
    tail,
    weight,
    sigmas,
    rhos,
    a,
    b,
    alpha,
    gamma,
    rng_state,
    move_other,
    output_metric,
    output_metric_kwds,
    epochs_per_edge,
    epochs_per_negative_sample,
    epoch_of_next_edge,
    epoch_of_next_negative_sample,
    current_epoch,
):
    n_vertices = head_embedding.shape[0]
    dim = head_embedding.shape[1]  # high-dim space
    n_edges = head.shape[0]

    for i in range(n_edges):
        if epoch_of_next_edge[i] <= current_epoch:
            # 1. Sample this edge
            j = head[i]
            k = tail[i]
            current = head_embedding[j]
            other = tail_embedding[k]
            dist_output, grad_dist_output = output_metric(
                current, other, *output_metric_kwds
            )
            w_l = weight[i]
            grad_coeff = -(1 / (w_l * sigmas[k] + 1e-6))

            # move the vertices towards each other
            for d in range(dim):
                grad_d = clip(grad_coeff * grad_dist_output[d])

                current[d] += grad_d * alpha
                if move_other:
                    other[d] += -grad_d * alpha

            epoch_of_next_edge[i] += epochs_per_edge[i]

            # 2. Sample random edges from same head node
            n_neg_samples = int(
                (current_epoch - epoch_of_next_negative_sample[i])
                / epochs_per_negative_sample[i]
            )
            for _ in range(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices
                other = tail_embedding[k]
                dist_output, grad_dist_output = output_metric(
                    current, other, *output_metric_kwds
                )
                # w_l = 0.0 # for negative samples, the edge does not exist
                w_h = np.exp(-max(dist_output - rhos[k], 1e-6) / (sigmas[k] + 1e-6))
                grad_coeff = -gamma * ((0 - w_h) / ((1 - w_h) * sigmas[k] + 1e-6))

                # move the head vertex away from the tail
                for d in range(dim):
                    grad_d = clip(grad_coeff * grad_dist_output[d])
                    current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )


def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in the lower
    dimensional layout. We want the smooth curve (from a pre-defined family
    with simple gradient) that best matches an offset exponential decay.
    """
    from scipy.optimize import curve_fit

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xv, yv)
    return params[0], params[1]


def make_epochs_per_edge(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs betwen sampling for each edge.

    Parameters
    ----------
    weights: array of shape (n_edges)
        The weights of how often we wish to sample each edge.

    n_epochs: int
        The total number of epochs we want to train for.

    Returns
    -------
    An array of number of epochs per sample, one for each edge.
    (The higher, the less frequently the edge will get sampled).

    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result


class UmapLayout(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    head_embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.

    tail_embedding: array of shape (source_samples, n_components)
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the head_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.

    head: array of shape (nnz)
        Nonzero row indices of the affinity graph

    tail: array of shape (nnz)
        Nonzero column indices of the affinity graph

    epochs_per_samples: array of shape (nnz)
        A float value of the number of epochs per edge. Edges with
        weaker membership strength will have more epochs between being sampled.

    a: float
        Parameter of differentiable approximation of right adjoint functor

    b: float
        Parameter of differentiable approximation of right adjoint functor

    rng_state: array of int64, shape (3,)
        The internal state of the rng

    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.

    initial_alpha: float (optional, default 1.0)
        Initial learning rate for the SGD.

    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.

    """

    def __init__(
        self,
        min_dist,
        spread,
        gamma=1.0,
        negative_sample_rate=5.0,
        random_state=None,
        # output_metric=dist.euclidean,
        # output_metric_kwds=(),
    ):
        self.a, self.b = find_ab_params(spread, min_dist)
        self.gamma = gamma
        self.negative_sample_rate = negative_sample_rate
        random_state = check_random_state(random_state)
        self.rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
        self.optimize_fn = _umap.optimize_layout_euclidean_single_epoch

    def _fit(
        self,
        head_embedding,
        tail_embedding,
        move_tail,
        epochs,
        initial_epoch,
        initial_alpha,
    ):
        weights = self.affinities_.data
        epochs_per_edge = make_epochs_per_edge(weights, epochs - initial_epoch)
        epochs_per_negative_sample = epochs_per_edge / self.negative_sample_rate

        alpha = initial_alpha
        for i in range(initial_epoch):
            alpha *= 1.0 - i / float(epochs)

        for current_epoch in range(initial_epoch, epochs):
            self.optimize_fn(
                head_embedding,
                tail_embedding,
                move_tail,
                self.affinities_.row,
                self.affinities_.col,
                self.a,
                self.b,
                self.gamma,
                self.rng_state,
                alpha,
                current_epoch,
                epochs_per_edge,
                epochs_per_negative_sample,
                epoch_of_next_edge=epochs_per_edge.copy(),
                epoch_of_next_negative_sample=epochs_per_negative_sample.copy(),
            )
            alpha *= 1.0 - (current_epoch / float(epochs))
            print(alpha)

    def run(self, affinities, y0, epochs=1, initial_epoch=0, initial_alpha=1.0):
        affinities = affinities.tocoo()
        affinities.data[affinities.data < (affinities.data.max() / float(epochs))] = 0.0
        affinities.eliminate_zeros()
        self.affinities_ = affinities
        self.embedding_ = self.reference_ = y0
        self._fit(
            self.embedding_,
            self.reference_,
            True,
            epochs,
            initial_epoch,
            initial_alpha,
        )
        return self.embedding_


# class UmapSequentialLayout(BaseEstimator, TransformerMixin):
#     def __init__(
#         self,
#         graph,
#         pred_embedding,
#         head_embedding,
#         tail_embedding,
#         min_dist,
#         spread,
#         initial_alpha=1.0,
#         initial_alpha2=0.5,
#         gamma=1.0,
#         negative_sample_rate=5.0,
#         random_state=None,
#         n_epochs=200,
#         # output_metric=dist.euclidean,
#         # output_metric_kwds=(),
#     ):
#         graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
#         graph.eliminate_zeros()
#         self.graph = graph.tocoo()
#         self.n_nodes = graph.shape[1]
#         self.n_edges = graph.data.shape[0]

#         self.pred_embedding = pred_embedding
#         self.head_embedding = head_embedding
#         self.tail_embedding = tail_embedding

#         self.a, self.b = find_ab_params(spread, min_dist)
#         self.alpha = initial_alpha
#         self.alpha2 = initial_alpha2
#         self.gamma = gamma
#         self.negative_sample_rate = negative_sample_rate

#         random_state = check_random_state(random_state)
#         self.rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

#         self.n_epochs = n_epochs
#         self.current_epoch = 0
#         self.epochs_per_edge = make_epochs_per_edge(graph.data, n_epochs)
#         self.epochs_per_negative_sample = (
#             self.epochs_per_edge / self.negative_sample_rate
#         )
#         self.epoch_of_next_edge = self.epochs_per_edge.copy()
#         self.epoch_of_next_negative_sample = self.epochs_per_negative_sample.copy()
#         self.move_tail = self.head_embedding.shape[0] == self.tail_embedding.shape[0]

#     def run(self, n_steps=1):
#         optimize_fn = _umap.optimize_layout_sequential_single_epoch
#         max_step = min(self.current_epoch + n_steps, self.n_epochs)
#         for current_epoch in range(self.current_epoch, max_step):
#             self.current_epoch = current_epoch
#             optimize_fn(
#                 self.pred_embedding,
#                 self.head_embedding,
#                 self.tail_embedding,
#                 self.graph.row,
#                 self.graph.col,
#                 self.a,
#                 self.b,
#                 self.alpha,
#                 self.alpha2,
#                 self.gamma,
#                 self.rng_state,
#                 self.move_tail,
#                 self.epochs_per_edge,
#                 self.epochs_per_negative_sample,
#                 self.epoch_of_next_edge,
#                 self.epoch_of_next_negative_sample,
#                 self.current_epoch,
#             )
#             self.alpha *= 1.0 - (float(current_epoch) / float(self.n_epochs))
#             self.alpha2 *= 1.0 - (float(current_epoch) / float(self.n_epochs))

#         return self.head_embedding
