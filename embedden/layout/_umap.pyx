#!python
#cython: embedsignature=True
import numpy as np
cimport numpy as np

cdef np.int32_t INT32_MIN = np.iinfo(np.int32).min + 1
cdef np.int32_t INT32_MAX = np.iinfo(np.int32).max - 1

ctypedef fused int_t:
    np.int32_t
    np.int64_t


cdef inline double clip(double val):
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val


cdef inline double rdist(np.ndarray[np.double_t, ndim=1] x, np.ndarray[np.double_t, ndim=1] y):
    cdef float result = 0.0
    cdef int dim = x.shape[0]
    cdef double diff
    cdef int i
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff
    return result


cpdef np.int32_t tau_rand_int(np.ndarray[np.int64_t, ndim=1] state):
    """A fast (pseudo)-random number generator.

    Parameters
    ----------
    state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    A (pseudo)-random int32 value

    """
    state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF) ^ (
        (((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19
    )
    state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF) ^ (
        (((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25
    )
    state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF) ^ (
        (((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11
    )

    return state[0] ^ state[1] ^ state[2]


def optimize_layout_euclidean_single_epoch(
    np.ndarray[np.double_t, ndim=2] head_embedding,
    np.ndarray[np.double_t, ndim=2] tail_embedding,
    bint move_other,
    np.ndarray[int_t, ndim=1] head,
    np.ndarray[int_t, ndim=1] tail,
    double a,
    double b,
    double gamma,
    np.ndarray[np.int64_t, ndim=1] rng_state,
    double alpha,
    int current_epoch,
    np.ndarray[np.double_t, ndim=1] epochs_per_edge,
    np.ndarray[np.double_t, ndim=1] epochs_per_negative_sample,
    np.ndarray[np.double_t, ndim=1] epoch_of_next_edge,
    np.ndarray[np.double_t, ndim=1] epoch_of_next_negative_sample,
):
    cdef int dim = head_embedding.shape[1]
    cdef int n_edges = head.shape[0]
    cdef int n_vertices = head_embedding.shape[0]

    cdef np.ndarray[np.double_t, ndim=1] current, other
    cdef double dist_squared, grad_coeff, grad_d
    cdef int i, j, d, _, n_neg_samples
    cdef np.int32_t k
    for i in range(n_edges):
        if epoch_of_next_edge[i] <= current_epoch:
            # 1. Sample this edge
            j = head[i]
            k = tail[i]
            current = head_embedding[j, :]
            other = tail_embedding[k, :]
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
                (current_epoch - epoch_of_next_negative_sample[i]) /
                epochs_per_negative_sample[i]
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


def optimize_layout_sequential_single_epoch(
    np.ndarray[np.double_t, ndim=2] pred_embedding,
    np.ndarray[np.double_t, ndim=2] head_embedding,
    np.ndarray[np.double_t, ndim=2] tail_embedding,
    np.ndarray[int_t, ndim=1] head,
    np.ndarray[int_t, ndim=1] tail,
    double a,
    double b,
    double alpha,
    double alpha2,
    double gamma,
    np.ndarray[np.int64_t, ndim=1] rng_state,
    bint move_other,
    np.ndarray[np.double_t, ndim=1] epochs_per_edge,
    np.ndarray[np.double_t, ndim=1] epochs_per_negative_sample,
    np.ndarray[np.double_t, ndim=1] epoch_of_next_edge,
    np.ndarray[np.double_t, ndim=1] epoch_of_next_negative_sample,
    int current_epoch,
):
    cdef int dim = head_embedding.shape[1]
    cdef int n_edges = head.shape[0]
    cdef int n_vertices = head_embedding.shape[0]

    cdef np.ndarray[np.double_t, ndim=1] current, other, previous
    cdef double dist_squared, grad_coeff, grad_d
    cdef int i, j, d, _, n_neg_samples
    cdef np.int32_t k
    for i in range(n_edges):
        if epoch_of_next_edge[i] <= current_epoch:
            # 1. Sample this edge
            j = head[i]
            k = tail[i]
            current = head_embedding[j, :]
            other = tail_embedding[k, :]

            # move the head vertex closer to its pinned predecessor
            previous = pred_embedding[j, :]
            dist_squared = rdist(current, previous)
            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0
            for d in range(dim):
                grad_d = clip(grad_coeff * (current[d] - previous[d]))
                current[d] += grad_d * alpha2

            # move the tail vertex closer to its pinned predecessor
            dist_squared = rdist(other, previous)
            if move_other:
                previous = pred_embedding[k, :]
                dist_squared = rdist(other, previous)
                if dist_squared > 0.0:
                    grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                    grad_coeff /= a * pow(dist_squared, b) + 1.0
                else:
                    grad_coeff = 0.0
                for d in range(dim):
                    grad_d = clip(grad_coeff * (other[d] - previous[d]))
                    other[d] += grad_d * alpha2

            # move the vertices towards each other
            dist_squared = rdist(current, other)
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
                (current_epoch - epoch_of_next_negative_sample[i]) /
                epochs_per_negative_sample[i]
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
