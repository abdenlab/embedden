#!python
#cython: embedsignature=True
import cython
import numpy as np

cimport numpy as np


cdef extern from "../../src/fitsne/tsne.h":

    cdef cppclass TSNE:
        TSNE() except +

        # Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
        void computeGradient(
            unsigned int *inp_row_P,
            unsigned int *inp_col_P,
            double *inp_val_P,
            double *Y,
            int N,
            int D,
            double *dC,
            double theta,
            unsigned int nthreads
        )

        # Compute the gradient of the t-SNE cost function using the FFT interpolation based approximation for for one dimensional Ys, with variable degree of freedom df
        void computeFftGradientOneDVariableDf(
            unsigned int *inp_row_P,
            unsigned int *inp_col_P,
            double *inp_val_P,
            double *Y,
            int N,
            int D,
            double *dC,
            int n_interpolation_points,
            double intervals_per_integer,
            int min_num_intervals,
            unsigned int nthreads,
            double df
        )

        # Compute the gradient of the t-SNE cost function using the FFT interpolation based approximation for for one dimensional Ys
        void computeFftGradientOneD(
            unsigned int *inp_row_P,
            unsigned int *inp_col_P,
            double *inp_val_P,
            double *Y,
            int N,
            int D,
            double *dC,
            int n_interpolation_points,
            double intervals_per_integer,
            int min_num_intervals,
            unsigned int nthreads
        )

        # Compute the gradient of the t-SNE cost function using the FFT interpolation based approximation, with variable degree of freedom df
        void computeFftGradientVariableDf(
            unsigned int *inp_row_P,
            unsigned int *inp_col_P,
            double *inp_val_P,
            double *Y,
            int N,
            int D,
            double *dC,
            int n_interpolation_points,
            double intervals_per_integer,
            int min_num_intervals,
            unsigned int nthreads,
            double df
        )

        #  Compute the gradient of the t-SNE cost function using the FFT interpolation based approximation
        void computeFftGradient(
            unsigned int *inp_row_P,
            unsigned int *inp_col_P,
            double *inp_val_P,
            double *Y,
            int N,
            int D,
            double *dC,
            int n_interpolation_points,
            double intervals_per_integer,
            int min_num_intervals,
            unsigned int nthreads
        )

        # Compute the exact gradient of the t-SNE cost function
        void computeExactGradient(
            double *P,
            double *Y,
            int N,
            int D,
            double *dC,
            double df
        )

        # Evaluate t-SNE cost function (exactly)
        double evaluateError(
            double *P,
            double *Y,
            int N,
            int D,
            double df
        )

        # Evaluate t-SNE cost function (approximately) using FFT
        double evaluateErrorFft(
            unsigned int *row_P,
            unsigned int *col_P,
            double *val_P,
            double *Y,
            int N,
            int D,
            unsigned int nthreads,
            double df
        )

        # Evaluate t-SNE cost function (approximately)
        double evaluateError(
            unsigned int *row_P,
            unsigned int *col_P,
            double *val_P,
            double *Y,
            int N,
            int D,
            double theta,
            unsigned int nthreads
        )


cdef class Gradient:
    cpdef void compute(
        self,
        double[:, ::1] dY,
        unsigned int[::1] P_row,
        unsigned int[::1] P_col,
        double[::1] P_val,
        double[:, ::1] Y,
    ):
        pass


cdef class GradientBarnesHut(Gradient):
    cdef TSNE *tsne
    cdef double dof
    cdef double theta
    cdef unsigned int nthreads

    def __cinit__(
        self, double dof=1.0, double theta=0.5, unsigned int nthreads=0
    ):
        self.tsne = new TSNE()
        self.dof = dof
        self.theta = theta
        self.nthreads = nthreads

    def __dealloc__(self):
        del self.tsne

    cpdef void compute(
        self,
        double[:, ::1] dY,
        unsigned int[::1] P_row,
        unsigned int[::1] P_col,
        double[::1] P_val,
        double[:, ::1] Y,
    ):
        cdef int n_samples = dY.shape[0]
        cdef int n_components = dY.shape[1]
        self.tsne.computeGradient(
            &P_row[0],
            &P_col[0],
            &P_val[0],
            &Y[0, 0],
            n_samples,
            n_components,
            &dY[0, 0],
            self.theta,
            self.nthreads,
        )


cdef class GradientFFT(Gradient):
    cdef TSNE *tsne
    cdef double dof
    cdef int nterms
    cdef double intervals_per_integer
    cdef int min_num_intervals
    cdef unsigned int nthreads

    def __cinit__(
        self, 
        double dof=1.0,
        int n_interpolation_points=3, 
        double intervals_per_integer=1, 
        int min_num_intervals=50, 
        unsigned int nthreads=0
    ):
        self.tsne = new TSNE()
        self.dof = dof
        self.nterms = n_interpolation_points
        self.intervals_per_integer = intervals_per_integer
        self.min_num_intervals = min_num_intervals
        self.nthreads = nthreads

    def __dealloc__(self):
        del self.tsne

    cpdef void compute(
        self, 
        double[:, ::1] dY,
        unsigned int[::1] P_row,
        unsigned int[::1] P_col,
        double[::1] P_val,
        double[:, ::1] Y,
    ):
        cdef int n_samples = dY.shape[0]
        cdef int n_components = dY.shape[1]
        self.tsne.computeFftGradientVariableDf(
            &P_row[0],
            &P_col[0],
            &P_val[0],
            &Y[0, 0],
            n_samples,
            n_components,
            &dY[0, 0],
            self.nterms,
            self.intervals_per_integer,
            self.min_num_intervals,
            self.nthreads,
            self.dof,
        )


def optimize(
    np.ndarray[np.double_t, ndim=2] Y,
    np.ndarray[np.uint32_t, ndim=1] P_row,
    np.ndarray[np.uint32_t, ndim=1] P_col,
    np.ndarray[np.double_t, ndim=1] P_val,
    Gradient grad_func,
    int max_iter=1000,
    double learning_rate=200,
    double momentum=0.5,
    double max_step_norm=5,
):
    cdef int m = Y.shape[0], n = Y.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] dY = np.zeros((m, n), dtype=float)
    cdef np.ndarray[np.double_t, ndim=2] uY = np.zeros((m, n), dtype=float)
    cdef np.ndarray[np.double_t, ndim=2] gains = np.ones((m, n), dtype=float)
    cdef np.ndarray[np.uint8_t, ndim=2, cast=True] mask
    cdef double step
    cdef int i

    for i in range(max_iter):

        grad_func.compute(dY, P_row, P_col, P_val, Y)

        if momentum == 0:
            # no trickery, just good old fashioned gradient descent
            Y -= learning_rate * dY
        else:
            # apply momentum before gradient descent
            mask = np.sign(dY) != np.sign(uY)
            gains[mask] *= 0.2
            gains[~mask] *= 0.8
            gains[gains < 0.01] = 0.01

            uY = momentum * uY - learning_rate * gains * dY

            if max_step_norm > 0:
                step = np.sqrt(np.sum(uY ** 2))
                if step > max_step_norm:
                    uY *= max_step_norm / step

            Y += uY

        # mean-center
        Y -= np.mean(Y, axis=0)

    return Y


# def solve(
#     np.ndarray[np.double_t, ndim=2] Y,
#     np.ndarray[np.uint32_t, ndim=1] P_row,
#     np.ndarray[np.uint32_t, ndim=1] P_col,
#     np.ndarray[np.double_t, ndim=1] P_val,
#     Gradient grad_func,
#     int max_iter=1000,
#     double learning_rate=200,
#     double early_exag_coeff=12.0,
#     int stop_early_exag_iter=250,
#     double late_exag_coeff=-1,
#     int start_late_exag_iter=-1,
#     double momentum=0.5,
#     double final_momentum=0.8,
#     int mom_switch_iter=250,
#     bint momentum_during_exag=True,
#     double max_step_norm=5,
# ):
#     cdef bint exag = early_exag_coeff != 0
#     cdef double max_sum_cols
#     if early_exag_coeff < 0:
#         max_sum_cols = np.max(np.sum(P_val, axis=0))
#         early_exag_coeff = 1.0 / (learning_rate * max_sum_cols)

#     # early exaggeration
#     P_val *= early_exag_coeff

#     cdef int m = Y.shape[0], n = Y.shape[1]
#     cdef np.ndarray[np.double_t, ndim=2] dY = np.zeros((m, n), dtype=float)
#     cdef np.ndarray[np.double_t, ndim=2] uY = np.zeros((m, n), dtype=float)
#     cdef np.ndarray[np.double_t, ndim=2] gains = np.ones((m, n), dtype=float)
#     cdef np.ndarray[np.uint8_t, ndim=2, cast=True] mask
#     cdef double step
#     cdef int i

#     for i in range(max_iter):

#         grad_func(dY, P_row, P_col, P_val, Y)

#         if momentum == 0 or exag and not momentum_during_exag:
#             # no trickery, just good old fashioned gradient descent
#             Y = Y - learning_rate * dY
#         else:
#             # apply momentum before gradient descent
#             mask = np.sign(dY) != np.sign(uY)
#             gains[mask] *= 0.2
#             gains[~mask] *= 0.8
#             gains[gains < 0.01] = 0.01

#             uY = momentum * uY - learning_rate * gains * dY

#             if max_step_norm > 0:
#                 step = np.sqrt(np.sum(uY ** 2))
#                 if step > max_step_norm:
#                     uY *= max_step_norm / step

#             Y = Y + uY

#         # mean-center
#         Y -= np.mean(Y, axis=0)

#         if i == stop_early_exag_iter:
#             P_val /= early_exag_coeff
#             exag = False

#         if i == start_late_exag_iter:
#             P_val *= late_exag_coeff
#             exag = True

#         if i == mom_switch_iter:
#             momentum = final_momentum

#     return Y

