#!python
#cython: embedsignature=True
import cython
import numpy as np

cimport numpy as np
from libcpp cimport bool


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


cpdef double[:, ::1] compute_gradient_barneshut(
    double[:, ::1] dY,
    unsigned int[::1] P_row, 
    unsigned int[::1] P_col, 
    double[::1] P_val, 
    double[:, ::1] Y, 
    double theta, 
    unsigned int nthreads
):
    tsne_obj = new TSNE()
    cdef int n_samples = dY.shape[0]
    cdef int n_components = dY.shape[1]
    try:
        tsne_obj.computeGradient(
            &P_row[0], 
            &P_col[0], 
            &P_val[0], 
            &Y[0, 0], 
            n_samples, 
            n_components, 
            &dY[0, 0], 
            theta, 
            nthreads
        )
    finally:
        del tsne_obj
    return dY


cpdef double[:, ::1] compute_gradient_fft(
    unsigned int[::1] P_row, 
    unsigned int[::1] P_col, 
    double[::1] P_val, 
    double[:, ::1] Y, 
    int nterms,
    double intervals_per_integer,
    int min_num_intervals, 
    unsigned int nthreads,
    double dof
):
    tsne_obj = new TSNE()
    cdef int n_samples = Y.shape[0]
    cdef int n_components = Y.shape[1]
    cdef double[:, ::1] dY = np.zeros(Y.shape)
    try:
        tsne_obj.computeFftGradientVariableDf(
            &P_row[0], 
            &P_col[0], 
            &P_val[0], 
            &Y[0, 0], 
            n_samples, 
            n_components, 
            &dY[0, 0], 
            nterms,
            intervals_per_integer, 
            min_num_intervals, 
            nthreads, 
            dof
        )
    finally:
        del tsne_obj
    return dY
