#pragma once

#ifdef HAVE_CUDA
#ifdef HAVE_CUBLAS

#include <cublas_v2.h>

inline cublasStatus_t
cublasGdot( cublasHandle_t handle, int n,
            const float        *x, int incx,
            const float        *y, int incy,
            float         *result )
{
    return cublasSdot( handle, n, x, incx, y, incy, result );
}

inline cublasStatus_t
cublasGdot( cublasHandle_t handle, int n,
            const double       *x, int incx,
            const double       *y, int incy,
            double        *result )
{
    return cublasDdot( handle, n, x, incx, y, incy, result );
}

#endif
#endif
