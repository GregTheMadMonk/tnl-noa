#pragma once

#ifdef HAVE_CUDA

#ifdef HAVE_BLAS

#include <cblas.h>

inline int blasIgamax( int n, const float *x, int incx )
{
   return cblas_isamax( n, x, incx );
}

inline int blasIgamax( int n, const double *x, int incx )
{
   return cblas_idamax( n, x, incx );
}


inline int blasIgamin( int n, const float *x, int incx )
{
   return cblas_Isamin( n, x, incx );
}

inline int blasIgamin( int n, const double *x, int incx )
{
   return cblas_Idamin( n, x, incx );
}


inline float blasGasum( int n, const float *x, int incx )
{
   return cblas_sasum( n, x, incx );
}

inline double blasGasum( int n, const double *x, int incx )
{
   return cblas_dasum( n, x, incx );
}


inline void
blasGaxpy( int n, const float *alpha, 
           const float *x, int incx,
           float *y, int incy )
{
   cblas_saxpy( n, alpha, x, incx, y, incy );
}

inline blasStatus_t
blasGaxpy( blasHandle_t int n,
             const double          *alpha,
             const double          *x, int incx,
             double                *y, int incy )
{
   return cblas_Daxpy( n, alpha, x, incx, y, incy );
}


inline blasStatus_t
blasGdot( blasHandle_t int n,
            const float        *x, int incx,
            const float        *y, int incy,
            float         *result )
{
   return cblas_Sdot( n, x, incx, y, incy, result );
}

inline blasStatus_t
blasGdot( blasHandle_t int n,
            const double       *x, int incx,
            const double       *y, int incy,
            double        *result )
{
   return cblas_Ddot( n, x, incx, y, incy, result );
}


inline blasStatus_t
blasGnrm2( blasHandle_t int n,
             const float           *x, int incx, float  *result )
{
   return cblas_Snrm2( n, x, incx, result );
}

inline blasStatus_t
blasGnrm2( blasHandle_t int n,
             const double          *x, int incx, double *result )
{
   return cblas_Dnrm2( n, x, incx, result );
}


inline blasStatus_t
blasGscal( blasHandle_t int n,
             const float           *alpha,
             float           *x, int incx )
{
   return cblas_Sscal( n, alpha, x, incx );
}

inline blasStatus_t
blasGscal( blasHandle_t int n,
             const double          *alpha,
             double          *x, int incx )
{
   return cblas_Dscal( n, alpha, x, incx );
}

#endif
#endif
