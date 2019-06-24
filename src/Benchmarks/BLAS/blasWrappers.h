#pragma once

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


/*inline int blasIgamin( int n, const float *x, int incx )
{
   return cblas_isamin( n, x, incx );
}

inline int blasIgamin( int n, const double *x, int incx )
{
   return cblas_idamin( n, x, incx );
}*/


inline float blasGasum( int n, const float *x, int incx )
{
   return cblas_sasum( n, x, incx );
}

inline double blasGasum( int n, const double *x, int incx )
{
   return cblas_dasum( n, x, incx );
}


inline void blasGaxpy( int n, const float alpha,
                       const float *x, int incx,
                       float *y, int incy )
{
   cblas_saxpy( n, alpha, x, incx, y, incy );
}

inline void blasGaxpy( int n, const double alpha,
                       const double* x, int incx,
                       double *y, int incy )
{
   cblas_daxpy( n, alpha, x, incx, y, incy );
}


inline float blasGdot( int n, const float* x, int incx,
                       const float* y, int incy )
{
   return cblas_sdot( n, x, incx, y, incy );
}

inline double blasGdot( int n, const double* x, int incx,
                        const double* y, int incy )
{
   return cblas_ddot( n, x, incx, y, incy );
}


inline float blasGnrm2( int n, const float* x, int incx )
{
   return cblas_snrm2( n, x, incx );
}

inline double blasGnrm2( int n, const double* x, int incx )
{
   return cblas_dnrm2( n, x, incx );
}


inline void blasGscal( int n, const float alpha,
                       float* x, int incx )
{
   cblas_sscal( n, alpha, x, incx );
}

inline void blasGscal( int n, const double alpha,
                       double* x, int incx )
{
   cblas_dscal( n, alpha, x, incx );
}
#endif
