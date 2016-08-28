/***************************************************************************
                          tnlCublasWrapper.h  -  description
                             -------------------
    begin                : Apr 7, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#if defined HAVE_CUBLAS && defined HAVE_CUDA
#include <cublas_v2.h>
#endif

namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename Real1,
          typename Real2,
          typename Index >
class tnlCublasWrapper
{
    public:
        static bool dot( const Real1* v1, const Real2* v2, const Index size, Real1& result)
        {
            return false;
        }
};

#if defined HAVE_CUBLAS && defined HAVE_CUDA

template< typename Index >
class tnlCublasWrapper< float, float, Index >
{
    public:
        static bool dot( const float* v1, const float* v2, const Index size, float& result)
        {

            cublasHandle_t handle;
            cublasCreate( &handle );
            cublasSdot( handle, size, v1, 1, v2, 1, &result );
            cublasDestroy( handle );
            return false;
        }
};

template< typename Index >
class tnlCublasWrapper< double, double, Index >
{
    public:
        static bool dot( const double* v1, const double* v2, const Index size, double& result)
        {
            cublasHandle_t handle;
            cublasCreate( &handle );
            cublasDdot( handle, size, v1, 1, v2, 1, &result );
            cublasDestroy( handle );
            return false;
        }
};
#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL

