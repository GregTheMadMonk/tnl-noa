/***************************************************************************
                          cuda-prefix-sum_impl.cu  -  description
                             -------------------
    begin                : Jan 18, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Containers/Algorithms/cuda-prefix-sum.h>
 
namespace TNL {
namespace Devices {

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

template bool cudaPrefixSum( const int size,
                             const int blockSize,
                             const int *deviceInput,
                             int* deviceOutput,
                             tnlParallelReductionSum< int, int >& operation,
                             const enumPrefixSumType prefixSumType );


#ifdef INSTANTIATE_FLOAT
template bool cudaPrefixSum( const int size,
                             const int blockSize,
                             const float *deviceInput,
                             float* deviceOutput,
                             tnlParallelReductionSum< float, int >& operation,
                             const enumPrefixSumType prefixSumType );
#endif

template bool cudaPrefixSum( const int size,
                             const int blockSize,
                             const double *deviceInput,
                             double* deviceOutput,
                             tnlParallelReductionSum< double, int >& operation,
                             const enumPrefixSumType prefixSumType );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool cudaPrefixSum( const int size,
                             const int blockSize,
                             const long double *deviceInput,
                             long double* deviceOutput,
                             tnlParallelReductionSum< long double, int >& operation,
                             const enumPrefixSumType prefixSumType );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool cudaPrefixSum( const long int size,
                             const long int blockSize,
                             const int *deviceInput,
                             int* deviceOutput,
                             tnlParallelReductionSum< int, long int >& operation,
                             const enumPrefixSumType prefixSumType );


#ifdef INSTANTIATE_FLOAT
template bool cudaPrefixSum( const long int size,
                             const long int blockSize,
                             const float *deviceInput,
                             float* deviceOutput,
                             tnlParallelReductionSum< float, long int >& operation,
                             const enumPrefixSumType prefixSumType );
#endif

template bool cudaPrefixSum( const long int size,
                             const long int blockSize,
                             const double *deviceInput,
                             double* deviceOutput,
                             tnlParallelReductionSum< double, long int >& operation,
                             const enumPrefixSumType prefixSumType );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool cudaPrefixSum( const long int size,
                             const long int blockSize,
                             const long double *deviceInput,
                             long double* deviceOutput,
                             tnlParallelReductionSum< long double, long int >& operation,
                             const enumPrefixSumType prefixSumType );
#endif
#endif
#endif

} // namespace Devices
} // namespace TNL
