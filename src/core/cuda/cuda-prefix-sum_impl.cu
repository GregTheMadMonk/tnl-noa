/***************************************************************************
                          cuda-prefix-sum_impl.cu  -  description
                             -------------------
    begin                : Jan 18, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 
#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#include <core/cuda/cuda-prefix-sum.h>
         
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
