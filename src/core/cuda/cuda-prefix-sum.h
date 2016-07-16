/***************************************************************************
                          cuda-prefix-sum.h  -  description
                             -------------------
    begin                : Jan 18, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef CUDA_PREFIX_SUM_H_
#define CUDA_PREFIX_SUM_H_

enum enumPrefixSumType { exclusivePrefixSum = 0,
                         inclusivePrefixSum };

template< typename DataType,
          typename Operation,
          typename Index >
bool cudaPrefixSum( const Index size,
                    const Index blockSize,
                    const DataType *deviceInput,
                    DataType* deviceOutput,
                    const Operation& operation,
                    const enumPrefixSumType prefixSumType = inclusivePrefixSum );


#include <core/cuda/cuda-prefix-sum_impl.h>

#endif /* CUDA_PREFIX_SUM_H_ */
