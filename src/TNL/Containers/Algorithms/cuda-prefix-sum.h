/***************************************************************************
                          cuda-prefix-sum.h  -  description
                             -------------------
    begin                : Jan 18, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

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

} // namespace TNL

#include <TNL/Containers/Algorithms/cuda-prefix-sum_impl.h>
