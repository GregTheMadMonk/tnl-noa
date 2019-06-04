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
namespace Containers {
namespace Algorithms {

enum class PrefixSumType
{
   exclusive,
   inclusive
};

template< typename DataType,
          typename Operation,
          typename VolatileOperation,
          typename Index >
void cudaPrefixSum( const Index size,
                    const Index blockSize,
                    const DataType *deviceInput,
                    DataType* deviceOutput,
                    Operation& operation,
                    VolatileOperation& volatileOperation,
                    const DataType& zero,
                    const PrefixSumType prefixSumType = PrefixSumType::inclusive );

} // namespace Algorithms
} // namespace Containers
} // namespace TNL

#include <TNL/Containers/Algorithms/cuda-prefix-sum_impl.h>
