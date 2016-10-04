#pragma once

namespace TNL {
namespace Containers {
namespace Algorithms {   

template< typename Operation >
bool multireductionOnCudaDevice( const Operation& operation,
                                 int n,
                                 const typename Operation::IndexType size,
                                 const typename Operation::RealType* deviceInput1,
                                 const typename Operation::RealType* deviceInput2,
                                 typename Operation::ResultType* hostResult );

template< typename Operation >
bool multireductionOnHostDevice( const Operation& operation,
                                 int n,
                                 const typename Operation::IndexType size,
                                 const typename Operation::RealType* deviceInput1,
                                 const typename Operation::RealType* deviceInput2,
                                 typename Operation::ResultType* hostResult );

} // namespace Algorithms
} // namespace Containers
} // namespace TNL

#include "Multireduction_impl.h"
