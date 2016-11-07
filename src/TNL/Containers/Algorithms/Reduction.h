/***************************************************************************
                          Reduction.h  -  description
                             -------------------
    begin                : Oct 28, 2010
    copyright            : (C) 2010 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

namespace TNL {
namespace Containers {
namespace Algorithms {   

// TODO: rename to
//   template< typename Device >
//   class Reduction
//   {};
//
// and make a specialization for Devices::Host (as it is done in Multireduction.h)
// It should be as fast as all the manual implementations in VectorOperations.

template< typename Operation >
bool reductionOnCudaDevice( const Operation& operation,
                            const typename Operation :: IndexType size,
                            const typename Operation :: RealType* deviceInput1,
                            const typename Operation :: RealType* deviceInput2,
                            typename Operation :: ResultType& result );

} // namespace Algorithms
} // namespace Containers
} // namespace TNL

#include <TNL/Containers/Algorithms/Reduction_impl.h>
