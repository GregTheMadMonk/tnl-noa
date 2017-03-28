/***************************************************************************
                          Multireduction.h  -  description
                             -------------------
    begin                : May 13, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Containers {
namespace Algorithms {   

template< typename Device >
class Multireduction
{
};

template<>
class Multireduction< Devices::Cuda >
{
public:
   template< typename Operation >
   static bool
   reduce( Operation& operation,
           int n,
           const typename Operation::IndexType size,
           const typename Operation::RealType* deviceInput1,
           const typename Operation::IndexType ldInput1,
           const typename Operation::RealType* deviceInput2,
           typename Operation::ResultType* hostResult );
};

template<>
class Multireduction< Devices::Host >
{
public:
   template< typename Operation >
   static bool
   reduce( Operation& operation,
           int n,
           const typename Operation::IndexType size,
           const typename Operation::RealType* deviceInput1,
           const typename Operation::IndexType ldInput1,
           const typename Operation::RealType* deviceInput2,
           typename Operation::ResultType* hostResult );
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL

#include "Multireduction_impl.h"
