/***************************************************************************
                          Reduction.h  -  description
                             -------------------
    begin                : Oct 28, 2010
    copyright            : (C) 2010 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once 

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/MIC.h>

namespace TNL {
namespace Containers {
namespace Algorithms {   

template< typename Device >
class Reduction
{
};

template<>
class Reduction< Devices::Cuda >
{
public:
   template< typename Operation, typename Index >
   static typename Operation::ResultType
   reduce( Operation& operation,
           const Index size,
           const typename Operation::DataType1* deviceInput1,
           const typename Operation::DataType2* deviceInput2 );
};

template<>
class Reduction< Devices::Host >
{
public:
   template< typename Operation, typename Index >
   static typename Operation::ResultType
   reduce( Operation& operation,
           const Index size,
           const typename Operation::DataType1* deviceInput1,
           const typename Operation::DataType2* deviceInput2 );
};

template<>
class Reduction< Devices::MIC >
{
public:
   template< typename Operation, typename Index >
   static typename Operation::ResultType
   reduce( Operation& operation,
           const Index size,
           const typename Operation::DataType1* deviceInput1,
           const typename Operation::DataType2* deviceInput2 );
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL

#include "Reduction_impl.h"
