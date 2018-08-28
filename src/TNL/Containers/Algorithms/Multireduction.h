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
#include <TNL/Devices/MIC.h>

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
   template< typename Operation, typename Index >
   static bool
   reduce( Operation& operation,
           const int n,
           const Index size,
           const typename Operation::DataType1* deviceInput1,
           const Index ldInput1,
           const typename Operation::DataType2* deviceInput2,
           typename Operation::ResultType* hostResult );
};

template<>
class Multireduction< Devices::Host >
{
public:
   template< typename Operation, typename Index >
   static bool
   reduce( Operation& operation,
           const int n,
           const Index size,
           const typename Operation::DataType1* deviceInput1,
           const Index ldInput1,
           const typename Operation::DataType2* deviceInput2,
           typename Operation::ResultType* hostResult );
};

template<>
class Multireduction< Devices::MIC >
{
public:
   template< typename Operation, typename Index >
   static bool
   reduce( Operation& operation,
           const int n,
           const Index size,
           const typename Operation::DataType1* deviceInput1,
           const Index ldInput1,
           const typename Operation::DataType2* deviceInput2,
           typename Operation::ResultType* hostResult );
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL

#include "Multireduction_impl.h"
