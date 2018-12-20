/***************************************************************************
                          grid-traversing.h  -  description
                             -------------------
    begin                : Dec 19, 2018
    copyright            : (C) 2018 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber

#pragma once

#include "../Benchmarks.h"


#include <TNL/Containers/Vector.h>

namespace TNL {
   namespace Benchmarks {
   
template< int Dimension,
          typename Real = double,
          typename Index = int >
class benchmarkTraversingFullGrid
{
   public:

      static void run ( Benchmark& benchmark, std::size_t size )
      {

      }
};
   } // namespace Benchmarks
} // namespace TNL