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
#include "WriteOne.h"

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
         auto reset = [&]()
         {};
         
         auto testHost = [&] ()
         {
            WriteOne< Dimension, Devices::Host, Real, Index >::run( size );
         }; 
         
         auto testCuda = [&] ()
         {
            WriteOne< Dimension, Devices::Cuda, Real, Index >::run( size );
         }; 
         
         benchmark.setOperation( "writeOne", size * sizeof( Real ) );
         benchmark.time( reset, "CPU", testHost );
#ifdef HAVE_CUDA
         benchmark.time( reset, "GPU", testCuda );
#endif

      }
};
   } // namespace Benchmarks
} // namespace TNL