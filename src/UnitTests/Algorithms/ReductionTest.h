/***************************************************************************
                          ReductionTest.h  -  description
                             -------------------
    begin                : Jul 2, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/Reduction.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

using namespace TNL;

#ifdef HAVE_GTEST
TEST( ReduceTest, sum )
{
   using Array = Containers::Array< int, Devices::Host >;
   Array a;
   for( int size = 100; size <= 1000; size *= 10 )
   {
      a.setSize( size );
      a.setValue( 1 );
      auto a_view = a.getView();

      auto fetch = [=] __cuda_callable__ ( int idx ) { return a_view[ idx ]; };
      auto res = Algorithms::reduce< Devices::Host >( ( int ) 0, size, fetch, TNL::Plus<>{} );
      EXPECT_EQ( res, size );

   }
}


#endif

#include "../main.h"
