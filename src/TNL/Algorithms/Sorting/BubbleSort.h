/***************************************************************************
                          BubbleSort.h  -  description
                             -------------------
    begin                : Jul 26, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber

#pragma once

#include <algorithm>

namespace TNL {
   namespace Algorithms {
      namespace Sorting {

struct BubbleSort
{
   template< typename Device, typename Index, typename Compare, typename Swap >
   void static inplaceSort( const Index begin, const Index end, const Compare& compare, const Swap& swap )
   {
      if( std::is_same< Device, Devices::Host >::value )
      {
         Index left( begin ), right( end );
         while( left < right )
         {
            int lastChange;
            for( int j = left; j < right - 1; j++ )
                  if( ! compare( j, j+1 ) )
                  {
                     swap( j, j+1 );
                     lastChange = j;
                  }
            right = lastChange;
            for( int j = right - 1; j >= left; j-- )
                  if( ! compare( j, j+1 ) )
                  {
                     swap( j, j+1 );
                     lastChange = j;
                  }
            left = lastChange + 1;
         }
      }
      else
         TNL_ASSERT( false, std::cerr <<  "inplace bubble sort is implemented only for CPU" << std::endl );
   }
};

      } // namespace Sorting
   } // namespace Algorithms
} //namespace TNL


