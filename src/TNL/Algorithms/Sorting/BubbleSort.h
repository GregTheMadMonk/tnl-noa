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
#include <TNL/Assert.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
   namespace Algorithms {
      namespace Sorting {

struct BubbleSort
{
   template< typename Device, typename Index, typename Compare, typename Swap >
   void static inplaceSort( const Index begin, const Index end, Compare& compare, Swap& swap )
   {
      if( std::is_same< Device, Devices::Host >::value ||
          std::is_same< Device, Devices::Sequential >::value )
      {
         Index left( begin ), right( end -1 );
         while( left < right )
         {
            //int lastChange( end -1 );
            for( int j = left; j < right - 1; j++ )
            {
               TNL_ASSERT_LT( j+1, end, "" );
               if( ! compare( j, j+1 ) )
               {
                  swap( j, j+1 );
                  //lastChange = j;
               }
            }
            right--; //lastChange;
            for( int j = right; j >= left; j-- )
            {
               TNL_ASSERT_LT( j+1, end, "" );
               if( ! compare( j, j+1 ) )
               {
                  swap( j, j+1 );
                  //lastChange = j;
               }
            }
            left++; //lastChange;
         }
      }
      else
         throw Exceptions::NotImplementedError();
   }
};

      } // namespace Sorting
   } // namespace Algorithms
} //namespace TNL


