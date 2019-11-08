/***************************************************************************
                          StaticVectorFor.h  -  description
                             -------------------
    begin                : July 12, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/StaticVector.h>

namespace TNL {
namespace Algorithms {

struct StaticVectorFor
{
   template< typename Index,
             typename Function,
             typename... FunctionArgs,
             int dim >
   static void exec( const Containers::StaticVector< dim, Index >& begin,
                     const Containers::StaticVector< dim, Index >& end,
                     Function f,
                     FunctionArgs... args )
   {
      static_assert( 1 <= dim && dim <= 3, "unsupported dimension" );
      Containers::StaticVector< dim, Index > index;

      if( dim == 1 ) {
         for( index[0] = begin[0]; index[0] < end[0]; index[0]++ )
            f( index, args... );
      }

      if( dim == 2 ) {
         for( index[1] = begin[1]; index[1] < end[1]; index[1]++ )
         for( index[0] = begin[0]; index[0] < end[0]; index[0]++ )
               f( index, args... );
      }

      if( dim == 3 ) {
         for( index[2] = begin[2]; index[2] < end[2]; index[2]++ )
         for( index[1] = begin[1]; index[1] < end[1]; index[1]++ )
         for( index[0] = begin[0]; index[0] < end[0]; index[0]++ )
            f( index, args... );
      }
   }
};

} // namespace Algorithms
} // namespace TNL
