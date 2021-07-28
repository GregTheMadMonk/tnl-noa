/***************************************************************************
                          STLSort.h  -  description
                             -------------------
    begin                : Jul 14, 2021
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

struct STLSort
{
   template< typename Array >
   void static sort( Array& array )
   {
      std::sort( array.getData(), array.getData() + array.getSize() );
   }

   template< typename Array, typename Compare >
   void static sort( Array& array, const Compare& compare )
   {
      std::sort( array.getData(), array.getData() + array.getSize(), compare );
   }
};

      } // namespace Sorting
   } // namespace Algorithms
} //namespace TNL
