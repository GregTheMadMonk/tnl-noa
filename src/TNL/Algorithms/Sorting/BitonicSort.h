/***************************************************************************
                          BitonicSort.h  -  description
                             -------------------
    begin                : Jul 14, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Xuan Thang Nguyen, Tomas Oberhuber

#pragma once

#include <TNL/Algorithms/Sorting/detail/bitonicSort.h>

namespace TNL {
   namespace Algorithms {
      namespace Sorting {

struct BitonicSort
{
   template< typename Array >
   void static sort( Array& array )
   {
      bitonicSort( array );
   }

   template< typename Array, typename Compare >
   void static sort( Array& array, const Compare& compare )
   {
      bitonicSort( array, compare );
   }

};

      } // namespace Sorting
   } // namespace Algorithms
} //namespace TNL


