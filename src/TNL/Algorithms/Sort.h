/***************************************************************************
                          Sort.h  -  description
                             -------------------
    begin                : Jul 12, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Xuan Thang Nguyen

#pragma once

#include <utility>  // std::pair, std::forward

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Algorithms/Sorting/bitonicSort.h>
#include <TNL/Algorithms/Sorting/quicksort.h>

namespace TNL {
   namespace Algorithms {


template <typename Array, typename Function>
bool isSorted( const Array& arr, const Function& cmp )
{
   using Device = typename Array::DeviceType;
   if (arr.getSize() <= 1)
      return true;

   auto view = arr.getConstView();
   auto fetch = [=] __cuda_callable__(int i) { return ! cmp( view[ i ], view[ i - 1 ] ); };
   auto reduction = [] __cuda_callable__(bool a, bool b) { return a && b; };
   return TNL::Algorithms::reduce< Device >( 1, arr.getSize(), fetch, reduction, true );
}

template< typename Array >
bool isSorted( const Array& arr)
{
   using Value = typename Array::ValueType;
   return isSorted( arr, [] __cuda_callable__( const Value& a, const Value& b ) { return a < b; });
}


   } // namespace Algorithms
} // namespace TNL
