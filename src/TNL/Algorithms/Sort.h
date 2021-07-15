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

#include <TNL/Algorithms/Sorting/DefaultSorter.h>

namespace TNL {
   namespace Algorithms {


template< typename Array,
          typename Sorter = typename Sorting::DefaultSorter< typename Array::DeviceType >::SorterType >
void sort( Array& array )
{
   Sorter::sort( array );
}

template< typename Array,
          typename Compare,
          typename Sorter = typename Sorting::DefaultSorter< typename Array::DeviceType >::SorterType >
void sort( Array& array, const Compare& compare )
{
   Sorter::sort( array, compare );
}

template< typename Device,
          typename Index,
          typename Fetch,
          typename Compare,
          typename Swap,
          typename Sorter = typename Sorting::DefaultInplaceSorter< Device >::SorterType >
void inplaceSort( const Index begin, const Index end, const Fetch& fetch, const Compare& compare, const Swap& swap )
{
   Sorter::inplaceSort( begin, end, fetch, compare, swap );
}

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
