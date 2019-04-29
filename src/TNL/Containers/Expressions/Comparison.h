/***************************************************************************
                          StaticComparison.h  -  description
                             -------------------
    begin                : Apr 19, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>
#include <TNL/Containers/Algorithms/Reduction.h>

namespace TNL {
   namespace Containers {
      namespace Expressions {

template< typename T1,
          typename T2 >
__cuda_callable__
bool StaticComparisonEQ( const T1& a, const T2& b )
{
   TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );
   for( int i = 0; i < a.getSize(); i++ )
      if( a[ i ] != b[ i ] )
         return false;
   return true;
}

template< typename T1,
          typename T2 >
__cuda_callable__
bool StaticComparisonNE( const T1& a, const T2& b )
{
   return ! StaticComparisonEQ( a, b );
}

template< typename T1,
          typename T2 >
__cuda_callable__
bool StaticComparisonGT( const T1& a, const T2& b )
{
   TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );
   for( int i = 0; i < a.getSize(); i++ )
      if( a[ i ] <= b[ i ] )
         return false;
   return true;
}

template< typename T1,
          typename T2 >
__cuda_callable__
bool StaticComparisonLE( const T1& a, const T2& b )
{
   return ! StaticComparisonGT( a, b );
}

template< typename T1,
          typename T2 >
__cuda_callable__
bool StaticComparisonLT( const T1& a, const T2& b )
{
   TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );
   for( int i = 0; i < a.getSize(); i++ )
      if( a[ i ] >= b[ i ] )
         return false;
   return true;
}

template< typename T1,
          typename T2 >
__cuda_callable__
bool StaticComparisonGE( const T1& a, const T2& b )
{
   return ! StaticComparisonLT( a, b );
}

////
// Non-static comparison
template< typename T1,
          typename T2 >
__cuda_callable__
bool ComparisonEQ( const T1& a, const T2& b )
{
   if( a.getSize() != b.getSize() )
      return false;
   if( a.getSize() == 0 )
      return true;

   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a[ i ] == b[ i ] ); };
   auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
   return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );
}

template< typename T1,
          typename T2 >
__cuda_callable__
bool ComparisonNE( const T1& a, const T2& b )
{
   return ! ComparisonEQ( a, b );
}

template< typename T1,
          typename T2 >
__cuda_callable__
bool ComparisonGT( const T1& a, const T2& b )
{
   TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );

   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a[ i ] > b[ i ] ); };
   auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
   return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );
}

template< typename T1,
          typename T2 >
__cuda_callable__
bool ComparisonLE( const T1& a, const T2& b )
{
   return ! ComparisonGT( a, b );
}

template< typename T1,
          typename T2 >
__cuda_callable__
bool ComparisonLT( const T1& a, const T2& b )
{
   TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );

   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a[ i ] < b[ i ] ); };
   auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
   return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );
}

template< typename T1,
          typename T2 >
__cuda_callable__
bool ComparisonGE( const T1& a, const T2& b )
{
   return ! ComparisonLT( a, b );
}

      } //namespace Expressions
   } // namespace Containers
} // namespace TNL
