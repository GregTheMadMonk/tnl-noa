/***************************************************************************
                          StaticArray_impl.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/param-types.h>
#include <TNL/Math.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/StaticFor.h>

namespace TNL {
namespace Containers {

namespace detail {

////
// Lambdas used together with StaticFor for static loop unrolling in the
// implementation of the StaticArray
template< typename LeftValue, typename RightValue = LeftValue >
auto assignArrayLambda = [] __cuda_callable__ ( int i, LeftValue* data, const RightValue* v ) { data[ i ] = v[ i ]; };

template< typename LeftValue, typename RightValue = LeftValue >
auto assignValueLambda = [] __cuda_callable__ ( int i, LeftValue* data, const RightValue v ) { data[ i ] = v; };

////
// StaticArrayComparator does static loop unrolling of array comparison
template< int Size, typename LeftValue, typename RightValue, int Index >
struct StaticArrayComparator
{
   __cuda_callable__
   static bool EQ( const StaticArray< Size, LeftValue >& left,
                   const StaticArray< Size, RightValue >& right )
   {
      if( left[ Index ] == right[ Index ] )
         return StaticArrayComparator< Size, LeftValue, RightValue, Index + 1 >::EQ( left, right );
      return false;
   }
};

template< int Size, typename LeftValue, typename RightValue >
struct StaticArrayComparator< Size, LeftValue, RightValue, Size >
{
   __cuda_callable__
   static bool EQ( const StaticArray< Size, LeftValue >& left,
                   const StaticArray< Size, RightValue >& right )
   {
      return true;
   }
};

////
// Static array sort does static loop unrolling of array sort.
// It performs static variant of bubble sort as follows:
// 
// for( int k = Size - 1; k > 0; k--)
//   for( int i = 0; i < k; i++ )
//      if( data[ i ] > data[ i+1 ] )
//         swap( data[ i ], data[ i+1 ] );
template< int k, int i, typename Value >
struct StaticArraySort
{
   __cuda_callable__
   static void exec( Value* data ) {
      if( data[ i ] > data[  i + 1 ] )
         swap( data[ i ], data[ i+1 ] );
      StaticArraySort< k, i + 1, Value >::exec( data );
   }
};

template< int k, typename Value >
struct StaticArraySort< k, k, Value >
{
   __cuda_callable__
   static void exec( Value* data ) {
      StaticArraySort< k - 1, 0, Value >::exec( data );
   }
};

template< typename Value >
struct StaticArraySort< 0, 0, Value >
{
   __cuda_callable__
   static void exec( Value* data ) {}
};

} // namespace detail


template< int Size, typename Value >
__cuda_callable__
constexpr int StaticArray< Size, Value >::getSize()
{
   return Size;
}

template< int Size, typename Value >
__cuda_callable__
StaticArray< Size, Value >::StaticArray()
{
}

template< int Size, typename Value >
   template< typename _unused >
__cuda_callable__
StaticArray< Size, Value >::StaticArray( const Value v[ Size ] )
{
   StaticFor< 0, Size >::exec( detail::assignArrayLambda< Value >, data, v );
}

template< int Size, typename Value >
__cuda_callable__
inline StaticArray< Size, Value >::StaticArray( const Value& v )
{
   StaticFor< 0, Size >::exec( detail::assignValueLambda< Value >, data, v );
}

template< int Size, typename Value >
__cuda_callable__
inline StaticArray< Size, Value >::StaticArray( const StaticArray< Size, Value >& v )
{
   StaticFor< 0, Size >::exec( detail::assignArrayLambda< Value >, data, v.getData() );
}

template< int Size, typename Value >
StaticArray< Size, Value >::StaticArray( const std::initializer_list< Value > &elems)
{
   auto it = elems.begin();
   for( int i = 0; i < getSize(); i++ )
      data[ i ] = *it++;
}

template< int Size, typename Value >
 __cuda_callable__
StaticArray< Size, Value >::StaticArray( const Value& v1, const Value& v2 )
{
   static_assert( Size == 2, "This constructor can be called only for arrays with Size = 2." );
   data[ 0 ] = v1;
   data[ 1 ] = v2;
}

template< int Size, typename Value >
 __cuda_callable__
StaticArray< Size, Value >::StaticArray( const Value& v1, const Value& v2, const Value& v3 )
{
   static_assert( Size == 3, "This constructor can be called only for arrays with Size = 3." );
   data[ 0 ] = v1;
   data[ 1 ] = v2;
   data[ 2 ] = v3;
}

template< int Size, typename Value >
String StaticArray< Size, Value >::getType()
{
   return String( "Containers::StaticArray< " ) +
          convertToString( Size ) +
          String( ", " ) +
          TNL::getType< Value >() +
          String( " >" );
}

template< int Size, typename Value >
__cuda_callable__
inline Value* StaticArray< Size, Value >::getData()
{
   return data;
}

template< int Size, typename Value >
__cuda_callable__
inline const Value* StaticArray< Size, Value >::getData() const
{
   return data;
}

template< int Size, typename Value >
__cuda_callable__
inline const Value& StaticArray< Size, Value >::operator[]( int i ) const
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, Size, "Element index is out of bounds." );
   return data[ i ];
}

template< int Size, typename Value >
__cuda_callable__
inline Value& StaticArray< Size, Value >::operator[]( int i )
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, Size, "Element index is out of bounds." );
   return data[ i ];
}
template< int Size, typename Value >
__cuda_callable__
inline Value& StaticArray< Size, Value >::x()
{
   return data[ 0 ];
}

template< int Size, typename Value >
__cuda_callable__
inline const Value& StaticArray< Size, Value >::x() const
{
   return data[ 0 ];
}

template< int Size, typename Value >
__cuda_callable__
inline Value& StaticArray< Size, Value >::y()
{
   static_assert( Size > 1, "Cannot call StaticArray< Size, Value >::y() for arrays with Size < 2." );
   return data[ 1 ];
}

template< int Size, typename Value >
__cuda_callable__
inline const Value& StaticArray< Size, Value >::y() const
{
   static_assert( Size > 1, "Cannot call StaticArray< Size, Value >::y() for arrays with Size < 2." );
   return data[ 1 ];
}

template< int Size, typename Value >
__cuda_callable__
inline Value& StaticArray< Size, Value >::z()
{
   static_assert( Size > 1, "Cannot call StaticArray< Size, Value >::z() for arrays with Size < 3." );
   return data[ 2 ];
}

template< int Size, typename Value >
__cuda_callable__
inline const Value& StaticArray< Size, Value >::z() const
{
   static_assert( Size > 1, "Cannot call StaticArray< Size, Value >::z() for arrays with Size < 3." );
   return data[ 2 ];
}

template< int Size, typename Value >
__cuda_callable__
inline StaticArray< Size, Value >& StaticArray< Size, Value >::operator=( const StaticArray< Size, Value >& array )
{
   StaticFor< 0, Size >::exec( detail::assignArrayLambda< Value >, data, array.getData() );
   return *this;
}

template< int Size, typename Value >
   template< typename Array >
__cuda_callable__
inline StaticArray< Size, Value >& StaticArray< Size, Value >::operator=( const Array& array )
{
   StaticFor< 0, Size >::exec( detail::assignArrayLambda< Value, typename Array::ValueType >, data, array.getData() );
   return *this;
}

template< int Size, typename Value >
   template< typename Array >
__cuda_callable__
inline bool StaticArray< Size, Value >::operator==( const Array& array ) const
{
   return detail::StaticArrayComparator< Size, Value, typename Array::ValueType, 0 >::EQ( *this, array );
}

template< int Size, typename Value >
   template< typename Array >
__cuda_callable__
inline bool StaticArray< Size, Value >::operator!=( const Array& array ) const
{
   return ! this->operator==( array );
}

template< int Size, typename Value >
   template< typename OtherValue >
__cuda_callable__
StaticArray< Size, Value >::
operator StaticArray< Size, OtherValue >() const
{
   StaticArray< Size, OtherValue > aux;
   StaticFor< 0, Size >::exec( detail::assignArrayLambda< OtherValue, Value >, aux.getData(), data );
   return aux;
}

template< int Size, typename Value >
__cuda_callable__
inline void StaticArray< Size, Value >::setValue( const ValueType& val )
{
   StaticFor< 0, Size >::exec( detail::assignValueLambda< Value >, data, val );
}

template< int Size, typename Value >
bool StaticArray< Size, Value >::save( File& file ) const
{
   file.save< Value, Value, Devices::Host >( data, Size );
   return true;
}

template< int Size, typename Value >
bool StaticArray< Size, Value >::load( File& file)
{
   file.load< Value, Value, Devices::Host >( data, Size );
   return true;
}

template< int Size, typename Value >
void StaticArray< Size, Value >::sort()
{
   detail::StaticArraySort< Size - 1, 0, Value >::exec( data );
}

template< int Size, typename Value >
std::ostream& StaticArray< Size, Value >::write( std::ostream& str, const char* separator ) const
{
   for( int i = 0; i < Size - 1; i++ )
      str << data[ i ] << separator;
   str << data[ Size - 1 ];
   return str;
}

template< int Size, typename Value >
std::ostream& operator<<( std::ostream& str, const StaticArray< Size, Value >& a )
{
   str << "[ ";
   a.write( str, ", " );
   str << " ]";
   return str;
}

} // namespace Containers
} // namespace TNL