/***************************************************************************
                          Math.h  -  description
                             -------------------
    begin                : 2005/07/05
    copyright            : (C) 2005 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <cmath>
#include <type_traits>
#include <algorithm>

#include <TNL/Devices/CudaCallable.h>

namespace TNL {

template< typename T1, typename T2 >
using enable_if_same_base = std::enable_if< std::is_same< typename std::decay< T1 >::type, T2 >::value, T2 >;

template< typename T1, typename T2 >
using both_integral_or_floating = typename std::conditional<
         ( std::is_integral< T1 >::value && std::is_integral< T2 >::value ) ||
         ( std::is_floating_point< T1 >::value && std::is_floating_point< T2 >::value ),
   std::true_type,
   std::false_type >::type;

// 1. If both types are integral or floating-point, the larger type is selected.
// 2. If one type is integral and the other floating-point, the floating-point type is selected.
// Casting both arguments to the same type is necessary because std::min and std::max
// are implemented as a single-type template.
template< typename T1, typename T2 >
using larger_type = typename std::conditional<
         ( both_integral_or_floating< T1, T2 >::value && sizeof(T1) >= sizeof(T2) ) ||
         std::is_floating_point<T1>::value,
   T1, T2 >::type;

/***
 * This function returns minimum of two numbers.
 * GPU device code uses the functions defined in the CUDA's math_functions.h,
 * MIC uses trivial override and host uses the STL functions.
 */
template< typename T1, typename T2, typename ResultType = larger_type< T1, T2 > >
__cuda_callable__ inline
ResultType min( const T1& a, const T2& b )
{
#if defined(__CUDA_ARCH__)
   return ::min( (ResultType) a, (ResultType) b );
#elif defined(__MIC__)
   return a < b ? a : b;
#else
   return std::min( (ResultType) a, (ResultType) b );
#endif
}


/***
 * This function returns maximum of two numbers.
 * GPU device code uses the functions defined in the CUDA's math_functions.h,
 * MIC uses trivial override and host uses the STL functions.
 */
template< typename T1, typename T2, typename ResultType = larger_type< T1, T2 > >
__cuda_callable__
ResultType max( const T1& a, const T2& b )
{
#if defined(__CUDA_ARCH__)
   return ::max( (ResultType) a, (ResultType) b );
#elif defined(__MIC__)
   return a > b ? a : b;
#else
   return std::max( (ResultType) a, (ResultType) b );
#endif
}


/***
 * This function returns absolute value of given number.
 */
template< class T >
__cuda_callable__ inline
T abs( const T& n )
{
#if defined(__MIC__)
   if( n < ( T ) 0 )
      return -n;
   return n;
#else
   return std::abs( n );
#endif
}


template< typename T1, typename T2, typename ResultType = larger_type< T1, T2 > >
__cuda_callable__ inline
ResultType pow( const T1& base, const T2& exp )
{
#if defined(__CUDA_ARCH__) || defined(__MIC__)
   return ::pow( (ResultType) base, (ResultType) exp );
#else
   return std::pow( (ResultType) base, (ResultType) exp );
#endif
}


template< typename T >
__cuda_callable__ inline
T sqrt( const T& value )
{
#if defined(__CUDA_ARCH__) || defined(__MIC__)
   return ::sqrt( value );
#else
   return std::sqrt( value );
#endif
}


template< typename Type >
__cuda_callable__
void swap( Type& a, Type& b )
{
   Type tmp( a );
   a = b;
   b = tmp;
}

template< class T >
__cuda_callable__
T sign( const T& a )
{
   if( a < ( T ) 0 ) return ( T ) -1;
   if( a == ( T ) 0 ) return ( T ) 0;
   return ( T ) 1;
}

template< typename Real >
__cuda_callable__
bool isSmall( const Real& v,
              const Real& tolerance = 1.0e-5 )
{
   return ( -tolerance <= v && v <= tolerance );
}

__cuda_callable__
inline int roundUpDivision( const int num, const int div )
{
   return num / div + ( num % div != 0 );
}

__cuda_callable__
inline int roundToMultiple( int number, int multiple )
{
   return multiple*( number/ multiple + ( number % multiple != 0 ) );
}

__cuda_callable__
inline bool isPow2( int x )
{
   return ( ( x & ( x - 1 ) ) == 0 );
}

__cuda_callable__
inline bool isPow2( long int x )
{
   return ( ( x & ( x - 1 ) ) == 0 );
}

} // namespace TNL
