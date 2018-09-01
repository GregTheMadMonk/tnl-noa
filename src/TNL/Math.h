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

/***
 * This function returns minimum of two numbers.
 * GPU device code uses the functions defined in the CUDA's math_functions.h,
 * MIC uses trivial override and host uses the STL functions.
 */
template< typename T1, typename T2, typename ResultType = typename std::common_type< T1, T2 >::type >
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
template< typename T1, typename T2, typename ResultType = typename std::common_type< T1, T2 >::type >
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

/***
 * This function returns argument of minimum of two numbers.
 */
template< typename T1, typename T2, typename ResultType = typename std::common_type< T1, T2 >::type >
__cuda_callable__ inline
ResultType argMin( const T1& a, const T2& b )
{
   return ( a < b ) ?  a : b;
}

/***
 * This function returns argument of maximum of two numbers.
 */
template< typename T1, typename T2, typename ResultType = typename std::common_type< T1, T2 >::type >
__cuda_callable__
ResultType argMax( const T1& a, const T2& b )
{
   return ( a > b ) ?  a : b;   
}

/***
 * This function returns argument of minimum of absolute values of two numbers.
 */
template< typename T1, typename T2, typename ResultType = typename std::common_type< T1, T2 >::type >
__cuda_callable__ inline
ResultType argAbsMin( const T1& a, const T2& b )
{
   return ( TNL::abs( a ) < TNL::abs( b ) ) ?  a : b;
}

/***
 * This function returns argument of maximum of absolute values of two numbers.
 */
template< typename T1, typename T2, typename ResultType = typename std::common_type< T1, T2 >::type >
__cuda_callable__
ResultType argAbsMax( const T1& a, const T2& b )
{
   return ( TNL::abs( a ) > TNL::abs( b ) ) ?  a : b;   
}

template< typename T1, typename T2, typename ResultType = typename std::common_type< T1, T2 >::type >
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
