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

/***
 * This function returns minimum of two numbers.
 * Specializations use the functions defined in the CUDA's math_functions.h
 * in CUDA device code and STL functions otherwise.
 */
template< typename Type1, typename Type2 >
__cuda_callable__ inline
Type1 min( const Type1& a, const Type2& b )
{
   return a < b ? a : b;
};

// specialization for int
template< class T >
__cuda_callable__ inline
typename enable_if_same_base< T, int >::type
min( const T& a, const T& b )
{
#ifdef __CUDA_ARCH__
   return ::min( a, b );
#else
   //return std::min( a, b );
   return !(b<a)?a:b;
#endif
}

// specialization for float
template< class T >
__cuda_callable__ inline
typename enable_if_same_base< T, float >::type
min( const T& a, const T& b )
{
#ifdef __CUDA_ARCH__
   return ::fminf( a, b );
#else
   return std::fmin( a, b );
#endif
}

// specialization for double
template< class T >
__cuda_callable__ inline
typename enable_if_same_base< T, double >::type
min( const T& a, const T& b )
{
#ifdef __CUDA_ARCH__
   return ::fmin( a, b );
#else
   return std::fmin( a, b );
#endif
}


/***
 * This function returns maximum of two numbers.
 * Specializations use the functions defined in the CUDA's math_functions.h
 * in CUDA device code and STL functions otherwise.
 */
template< typename Type1, typename Type2 >
__cuda_callable__
Type1 max( const Type1& a, const Type2& b )
{
   return a > b ? a : b;
};

// specialization for int
template< class T >
__cuda_callable__ inline
typename enable_if_same_base< T, int >::type
max( const T& a, const T& b )
{
#ifdef __CUDA_ARCH__
   
#else
    #ifdef __MIC__
       return ::max( a, b );
    #else
       return std::max( a, b );
    #endif
#endif
}

// specialization for float
template< class T >
__cuda_callable__ inline
typename enable_if_same_base< T, float >::type
max( const T& a, const T& b )
{
#ifdef __CUDA_ARCH__
   return ::fmaxf( a, b );
#else
   return std::fmax( a, b );
#endif
}

// specialization for double
template< class T >
__cuda_callable__ inline
typename enable_if_same_base< T, double >::type
max( const T& a, const T& b )
{
#ifdef __CUDA_ARCH__
   return ::fmax( a, b );
#else
   return std::fmax( a, b );
#endif
}


/***
 * This function returns absolute value of given number.
 * Specializations use the functions defined in the CUDA's math_functions.h
 * in CUDA device code and STL functions otherwise.
 */
template< class T >
__cuda_callable__ inline
typename std::enable_if< ! std::is_arithmetic< T >::value, T >::type
abs( const T& n )
{
   if( n < ( T ) 0 )
      return -n;
   return n;
}

// specialization for any arithmetic type (e.g. int, float, double)
template< class T >
__cuda_callable__ inline
typename std::enable_if< std::is_arithmetic< T >::value, T >::type
abs( const T& n )
{
#ifdef __CUDA_ARCH__
   return ::abs( n );
#else
   /*return std::abs( n );*/
   if(n>=0)
       return n;
   else
       return -n;
#endif
}


template< class T >
__cuda_callable__ inline
T pow( const T& base, const T& exp )
{
#ifdef __CUDA_ARCH__
   return ::pow( base, exp );
#else
   return std::pow( base, exp );
#endif
}


template< typename Type >
__cuda_callable__
void swap( Type& a, Type& b )
{
   Type tmp( a );
   a = b;
   b = tmp;
};

template< class T >
__cuda_callable__
T sign( const T& a )
{
   if( a < ( T ) 0 ) return ( T ) -1;
   if( a == ( T ) 0 ) return ( T ) 0;
   return ( T ) 1;
};

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

