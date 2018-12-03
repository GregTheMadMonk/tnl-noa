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

/**
 * \brief This function returns minimum of two numbers.
 *
 * GPU device code uses the functions defined in the CUDA's math_functions.h,
 * MIC uses trivial override and host uses the STL functions.
 */
template< typename T1, typename T2, typename ResultType = typename std::common_type< T1, T2 >::type >
__cuda_callable__ inline
ResultType min( const T1& a, const T2& b )
{
#if __cplusplus >= 201402L
   // std::min is constexpr since C++14 so it can be reused directly
   return std::min( (ResultType) a, (ResultType) b );
#else
 #if defined(__CUDA_ARCH__)
   return ::min( (ResultType) a, (ResultType) b );
 #elif defined(__MIC__)
   return a < b ? a : b;
 #else
   return std::min( (ResultType) a, (ResultType) b );
 #endif
#endif
}


/**
 * \brief This function returns maximum of two numbers.
 *
 * GPU device code uses the functions defined in the CUDA's math_functions.h,
 * MIC uses trivial override and host uses the STL functions.
 */
template< typename T1, typename T2, typename ResultType = typename std::common_type< T1, T2 >::type >
__cuda_callable__
ResultType max( const T1& a, const T2& b )
{
#if __cplusplus >= 201402L
   // std::max is constexpr since C++14 so it can be reused directly
   return std::max( (ResultType) a, (ResultType) b );
#else
 #if defined(__CUDA_ARCH__)
   return ::max( (ResultType) a, (ResultType) b );
 #elif defined(__MIC__)
   return a > b ? a : b;
 #else
   return std::max( (ResultType) a, (ResultType) b );
 #endif
#endif
}

/**
 * \brief This function returns absolute value of given number \e n.
 */
template< class T >
__cuda_callable__ inline
T abs( const T& n )
{
#if defined(__CUDA_ARCH__)
   if( std::is_integral< T >::value )
      return ::abs( n );
   else
      return ::fabs( n );
#elif defined(__MIC__)
   if( n < ( T ) 0 )
      return -n;
   return n;
#else
   return std::abs( n );
#endif
}

/***
 * \brief This function returns argument of minimum of two numbers.
 */
template< typename T1, typename T2, typename ResultType = typename std::common_type< T1, T2 >::type >
__cuda_callable__ inline
ResultType argMin( const T1& a, const T2& b )
{
   return ( a < b ) ?  a : b;
}

/***
 * \brief This function returns argument of maximum of two numbers.
 */
template< typename T1, typename T2, typename ResultType = typename std::common_type< T1, T2 >::type >
__cuda_callable__
ResultType argMax( const T1& a, const T2& b )
{
   return ( a > b ) ?  a : b;   
}

/***
 * \brief This function returns argument of minimum of absolute values of two numbers.
 */
template< typename T1, typename T2, typename ResultType = typename std::common_type< T1, T2 >::type >
__cuda_callable__ inline
ResultType argAbsMin( const T1& a, const T2& b )
{
   return ( TNL::abs( a ) < TNL::abs( b ) ) ?  a : b;
}

/***
 * \brief This function returns argument of maximum of absolute values of two numbers.
 */
template< typename T1, typename T2, typename ResultType = typename std::common_type< T1, T2 >::type >
__cuda_callable__
ResultType argAbsMax( const T1& a, const T2& b )
{
   return ( TNL::abs( a ) > TNL::abs( b ) ) ?  a : b;   
}

/**
 * \brief This function returns the result of \e base to the power of \e exp.
 */
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

/**
 * \brief This function returns square root of the given \e value.
 */
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

/**
 * \brief This function swaps values of two parameters.
 *
 * It assigns the value of \e a to the parameter \e b and vice versa.
 */
template< typename Type >
__cuda_callable__
void swap( Type& a, Type& b )
{
   Type tmp( a );
   a = b;
   b = tmp;
}

/**
 * \brief This function represents the signum function.
 *
 * It extracts the sign of the number \e a. In other words, the signum function projects
 * negative numbers to value -1, positive numbers to value 1 and zero to value 0.
 * Non-zero complex numbers are projected to the unit circle.
 */
template< class T >
__cuda_callable__
T sign( const T& a )
{
   if( a < ( T ) 0 ) return ( T ) -1;
   if( a == ( T ) 0 ) return ( T ) 0;
   return ( T ) 1;
}

/**
 * \brief This function tests whether the given real number is small.
 *
 * It tests whether the number \e v is in \e tolerance, in other words, whether
 * \e v in absolute value is less then or equal to \e tolerance.
 * \param v Real number.
 * \param tolerance Critical value which is set to 0.00001 by defalt.
 */
template< typename Real >
__cuda_callable__
bool isSmall( const Real& v,
              const Real& tolerance = 1.0e-5 )
{
   return ( -tolerance <= v && v <= tolerance );
}

/**
 * \brief This function divides \e num by \e div and rounds up the result.
 *
 * \param num An integer considered as dividend.
 * \param div An integer considered as divisor.
 */
__cuda_callable__
inline int roundUpDivision( const int num, const int div )
{
   return num / div + ( num % div != 0 );
}

/**
 * \brief This function rounds up \e number to the nearest multiple of number \e multiple.
 *
 * \param number Integer we want to round.
 * \param multiple Integer.
 */
__cuda_callable__
inline int roundToMultiple( int number, int multiple )
{
   return multiple*( number/ multiple + ( number % multiple != 0 ) );
}

/**
 * \brief This function checks if \e x is an integral power of two.
 *
 * Returns \e true if \e x is a power of two. Otherwise returns \e false.
 * \param x Integer.
 */
__cuda_callable__
inline bool isPow2( int x )
{
   return ( ( x & ( x - 1 ) ) == 0 );
}

/**
 * \brief This function checks if \e x is an integral power of two.
 *
 * Returns \e true if \e x is a power of two. Otherwise returns \e false.
 * \param x Long integer.
 */
__cuda_callable__
inline bool isPow2( long int x )
{
   return ( ( x & ( x - 1 ) ) == 0 );
}

} // namespace TNL
