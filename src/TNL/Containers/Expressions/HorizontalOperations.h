/***************************************************************************
                          HorizontalOperations.h  -  description
                             -------------------
    begin                : Apr 18, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Math.h>

namespace TNL {
namespace Containers {
namespace Expressions {

template< typename T1, typename T2 >
struct Addition
{
   __cuda_callable__
   static auto evaluate( const T1& a, const T2& b ) -> decltype( a + b )
   {
      return a + b;
   }
};

template< typename T1, typename T2 >
struct Subtraction
{
   __cuda_callable__
   static auto evaluate( const T1& a, const T2& b ) -> decltype( a - b )
   {
      return a - b;
   }
};

template< typename T1, typename T2 >
struct Multiplication
{
   __cuda_callable__
   static auto evaluate( const T1& a, const T2& b ) -> decltype( a * b )
   {
      return a * b;
   }
};

template< typename T1, typename T2 >
struct Division
{
   __cuda_callable__
   static auto evaluate( const T1& a, const T2& b ) -> decltype( a / b )
   {
      return a / b;
   }
};

template< typename T1, typename T2 >
struct Min
{
   __cuda_callable__
   static auto evaluate( const T1& a, const T2& b ) -> decltype( min( a , b ) )
   {
      return min( a, b );
   }
};

template< typename T1, typename T2 >
struct Max
{
   __cuda_callable__
   static auto evaluate( const T1& a, const T2& b ) -> decltype( max( a, b ) )
   {
      return max( a, b );
   }
};

template< typename T1 >
struct Minus
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( -a )
   {
      return -a;
   }
};

template< typename T1 >
struct Abs
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( abs( a ) )
   {
      return abs( a );
   }
};

template< typename T1, typename T2 >
struct Pow
{
   __cuda_callable__
   static auto evaluate( const T1& a, const T2& exp ) -> decltype( pow( a, exp ) )
   {
      return pow( a, exp );
   }
};

template< typename T1 >
struct Exp
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( exp( a ) )
   {
      return exp( a );
   }
};

template< typename T1 >
struct Sqrt
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( sqrt( a ) )
   {
      return sqrt( a );
   }
};

template< typename T1 >
struct Cbrt
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( cbrt( a ) )
   {
      return cbrt( a );
   }
};

template< typename T1 >
struct Log
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( log( a ) )
   {
      return log( a );
   }
};

template< typename T1 >
struct Log10
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( log10( a ) )
   {
      return log10( a );
   }
};

template< typename T1 >
struct Log2
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( log2( a ) )
   {
      return log2( a );
   }
};

template< typename T1 >
struct Sin
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( sin( a ) )
   {
      return sin( a );
   }
};

template< typename T1 >
struct Cos
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( cos( a ) )
   {
      return cos( a );
   }
};

template< typename T1 >
struct Tan
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( tan( a ) )
   {
      return tan( a );
   }
};

template< typename T1 >
struct Asin
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( asin( a ) )
   {
      return asin( a );
   }
};

template< typename T1 >
struct Acos
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( acos( a ) )
   {
      return acos( a );
   }
};

template< typename T1 >
struct Atan
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( atan( a ) )
   {
      return atan( a );
   }
};

template< typename T1 >
struct Sinh
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( sinh( a ) )
   {
      return sinh( a );
   }
};

template< typename T1 >
struct Cosh
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( cosh( a ) )
   {
      return cosh( a );
   }
};

template< typename T1 >
struct Tanh
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( tanh( a ) )
   {
      return tanh( a );
   }
};

template< typename T1 >
struct Asinh
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( asinh( a ) )
   {
      return asinh( a );
   }
};

template< typename T1 >
struct Acosh
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( acosh( a ) )
   {
      return acosh( a );
   }
};

template< typename T1 >
struct Atanh
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( atanh( a ) )
   {
      return atanh( a );
   }
};

template< typename T1 >
struct Floor
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( floor( a ) )
   {
      return floor( a );
   }
};

template< typename T1 >
struct Ceil
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( ceil( a ) )
   {
      return ceil( a );
   }
};

template< typename T1 >
struct Sign
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( sign( a ) )
   {
      return sign( a );
   }
};

template< typename ResultType >
struct Cast
{
   template< typename T1 >
   struct Operation
   {
      __cuda_callable__
      static auto evaluate( const T1& a ) -> ResultType
      {
         return static_cast<ResultType>( a );
      }
   };
};

} // namespace Expressions
} // namespace Containers
} // namespace TNL
