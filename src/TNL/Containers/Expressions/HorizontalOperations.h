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

struct Addition
{
   template< typename T1, typename T2 >
   __cuda_callable__
   static auto evaluate( const T1& a, const T2& b ) -> decltype( a + b )
   {
      return a + b;
   }
};

struct Subtraction
{
   template< typename T1, typename T2 >
   __cuda_callable__
   static auto evaluate( const T1& a, const T2& b ) -> decltype( a - b )
   {
      return a - b;
   }
};

struct Multiplication
{
   template< typename T1, typename T2 >
   __cuda_callable__
   static auto evaluate( const T1& a, const T2& b ) -> decltype( a * b )
   {
      return a * b;
   }
};

struct Division
{
   template< typename T1, typename T2 >
   __cuda_callable__
   static auto evaluate( const T1& a, const T2& b ) -> decltype( a / b )
   {
      return a / b;
   }
};

struct Min
{
   template< typename T1, typename T2 >
   __cuda_callable__
   static auto evaluate( const T1& a, const T2& b ) -> decltype( min( a, b ) )
   {
      return min( a, b );
   }
};

struct Max
{
   template< typename T1, typename T2 >
   __cuda_callable__
   static auto evaluate( const T1& a, const T2& b ) -> decltype( max( a, b ) )
   {
      return max( a, b );
   }
};

struct UnaryPlus
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( +a )
   {
      return +a;
   }
};

struct UnaryMinus
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( -a )
   {
      return -a;
   }
};

struct Abs
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( abs( a ) )
   {
      return abs( a );
   }
};

struct Pow
{
   template< typename T1, typename T2 >
   __cuda_callable__
   static auto evaluate( const T1& a, const T2& exp ) -> decltype( pow( a, exp ) )
   {
      return pow( a, exp );
   }
};

struct Exp
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( exp( a ) )
   {
      return exp( a );
   }
};

struct Sqrt
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( sqrt( a ) )
   {
      return sqrt( a );
   }
};

struct Cbrt
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( cbrt( a ) )
   {
      return cbrt( a );
   }
};

struct Log
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( log( a ) )
   {
      return log( a );
   }
};

struct Log10
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( log10( a ) )
   {
      return log10( a );
   }
};

struct Log2
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( log2( a ) )
   {
      return log2( a );
   }
};

struct Sin
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( sin( a ) )
   {
      return sin( a );
   }
};

struct Cos
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( cos( a ) )
   {
      return cos( a );
   }
};

struct Tan
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( tan( a ) )
   {
      return tan( a );
   }
};

struct Asin
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( asin( a ) )
   {
      return asin( a );
   }
};

struct Acos
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( acos( a ) )
   {
      return acos( a );
   }
};

struct Atan
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( atan( a ) )
   {
      return atan( a );
   }
};

struct Sinh
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( sinh( a ) )
   {
      return sinh( a );
   }
};

struct Cosh
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( cosh( a ) )
   {
      return cosh( a );
   }
};

struct Tanh
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( tanh( a ) )
   {
      return tanh( a );
   }
};

struct Asinh
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( asinh( a ) )
   {
      return asinh( a );
   }
};

struct Acosh
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( acosh( a ) )
   {
      return acosh( a );
   }
};

struct Atanh
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( atanh( a ) )
   {
      return atanh( a );
   }
};

struct Floor
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( floor( a ) )
   {
      return floor( a );
   }
};

struct Ceil
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( ceil( a ) )
   {
      return ceil( a );
   }
};

struct Sign
{
   template< typename T1 >
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( sign( a ) )
   {
      return sign( a );
   }
};

template< typename ResultType >
struct Cast
{
   struct Operation
   {
      template< typename T1 >
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
