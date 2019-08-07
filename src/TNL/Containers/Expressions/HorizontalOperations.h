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
   static auto evaluate( const T1& a, const T2& b ) -> decltype( TNL::min( a , b ) )
   {
      return TNL::min( a, b );
   }
};

template< typename T1, typename T2 >
struct Max
{
   __cuda_callable__
   static auto evaluate( const T1& a, const T2& b ) -> decltype( TNL::max( a, b ) )
   {
      return TNL::max( a, b );
   }
};

template< typename T1 >
struct Minus
{
   __cuda_callable__
   static T1 evaluate( const T1& a )
   {
      return -a;
   }
};

template< typename T1 >
struct Abs
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::abs( a ) )
   {
      return TNL::abs( a );
   }
};

template< typename T1, typename T2 >
struct Pow
{
   __cuda_callable__
   static auto evaluate( const T1& a, const T2& exp ) -> decltype( TNL::pow( a, exp ) )
   {
      return TNL::pow( a, exp );
   }
};

template< typename T1 >
struct Exp
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::exp( a ) )
   {
      return TNL::exp( a );
   }
};

template< typename T1 >
struct Sqrt
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::sqrt( a ) )
   {
      return TNL::sqrt( a );
   }
};

template< typename T1 >
struct Cbrt
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::cbrt( a ) )
   {
      return TNL::cbrt( a );
   }
};

template< typename T1 >
struct Log
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::log( a ) )
   {
      return TNL::log( a );
   }
};

template< typename T1 >
struct Log10
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::log10( a ) )
   {
      return TNL::log10( a );
   }
};

template< typename T1 >
struct Log2
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::log2( a ) )
   {
      return TNL::log2( a );
   }
};

template< typename T1 >
struct Sin
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::sin( a ) )
   {
      return TNL::sin( a );
   }
};

template< typename T1 >
struct Cos
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::cos( a ) )
   {
      return TNL::cos( a );
   }
};

template< typename T1 >
struct Tan
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::tan( a ) )
   {
      return TNL::tan( a );
   }
};

template< typename T1 >
struct Asin
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::asin( a ) )
   {
      return TNL::asin( a );
   }
};

template< typename T1 >
struct Acos
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::acos( a ) )
   {
      return TNL::acos( a );
   }
};

template< typename T1 >
struct Atan
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::atan( a ) )
   {
      return TNL::atan( a );
   }
};

template< typename T1 >
struct Sinh
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::sinh( a ) )
   {
      return TNL::sinh( a );
   }
};

template< typename T1 >
struct Cosh
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::cosh( a ) )
   {
      return TNL::cosh( a );
   }
};

template< typename T1 >
struct Tanh
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::tanh( a ) )
   {
      return TNL::tanh( a );
   }
};

template< typename T1 >
struct Asinh
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::asinh( a ) )
   {
      return TNL::asinh( a );
   }
};

template< typename T1 >
struct Acosh
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::acosh( a ) )
   {
      return TNL::acosh( a );
   }
};

template< typename T1 >
struct Atanh
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::atanh( a ) )
   {
      return TNL::atanh( a );
   }
};

template< typename T1 >
struct Floor
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::floor( a ) )
   {
      return TNL::floor( a );
   }
};

template< typename T1 >
struct Ceil
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::ceil( a ) )
   {
      return TNL::ceil( a );
   }
};

template< typename T1 >
struct Sign
{
   __cuda_callable__
   static auto evaluate( const T1& a ) -> decltype( TNL::sign( a ) )
   {
      return TNL::sign( a );
   }
};

} // namespace Expressions
} // namespace Containers
} // namespace TNL
