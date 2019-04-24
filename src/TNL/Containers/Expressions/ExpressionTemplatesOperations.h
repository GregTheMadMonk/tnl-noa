/***************************************************************************
                          ExpressionTemplatesOperations.h  -  description
                             -------------------
    begin                : Apr 18, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

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
      } //namespace Expressions
   } // namespace Containers
} // namespace TNL