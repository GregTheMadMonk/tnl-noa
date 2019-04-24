/***************************************************************************
                          UnaryExpressionTemplate.h  -  description
                             -------------------
    begin                : Apr 24, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>

namespace TNL {
   namespace Containers {
      namespace Expressions {


template< typename T1,
          template< typename, typename > class Operation,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value >
struct UnaryExpressionTemplate
{
};

template< typename T1,
          template< typename, typename > class Operation >
struct UnaryExpressionTemplate< T1, Operation, VectorVariable >
{
   using RealType = typename T1::RealType;
   using IsExpressionTemplate = bool;

   __cuda_callable__
   UnaryExpressionTemplate( const T1& a ): operand( a ){}

   __cuda_callable__
   static UnaryExpressionTemplate evaluate( const T1& a )
   {
      return UnaryExpressionTemplate( a );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
       return Operation< typename T1::RealType, typename T2::RealType >::evaluate( operand[ i ] );
   }

   __cuda_callable__
   int getSize() const
   {
       return operand.getSize();
   }

   protected:
      const T1 &operand;
};

////
// Abs
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Abs >
abs( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate< 
      Expressions::BinaryExpressionTemplate< L1, L2, ROperation >,
      Expressions::Abs >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Abs >
abs( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate< 
      Expressions::UnaryExpressionTemplate< L1, L2, ROperation >,
      Expressions::Abs >( a );
}



         
      } //namespace Expressions
   } //namespace Containers
} // namespace TNL