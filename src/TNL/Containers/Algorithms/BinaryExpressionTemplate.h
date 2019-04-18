/***************************************************************************
                          BinaryExpressionTemplate.h  -  description
                             -------------------
    begin                : Apr 18, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Algorithms/ExpressionTemplatesOperations.h>
#include <TNL/Containers/Algorithms/ExpressionVariableType.h>

namespace TNL {
   namespace Containers {
      namespace Algorithms {

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value,
          ExpressionVariableType T2Type = ExpressionVariableTypeGetter< T2 >::value >
struct BinaryExpressionTemplate
{
   BinaryExpressionTemplate( const T1& a, const T2& b ){};

   static T1 evaluate( const T1& a, const T2& b )
   {
      return Operation< T1, T2 >::evaluate( a, b );
   }
};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct BinaryExpressionTemplate< T1, T2, Operation, VectorVariable, VectorVariable >
{
   using RealType = typename T1::RealType;
   using IsExpressionTemplate = bool;

   BinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   static BinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return BinaryExpressionTemplate( a, b );
   }

   RealType operator[]( const int i ) const
   {
       return Operation< typename T1::RealType, typename T2::RealType >::evaluate( op1[ i ], op2[ i ] );
   }

   int getSize() const
   {
       return op1.getSize();
   }

   protected:
      const T1 &op1;
      const T2 &op2;

};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct BinaryExpressionTemplate< T1, T2, Operation, VectorVariable, ArithmeticVariable  >
{
   using RealType = typename T1::RealType;
   using IsExpressionTemplate = bool;

   BinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   BinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return BinaryExpressionTemplate( a, b );
   }

   RealType operator[]( const int i ) const
   {
       return Operation< typename T1::RealType, T2 >::evaluate( op1[ i ], op2 );
   }

   int getSize() const
   {
       return op1.getSize();
   }

   protected:
      const T1 &op1;
      const T2 &op2;

};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct BinaryExpressionTemplate< T1, T2, Operation, ArithmeticVariable, VectorVariable  >
{
   using RealType = typename T2::RealType;
   using IsExpressionTemplate = bool;

   BinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   BinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return BinaryExpressionTemplate( a, b );
   }

   RealType operator[]( const int i ) const
   {
       return Operation< T1, typename T2::RealType >::evaluate( op1, op2[ i ] );
   }

   int getSize() const
   {
       return op2.getSize();
   }

   protected:
      const T1& op1;
      const T2& op2;

};

/*
template< typename T1, typename T2 >
BinaryExpressionTemplate< T1, T2, ExpressionTemplates::Addition > operator+( const T1 &a, const T2 &b )
{
    return BinaryExpressionTemplate< T1, T2, ExpressionTemplates::Addition >( a, b );
}

template< typename T1, typename T2 >
BinaryExpressionTemplate< T1, T2, ExpressionTemplates::Subtraction > operator-( const T1 &a, const T2 &b )
{
    return BinaryExpressionTemplate< T1, T2, ExpressionTemplates::Subtraction >( a, b );
}

template< typename T1, typename T2 >
BinaryExpressionTemplate< T1, T2, ExpressionTemplates::Multiplication > operator*( const T1 &a, const T2 &b )
{
    return BinaryExpressionTemplate< T1, T2, ExpressionTemplates::Multiplication >( a, b );
}

template< typename T1, typename T2 >
BinaryExpressionTemplate< T1, T2, ExpressionTemplates::Division > operator/( const T1 &a, const T2 &b )
{
    return BinaryExpressionTemplate< T1, T2, ExpressionTemplates::Division >( a, b );
}
 */
      } //namespace Algorithms
   } //namespace Containers
} // namespace TNL