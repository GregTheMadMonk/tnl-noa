/***************************************************************************
                          BinaryExpressionTemplate.h  -  description
                             -------------------
    begin                : Apr 18, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>
#include <TNL/Containers/Expressions/ExpressionTemplatesOperations.h>
#include <TNL/Containers/Expressions/ExpressionVariableType.h>
#include <TNL/Containers/Expressions/StaticComparison.h>

namespace TNL {
   namespace Containers {
      namespace Expressions {

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value,
          ExpressionVariableType T2Type = ExpressionVariableTypeGetter< T2 >::value >
struct BinaryExpressionTemplate
{
   /*BinaryExpressionTemplate( const T1& a, const T2& b ){};

   static T1 evaluate( const T1& a, const T2& b )
   {
      return Operation< T1, T2 >::evaluate( a, b );
   }*/
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

////
// Binary expressions addition
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Addition >
operator + ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate< 
      Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Expressions::Addition >
operator + ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::BinaryExpressionTemplate< 
      Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Expressions::Addition >( a, b );
}

////
// Binary expression subtraction
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Subtraction >
operator - ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate< 
      Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Expressions::Subtraction >
operator - ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::BinaryExpressionTemplate< 
      Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Expressions::Subtraction >( a, b );
}

////
// Binary expression multiplication
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Multiplication >
operator * ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate< 
      Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Expressions::Multiplication >
operator * ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::BinaryExpressionTemplate< 
      Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Expressions::Multiplication >( a, b );
}

////
// Binary expression division
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Division >
operator / ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate< 
      Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Expressions::Division >
operator / ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::BinaryExpressionTemplate< 
      Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Expressions::Division >( a, b );
}

////
// Comparison operator ==
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator == ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticComparisonEQ( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator == ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::StaticComparisonEQ( a, b );
}

////
// Comparison operator !=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator != ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticComparisonNE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator != ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::StaticComparisonNE( a, b );
}

////
// Comparison operator <
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator < ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticComparisonLT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator < ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::StaticComparisonLT( a, b );
}

////
// Comparison operator <=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator <= ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticComparisonLE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator <= ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::StaticComparisonLE( a, b );
}

////
// Comparison operator >
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator > ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticComparisonGT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator > ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::StaticComparisonGT( a, b );
}

////
// Comparison operator >=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator >= ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
              const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticComparisonGE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator >= ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::StaticComparisonGE( a, b );
}

////
// Output stream
template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
std::ostream& operator << ( std::ostream& str, const BinaryExpressionTemplate< T1, T2, Operation >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression[ i ] << " ";
   str << expression[ expression.getSize() - 1 ] << " ]";
   return str;
}

      } //namespace Expressions
   } //namespace Containers
} // namespace TNL
