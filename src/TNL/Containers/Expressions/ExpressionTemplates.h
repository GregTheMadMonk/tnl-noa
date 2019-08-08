/***************************************************************************
                          ExpressionTemplates.h  -  description
                             -------------------
    begin                : Apr 18, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <ostream>
#include <utility>

#include <TNL/TypeTraits.h>
#include <TNL/Containers/Expressions/TypeTraits.h>
#include <TNL/Containers/Expressions/ExpressionVariableType.h>
#include <TNL/Containers/Expressions/Comparison.h>
#include <TNL/Containers/Expressions/HorizontalOperations.h>
#include <TNL/Containers/Expressions/VerticalOperations.h>

namespace TNL {
namespace Containers {
namespace Expressions {

////
// Non-static unary expression template
template< typename T1,
          template< typename > class Operation,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value >
struct UnaryExpressionTemplate
{};

template< typename T1,
          template< typename > class Operation,
          ExpressionVariableType T1Type >
struct IsExpressionTemplate< UnaryExpressionTemplate< T1, Operation, T1Type > >
: std::true_type
{};

////
// Non-static binary expression template
template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value,
          ExpressionVariableType T2Type = ExpressionVariableTypeGetter< T2 >::value >
struct BinaryExpressionTemplate
{};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          ExpressionVariableType T1Type,
          ExpressionVariableType T2Type >
struct IsExpressionTemplate< BinaryExpressionTemplate< T1, T2, Operation, T1Type, T2Type > >
: std::true_type
{};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct BinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, VectorExpressionVariable >
{
   using RealType = decltype( Operation< typename T1::RealType, typename T2::RealType >::
                              evaluate( std::declval<T1>()[0], std::declval<T2>()[0] ) );
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;

   static_assert( std::is_same< typename T1::DeviceType, typename T2::DeviceType >::value, "Attempt to mix operands allocated on different device types." );
   static_assert( IsStaticArrayType< T1 >::value == IsStaticArrayType< T2 >::value, "Attempt to mix static and non-static operands in binary expression templates." );

   BinaryExpressionTemplate( const T1& a, const T2& b )
   : op1( a ), op2( b ) {}

   RealType getElement( const IndexType i ) const
   {
      return Operation< typename T1::RealType, typename T2::RealType >::evaluate( op1.getElement( i ), op2.getElement( i ) );
   }

   __cuda_callable__
   RealType operator[]( const IndexType i ) const
   {
      return Operation< typename T1::RealType, typename T2::RealType >::evaluate( op1[ i ], op2[ i ] );
   }

   __cuda_callable__
   IndexType getSize() const
   {
      return op1.getSize();
   }

protected:
   const T1 op1;
   const T2 op2;
};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct BinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, ArithmeticVariable >
{
   using RealType = decltype( Operation< typename T1::RealType, T2 >::
                              evaluate( std::declval<T1>()[0], std::declval<T2>() ) );
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;

   BinaryExpressionTemplate( const T1& a, const T2& b )
   : op1( a ), op2( b ) {}

   RealType getElement( const IndexType i ) const
   {
      return Operation< typename T1::RealType, T2 >::evaluate( op1.getElement( i ), op2 );
   }

   __cuda_callable__
   RealType operator[]( const IndexType i ) const
   {
      return Operation< typename T1::RealType, T2 >::evaluate( op1[ i ], op2 );
   }

   __cuda_callable__
   IndexType getSize() const
   {
      return op1.getSize();
   }

protected:
   const T1 op1;
   const T2 op2;
};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct BinaryExpressionTemplate< T1, T2, Operation, ArithmeticVariable, VectorExpressionVariable >
{
   using RealType = decltype( Operation< T1, typename T2::RealType >::
                              evaluate( std::declval<T1>(), std::declval<T2>()[0] ) );
   using DeviceType = typename T2::DeviceType;
   using IndexType = typename T2::IndexType;

   BinaryExpressionTemplate( const T1& a, const T2& b )
   : op1( a ), op2( b ) {}

   RealType getElement( const IndexType i ) const
   {
      return Operation< T1, typename T2::RealType >::evaluate( op1, op2.getElement( i ) );
   }

   __cuda_callable__
   RealType operator[]( const IndexType i ) const
   {
      return Operation< T1, typename T2::RealType >::evaluate( op1, op2[ i ] );
   }

   __cuda_callable__
   IndexType getSize() const
   {
      return op2.getSize();
   }

protected:
   const T1 op1;
   const T2 op2;
};

////
// Non-static unary expression template
template< typename T1,
          template< typename > class Operation >
struct UnaryExpressionTemplate< T1, Operation, VectorExpressionVariable >
{
   using RealType = decltype( Operation< typename T1::RealType >::
                              evaluate( std::declval<T1>()[0] ) );
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;

   UnaryExpressionTemplate( const T1& a )
   : operand( a ) {}

   RealType getElement( const IndexType i ) const
   {
      return Operation< typename T1::RealType >::evaluate( operand.getElement( i ) );
   }

   __cuda_callable__
   RealType operator[]( const IndexType i ) const
   {
      return Operation< typename T1::RealType >::evaluate( operand[ i ] );
   }

   __cuda_callable__
   IndexType getSize() const
   {
      return operand.getSize();
   }

protected:
   const T1 operand;
};

////
// Output stream
template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
std::ostream& operator<<( std::ostream& str, const BinaryExpressionTemplate< T1, T2, Operation >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( expression.getSize() - 1 ) << " ]";
   return str;
}

template< typename T,
          template< typename > class Operation >
std::ostream& operator<<( std::ostream& str, const UnaryExpressionTemplate< T, Operation >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( expression.getSize() - 1 ) << " ]";
   return str;
}

////
// Operators are supposed to be in the same namespace as the expression templates

////
// Binary expressions addition
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator+( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
           const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator+( const BinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator+( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator+( const UnaryExpressionTemplate< T1, Operation >& a,
           const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator+( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
           const UnaryExpressionTemplate< T1, Operation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator+( const UnaryExpressionTemplate< L1, LOperation >& a,
           const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator+( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
           const UnaryExpressionTemplate< R1,ROperation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator+( const UnaryExpressionTemplate< L1,LOperation >& a,
           const UnaryExpressionTemplate< R1,ROperation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

////
// Binary expression subtraction
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator-( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
           const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator-( const BinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator-( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator-( const UnaryExpressionTemplate< T1, Operation >& a,
           const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator-( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
           const UnaryExpressionTemplate< T1, Operation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator-( const UnaryExpressionTemplate< L1, LOperation >& a,
           const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator-( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
           const UnaryExpressionTemplate< R1,ROperation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator-( const UnaryExpressionTemplate< L1,LOperation >& a,
           const UnaryExpressionTemplate< R1,ROperation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

////
// Binary expression multiplication
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator*( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
           const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator*( const BinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator*( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator*( const UnaryExpressionTemplate< T1, Operation >& a,
           const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator*( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
           const UnaryExpressionTemplate< T1, Operation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator*( const UnaryExpressionTemplate< L1, LOperation >& a,
           const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator*( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
           const UnaryExpressionTemplate< R1, ROperation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator*( const UnaryExpressionTemplate< L1,LOperation >& a,
           const UnaryExpressionTemplate< R1,ROperation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

////
// Binary expression division
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator/( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
           const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator/( const BinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator/( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator/( const UnaryExpressionTemplate< T1, Operation >& a,
           const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator/( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
           const UnaryExpressionTemplate< T1, Operation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator/( const UnaryExpressionTemplate< L1, LOperation >& a,
           const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator/( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
           const UnaryExpressionTemplate< R1,ROperation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator/( const UnaryExpressionTemplate< L1,LOperation >& a,
           const UnaryExpressionTemplate< R1,ROperation >& b )
{
   return BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
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
operator==( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
            const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator==( const BinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator==( const UnaryExpressionTemplate< T1, Operation >& a,
            const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator==( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator==( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
            const UnaryExpressionTemplate< T1, Operation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator==( const UnaryExpressionTemplate< L1, LOperation >& a,
            const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator==( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
            const UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
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
operator!=( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
            const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator!=( const BinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator!=( const UnaryExpressionTemplate< L1, LOperation >& a,
            const UnaryExpressionTemplate< R1, ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator!=( const UnaryExpressionTemplate< T1, Operation >& a,
            const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator!=( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator!=( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
            const UnaryExpressionTemplate< T1, Operation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator!=( const UnaryExpressionTemplate< L1, LOperation >& a,
            const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator!=( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
            const UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
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
operator<( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
           const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator<( const BinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator<( const UnaryExpressionTemplate< L1, LOperation >& a,
           const UnaryExpressionTemplate< R1, ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator<( const UnaryExpressionTemplate< T1, Operation >& a,
           const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator<( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator<( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
           const UnaryExpressionTemplate< T1, Operation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator<( const UnaryExpressionTemplate< L1, LOperation >& a,
           const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator<( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
           const UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
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
operator<=( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
            const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator<=( const BinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator<=( const UnaryExpressionTemplate< L1, LOperation >& a,
            const UnaryExpressionTemplate< R1, ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator<=( const UnaryExpressionTemplate< T1, Operation >& a,
            const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator<=( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator<=( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
            const UnaryExpressionTemplate< T1, Operation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator<=( const UnaryExpressionTemplate< L1, LOperation >& a,
            const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator<=( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
            const UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
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
operator>( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
           const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator>( const BinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator>( const UnaryExpressionTemplate< T1, Operation >& a,
           const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator>( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator>( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
           const UnaryExpressionTemplate< T1, Operation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator>( const UnaryExpressionTemplate< L1, LOperation >& a,
           const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator>( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
           const UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
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
operator>=( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
            const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator>=( const BinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator>=( const UnaryExpressionTemplate< T1, Operation >& a,
            const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator>=( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator>=( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
            const UnaryExpressionTemplate< T1, Operation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator>=( const UnaryExpressionTemplate< L1, LOperation >& a,
            const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator>=( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
            const UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Comparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

////
// Unary operations

////
// Minus
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
operator-( const BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return UnaryExpressionTemplate< std::decay_t<decltype(a)>, Minus >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
operator-( const UnaryExpressionTemplate< L1, LOperation >& a )
{
   return UnaryExpressionTemplate< std::decay_t<decltype(a)>, Minus >( a );
}

////
// Scalar product
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator,( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
           const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return ExpressionSum( a * b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator,( const UnaryExpressionTemplate< L1, LOperation >& a,
           const UnaryExpressionTemplate< R1, ROperation >& b )
{
   return ExpressionSum( a * b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator,( const UnaryExpressionTemplate< L1, LOperation >& a,
           const BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return ExpressionSum( a * b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator,( const BinaryExpressionTemplate< L1, L2, LOperation >& a,
           const UnaryExpressionTemplate< R1,ROperation >& b )
{
   return ExpressionSum( a * b );
}

} // namespace Expressions
} // namespace Containers

////
// All operations are supposed to be in namespace TNL

////
// Binary expression min
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
min( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
min( const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
min( const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
     const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
min( const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& a,
     const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
min( const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& a,
     const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
min( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
     const typename Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
min( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
min( const Containers::Expressions::UnaryExpressionTemplate< L1,LOperation >& a,
     const Containers::Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

////
// Binary expression max
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
max( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
max( const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
max( const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
     const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
max( const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& a,
     const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
max( const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& a,
     const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
max( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
     const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
max( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
max( const Containers::Expressions::UnaryExpressionTemplate< L1,LOperation >& a,
     const Containers::Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

////
// Abs
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
abs( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Abs >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
abs( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Abs >( a );
}

////
// Sin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
sin( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sin >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
sin( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sin >( a );
}

////
// Cos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
cos( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Cos >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
cos( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Cos >( a );
}

////
// Tan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
tan( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Tan >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
tan( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Tan >( a );
}

////
// Sqrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
sqrt( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sqrt >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
sqrt( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sqrt >( a );
}

////
// Cbrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
cbrt( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Cbrt >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
cbrt( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Cbrt >( a );
}

////
// Pow
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Real >
auto
pow( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a, const Real& exp )
{
   return Containers::Expressions::BinaryExpressionTemplate< std::decay_t<decltype(a)>, Real, Containers::Expressions::Pow >( a, exp );
}

template< typename L1,
          template< typename > class LOperation,
          typename Real >
auto
pow( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a, const Real& exp )
{
   return Containers::Expressions::BinaryExpressionTemplate< std::decay_t<decltype(a)>, Real, Containers::Expressions::Pow >( a, exp );
}

////
// Floor
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
floor( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Floor >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
floor( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Floor >( a );
}

////
// Ceil
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
ceil( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Ceil >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
ceil( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Ceil >( a );
}

////
// Asin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
asin( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Asin >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
asin( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Asin >( a );
}

////
// Acos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
acos( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Acos >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
acos( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Acos >( a );
}

////
// Atan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
atan( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Atan >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
atan( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Atan >( a );
}

////
// Sinh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
sinh( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sinh >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
sinh( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sinh >( a );
}

////
// Cosh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
cosh( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Cosh >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
cosh( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Cosh >( a );
}

////
// Tanh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
tanh( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Tanh >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
tanh( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Tanh >( a );
}

////
// Log
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
log( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Log >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
log( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Log >( a );
}

////
// Log10
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
log10( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Log10 >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
log10( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Log10 >( a );
}

////
// Log2
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
log2( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Log2 >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
log2( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Log2 >( a );
}

////
// Exp
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
exp( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Exp >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
exp( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Exp >( a );
}

////
// Vertical operations - min
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
min( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionMin( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
min( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return ExpressionMin( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Index >
auto
argMin( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a, Index& arg )
{
   return ExpressionArgMin( a, arg );
}

template< typename L1,
          template< typename > class LOperation,
          typename Index >
auto
argMin( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a, Index& arg )
{
   return ExpressionArgMin( a, arg );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
max( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionMax( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
max( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return ExpressionMax( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Index >
auto
argMax( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a, Index& arg )
{
   return ExpressionArgMax( a, arg );
}

template< typename L1,
          template< typename > class LOperation,
          typename Index >
auto
argMax( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a, Index& arg )
{
   return ExpressionArgMax( a, arg );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
sum( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionSum( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
sum( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return ExpressionSum( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
maxNorm( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return max( abs( a ) );
}

template< typename L1,
          template< typename > class LOperation >
auto
maxNorm( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return max( abs( a ) );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
l1Norm( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionL1Norm( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
l1Norm( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return ExpressionL1Norm( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
l2Norm( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return TNL::sqrt( ExpressionL2Norm( a ) );
}

template< typename L1,
          template< typename > class LOperation >
auto
l2Norm( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return TNL::sqrt( ExpressionL2Norm( a ) );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Real >
auto
lpNorm( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a, const Real& p )
// since (1.0 / p) has type double, TNL::pow returns double
-> double
{
   if( p == 1.0 )
      return l1Norm( a );
   if( p == 2.0 )
      return l2Norm( a );
   return TNL::pow( ExpressionLpNorm( a, p ), 1.0 / p );
}

template< typename L1,
          template< typename > class LOperation,
          typename Real >
auto
lpNorm( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a, const Real& p )
// since (1.0 / p) has type double, TNL::pow returns double
-> double
{
   if( p == 1.0 )
      return l1Norm( a );
   if( p == 2.0 )
      return l2Norm( a );
   return TNL::pow( ExpressionLpNorm( a, p ), 1.0 / p );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
product( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionProduct( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
product( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return ExpressionProduct( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
logicalOr( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionLogicalOr( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
logicalOr( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return ExpressionLogicalOr( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
logicalAnd( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionLogicalAnd( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
logicalAnd( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return ExpressionLogicalAnd( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
binaryOr( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionBinaryOr( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
binaryOr( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return ExpressionBinaryOr( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
binaryAnd( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionBinaryAnd( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
binaryAnd( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return ExpressionBinaryAnd( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
dot( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return (a, b);
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
dot( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
     const Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >& b )
{
   return (a, b);
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
dot( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
     const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return (a, b);
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
dot( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >& b )
{
   return (a, b);
}


////
// Evaluation with reduction
template< typename Vector,
   typename T1,
   typename T2,
   template< typename, typename > class Operation,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
Result evaluateAndReduce( Vector& lhs,
   const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType { return ( lhs_data[ i ] = expression[ i ] ); };
   return Containers::Algorithms::Reduction< DeviceType >::reduce( lhs.getSize(), reduction, volatileReduction, fetch, zero );
}

template< typename Vector,
   typename T1,
   template< typename > class Operation,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
Result evaluateAndReduce( Vector& lhs,
   const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType { return ( lhs_data[ i ] = expression[ i ] ); };
   return Containers::Algorithms::Reduction< DeviceType >::reduce( lhs.getSize(), reduction, volatileReduction, fetch, zero );
}

////
// Addition and reduction
template< typename Vector,
   typename T1,
   typename T2,
   template< typename, typename > class Operation,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
Result addAndReduce( Vector& lhs,
   const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType {
      const RealType aux = expression[ i ];
      lhs_data[ i ] += aux;
      return aux;
   };
   return Containers::Algorithms::Reduction< DeviceType >::reduce( lhs.getSize(), reduction, volatileReduction, fetch, zero );
}

template< typename Vector,
   typename T1,
   template< typename > class Operation,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
Result addAndReduce( Vector& lhs,
   const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType {
      const RealType aux = expression[ i ];
      lhs_data[ i ] += aux;
      return aux;
   };
   return Containers::Algorithms::Reduction< DeviceType >::reduce( lhs.getSize(), reduction, volatileReduction, fetch, zero );
}

////
// Addition and reduction
template< typename Vector,
   typename T1,
   typename T2,
   template< typename, typename > class Operation,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
Result addAndReduceAbs( Vector& lhs,
   const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType {
      const RealType aux = expression[ i ];
      lhs_data[ i ] += aux;
      return TNL::abs( aux );
   };
   return Containers::Algorithms::Reduction< DeviceType >::reduce( lhs.getSize(), reduction, volatileReduction, fetch, zero );
}

template< typename Vector,
   typename T1,
   template< typename > class Operation,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
Result addAndReduceAbs( Vector& lhs,
   const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType {
      const RealType aux = expression[ i ];
      lhs_data[ i ] += aux;
      return TNL::abs( aux );
   };
   return Containers::Algorithms::Reduction< DeviceType >::reduce( lhs.getSize(), reduction, volatileReduction, fetch, zero );
}

} // namespace TNL
