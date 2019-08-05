/***************************************************************************
                          ExpressionTemplates.h  -  description
                             -------------------
    begin                : Apr 18, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>

#include <TNL/TypeTraits.h>
#include <TNL/Containers/Expressions/TypeTraits.h>
#include <TNL/Containers/Expressions/ExpressionTemplatesOperations.h>
#include <TNL/Containers/Expressions/ExpressionVariableType.h>
#include <TNL/Containers/Expressions/Comparison.h>

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
   using RealType = typename T1::RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;

   static_assert( std::is_same< typename T1::DeviceType, typename T2::DeviceType >::value, "Attempt to mix operands allocated on different device types." );
   static_assert( IsStaticArrayType< T1 >::value == IsStaticArrayType< T2 >::value, "Attempt to mix static and non-static operands in binary expression templates." );

   BinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   static BinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return BinaryExpressionTemplate( a, b );
   }

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
   using RealType = typename T1::RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;

   BinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   BinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return BinaryExpressionTemplate( a, b );
   }

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
   using RealType = typename T2::RealType;
   using DeviceType = typename T2::DeviceType;
   using IndexType = typename T2::IndexType;

   BinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   BinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return BinaryExpressionTemplate( a, b );
   }

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
   using RealType = typename T1::RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;

   UnaryExpressionTemplate( const T1& a ): operand( a ){}

   static UnaryExpressionTemplate evaluate( const T1& a )
   {
      return UnaryExpressionTemplate( a );
   }

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
   return BinaryExpressionTemplate<
      BinaryExpressionTemplate< L1, L2, LOperation >,
      BinaryExpressionTemplate< R1, R2, ROperation >,
      Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator+( const BinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return BinaryExpressionTemplate<
      BinaryExpressionTemplate< T1, T2, Operation >,
      typename BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator+( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return BinaryExpressionTemplate<
      typename BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      BinaryExpressionTemplate< T1, T2, Operation >,
      Addition >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator+( const UnaryExpressionTemplate< T1, Operation >& a,
           const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return BinaryExpressionTemplate<
      UnaryExpressionTemplate< T1, Operation >,
      typename UnaryExpressionTemplate< T1, Operation >::RealType,
      Addition >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator+( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
           const UnaryExpressionTemplate< T1, Operation >& b )
{
   return BinaryExpressionTemplate<
      typename UnaryExpressionTemplate< T1, Operation >::RealType,
      UnaryExpressionTemplate< T1, Operation >,
      Addition >( a, b );
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
   return BinaryExpressionTemplate<
      UnaryExpressionTemplate< L1, LOperation >,
      BinaryExpressionTemplate< R1, R2, ROperation >,
      Addition >( a, b );
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
   return BinaryExpressionTemplate<
      BinaryExpressionTemplate< L1, L2, LOperation >,
      UnaryExpressionTemplate< R1, ROperation >,
      Addition >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator+( const UnaryExpressionTemplate< L1,LOperation >& a,
           const UnaryExpressionTemplate< R1,ROperation >& b )
{
   return BinaryExpressionTemplate<
      UnaryExpressionTemplate< L1, LOperation >,
      UnaryExpressionTemplate< R1, ROperation >,
      Addition >( a, b );
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
   return BinaryExpressionTemplate<
      BinaryExpressionTemplate< L1, L2, LOperation >,
      BinaryExpressionTemplate< R1, R2, ROperation >,
      Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator-( const BinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return BinaryExpressionTemplate<
      BinaryExpressionTemplate< T1, T2, Operation >,
      typename BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator-( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return BinaryExpressionTemplate<
      typename BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      BinaryExpressionTemplate< T1, T2, Operation >,
      Subtraction >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator-( const UnaryExpressionTemplate< T1, Operation >& a,
           const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return BinaryExpressionTemplate<
      UnaryExpressionTemplate< T1, Operation >,
      typename UnaryExpressionTemplate< T1, Operation >::RealType,
      Subtraction >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator-( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
           const UnaryExpressionTemplate< T1, Operation >& b )
{
   return BinaryExpressionTemplate<
      typename UnaryExpressionTemplate< T1, Operation >::RealType,
      UnaryExpressionTemplate< T1, Operation >,
      Subtraction >( a, b );
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
   return BinaryExpressionTemplate<
      UnaryExpressionTemplate< L1, LOperation >,
      BinaryExpressionTemplate< R1, R2, ROperation >,
      Subtraction >( a, b );
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
   return BinaryExpressionTemplate<
      BinaryExpressionTemplate< L1, L2, LOperation >,
      UnaryExpressionTemplate< R1, ROperation >,
      Subtraction >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator-( const UnaryExpressionTemplate< L1,LOperation >& a,
           const UnaryExpressionTemplate< R1,ROperation >& b )
{
   return BinaryExpressionTemplate<
      UnaryExpressionTemplate< L1, LOperation >,
      UnaryExpressionTemplate< R1, ROperation >,
      Subtraction >( a, b );
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
   return BinaryExpressionTemplate<
      BinaryExpressionTemplate< L1, L2, LOperation >,
      BinaryExpressionTemplate< R1, R2, ROperation >,
      Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator*( const BinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return BinaryExpressionTemplate<
      BinaryExpressionTemplate< T1, T2, Operation >,
      typename BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator*( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return BinaryExpressionTemplate<
      typename BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      BinaryExpressionTemplate< T1, T2, Operation >,
      Multiplication >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator*( const UnaryExpressionTemplate< T1, Operation >& a,
           const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return BinaryExpressionTemplate<
      UnaryExpressionTemplate< T1, Operation >,
      typename UnaryExpressionTemplate< T1, Operation >::RealType,
      Multiplication >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator*( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
           const UnaryExpressionTemplate< T1, Operation >& b )
{
   return BinaryExpressionTemplate<
      typename UnaryExpressionTemplate< T1, Operation >::RealType,
      UnaryExpressionTemplate< T1, Operation >,
      Multiplication >( a, b );
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
   return BinaryExpressionTemplate<
      UnaryExpressionTemplate< L1, LOperation >,
      BinaryExpressionTemplate< R1, R2, ROperation >,
      Multiplication >( a, b );
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
   return BinaryExpressionTemplate<
      BinaryExpressionTemplate< L1, L2, LOperation >,
      UnaryExpressionTemplate< R1, ROperation >,
      Multiplication >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator*( const UnaryExpressionTemplate< L1,LOperation >& a,
           const UnaryExpressionTemplate< R1,ROperation >& b )
{
   return BinaryExpressionTemplate<
      UnaryExpressionTemplate< L1, LOperation >,
      UnaryExpressionTemplate< R1, ROperation >,
      Multiplication >( a, b );
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
   return BinaryExpressionTemplate<
      BinaryExpressionTemplate< L1, L2, LOperation >,
      BinaryExpressionTemplate< R1, R2, ROperation >,
      Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator/( const BinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return BinaryExpressionTemplate<
      BinaryExpressionTemplate< T1, T2, Operation >,
      typename BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator/( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return BinaryExpressionTemplate<
      typename BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      BinaryExpressionTemplate< T1, T2, Operation >,
      Division >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator/( const UnaryExpressionTemplate< T1, Operation >& a,
           const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return BinaryExpressionTemplate<
      UnaryExpressionTemplate< T1, Operation >,
      typename UnaryExpressionTemplate< T1, Operation >::RealType,
      Division >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator/( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
           const UnaryExpressionTemplate< T1, Operation >& b )
{
   return BinaryExpressionTemplate<
      typename UnaryExpressionTemplate< T1, Operation >::RealType,
      UnaryExpressionTemplate< T1, Operation >,
      Division >( a, b );
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
   return BinaryExpressionTemplate<
      UnaryExpressionTemplate< L1, LOperation >,
      BinaryExpressionTemplate< R1, R2, ROperation >,
      Division >( a, b );
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
   return BinaryExpressionTemplate<
      BinaryExpressionTemplate< L1, L2, LOperation >,
      UnaryExpressionTemplate< R1, ROperation >,
      Division >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator/( const UnaryExpressionTemplate< L1,LOperation >& a,
           const UnaryExpressionTemplate< R1,ROperation >& b )
{
   return BinaryExpressionTemplate<
      UnaryExpressionTemplate< L1, LOperation >,
      UnaryExpressionTemplate< R1, ROperation >,
      Division >( a, b );
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
   using Left = BinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = BinaryExpressionTemplate< R1, R2, ROperation >;
   return Comparison< Left, Right >::EQ( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator==( const BinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = BinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename BinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return Comparison< Left, Right >::EQ( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator==( const UnaryExpressionTemplate< T1, Operation >& a,
            const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   using Left = UnaryExpressionTemplate< T1, Operation >;
   using Right = typename UnaryExpressionTemplate< T1, Operation >::RealType;
   return Comparison< Left, Right >::EQ( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator==( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename BinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = BinaryExpressionTemplate< T1, T2, Operation >;
   return Comparison< Left, Right >::EQ( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator==( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
            const UnaryExpressionTemplate< T1, Operation >& b )
{
   using Left = typename UnaryExpressionTemplate< T1, Operation >::RealType;
   using Right = UnaryExpressionTemplate< T1, Operation >;
   return Comparison< Left, Right >::EQ( a, b );
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
   using Left = UnaryExpressionTemplate< L1, LOperation >;
   using Right = BinaryExpressionTemplate< R1, R2, ROperation >;
   return Comparison< Left, Right >::EQ( a, b );
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
   using Left = BinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = UnaryExpressionTemplate< R1, ROperation >;
   return Comparison< Left, Right >::EQ( a, b );
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
   using Left = BinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = BinaryExpressionTemplate< R1, R2, ROperation >;
   return Comparison< Left, Right >::NE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator!=( const BinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = BinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename BinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return Comparison< Left, Right >::NE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator!=( const UnaryExpressionTemplate< L1, LOperation >& a,
            const UnaryExpressionTemplate< R1, ROperation >& b )
{
   using Left = UnaryExpressionTemplate< L1, LOperation >;
   using Right = UnaryExpressionTemplate< R1, ROperation >;
   return Comparison< Left, Right >::NE( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator!=( const UnaryExpressionTemplate< T1, Operation >& a,
            const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   using Left = UnaryExpressionTemplate< T1, Operation >;
   using Right = typename UnaryExpressionTemplate< T1, Operation >::RealType;
   return Comparison< Left, Right >::NE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator!=( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename BinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = BinaryExpressionTemplate< T1, T2, Operation >;
   return Comparison< Left, Right >::NE( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator!=( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
            const UnaryExpressionTemplate< T1, Operation >& b )
{
   using Left = typename UnaryExpressionTemplate< T1, Operation >::RealType;
   using Right = UnaryExpressionTemplate< T1, Operation >;
   return Comparison< Left, Right >::NE( a, b );
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
   using Left = UnaryExpressionTemplate< L1, LOperation >;
   using Right = BinaryExpressionTemplate< R1, R2, ROperation >;
   return Comparison< Left, Right >::NE( a, b );
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
   using Left = BinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = UnaryExpressionTemplate< R1, ROperation >;
   return Comparison< Left, Right >::NE( a, b );
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
   using Left = BinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = BinaryExpressionTemplate< R1, R2, ROperation >;
   return Comparison< Left, Right >::LT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator<( const BinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = BinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename BinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return Comparison< Left, Right >::LT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator<( const UnaryExpressionTemplate< L1, LOperation >& a,
           const UnaryExpressionTemplate< R1, ROperation >& b )
{
   using Left = UnaryExpressionTemplate< L1, LOperation >;
   using Right = UnaryExpressionTemplate< R1, ROperation >;
   return Comparison< Left, Right >::LT( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator<( const UnaryExpressionTemplate< T1, Operation >& a,
           const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   using Left = UnaryExpressionTemplate< T1, Operation >;
   using Right = typename UnaryExpressionTemplate< T1, Operation >::RealType;
   return Comparison< Left, Right >::LT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator<( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename BinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = BinaryExpressionTemplate< T1, T2, Operation >;
   return Comparison< Left, Right >::LT( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator<( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
           const UnaryExpressionTemplate< T1, Operation >& b )
{
   using Left = typename UnaryExpressionTemplate< T1, Operation >::RealType;
   using Right = UnaryExpressionTemplate< T1, Operation >;
   return Comparison< Left, Right >::LT( a, b );
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
   using Left = UnaryExpressionTemplate< L1, LOperation >;
   using Right = BinaryExpressionTemplate< R1, R2, ROperation >;
   return Comparison< Left, Right >::LT( a, b );
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
   using Left = BinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = UnaryExpressionTemplate< R1, ROperation >;
   return Comparison< Left, Right >::LT( a, b );
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
   using Left = BinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = BinaryExpressionTemplate< R1, R2, ROperation >;
   return Comparison< Left, Right >::LE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator<=( const BinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = BinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename BinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return Comparison< Left, Right >::LE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator<=( const UnaryExpressionTemplate< L1, LOperation >& a,
            const UnaryExpressionTemplate< R1, ROperation >& b )
{
   using Left = UnaryExpressionTemplate< L1, LOperation >;
   using Right = UnaryExpressionTemplate< R1, ROperation >;
   return Comparison< Left, Right >::LE( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator<=( const UnaryExpressionTemplate< T1, Operation >& a,
            const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   using Left = UnaryExpressionTemplate< T1, Operation >;
   using Right = typename UnaryExpressionTemplate< T1, Operation >::RealType;
   return Comparison< Left, Right >::LE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator<=( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename BinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = BinaryExpressionTemplate< T1, T2, Operation >;
   return Comparison< Left, Right >::LE( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator<=( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
            const UnaryExpressionTemplate< T1, Operation >& b )
{
   using Left = typename UnaryExpressionTemplate< T1, Operation >::RealType;
   using Right = UnaryExpressionTemplate< T1, Operation >;
   return Comparison< Left, Right >::LE( a, b );
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
   using Left = UnaryExpressionTemplate< L1, LOperation >;
   using Right = BinaryExpressionTemplate< R1, R2, ROperation >;
   return Comparison< Left, Right >::LE( a, b );
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
   using Left = BinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = UnaryExpressionTemplate< R1, ROperation >;
   return Comparison< Left, Right >::LE( a, b );
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
   using Left = BinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = BinaryExpressionTemplate< R1, R2, ROperation >;
   return Comparison< Left, Right >::GT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator>( const BinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = BinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename BinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return Comparison< Left, Right >::GT( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator>( const UnaryExpressionTemplate< T1, Operation >& a,
           const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   using Left = UnaryExpressionTemplate< T1, Operation >;
   using Right = typename UnaryExpressionTemplate< T1, Operation >::RealType;
   return Comparison< Left, Right >::GT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator>( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename BinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = BinaryExpressionTemplate< T1, T2, Operation >;
   return Comparison< Left, Right >::GT( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator>( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
           const UnaryExpressionTemplate< T1, Operation >& b )
{
   using Left = typename UnaryExpressionTemplate< T1, Operation >::RealType;
   using Right = UnaryExpressionTemplate< T1, Operation >;
   return Comparison< Left, Right >::GT( a, b );
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
   using Left = UnaryExpressionTemplate< L1, LOperation >;
   using Right = BinaryExpressionTemplate< R1, R2, ROperation >;
   return Comparison< Left, Right >::GT( a, b );
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
   using Left = BinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = UnaryExpressionTemplate< R1, ROperation >;
   return Comparison< Left, Right >::GT( a, b );
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
   using Left = BinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = BinaryExpressionTemplate< R1, R2, ROperation >;
   return Comparison< Left, Right >::GE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator>=( const BinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = BinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename BinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return Comparison< Left, Right >::GE( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator>=( const UnaryExpressionTemplate< T1, Operation >& a,
            const typename UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   using Left = UnaryExpressionTemplate< T1, Operation >;
   using Right = typename UnaryExpressionTemplate< T1, Operation >::RealType;
   return Comparison< Left, Right >::GE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator>=( const typename BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename BinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = BinaryExpressionTemplate< T1, T2, Operation >;
   return Comparison< Left, Right >::GE( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator>=( const typename UnaryExpressionTemplate< T1, Operation >::RealType& a,
            const UnaryExpressionTemplate< T1, Operation >& b )
{
   using Left = typename UnaryExpressionTemplate< T1, Operation >::RealType;
   using Right = UnaryExpressionTemplate< T1, Operation >;
   return Comparison< Left, Right >::GE( a, b );
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
   using Left = UnaryExpressionTemplate< L1, LOperation >;
   using Right = BinaryExpressionTemplate< R1, R2, ROperation >;
   return Comparison< Left, Right >::GE( a, b );
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
   using Left = BinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = UnaryExpressionTemplate< R1, ROperation >;
   return Comparison< Left, Right >::GE( a, b );
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
   return UnaryExpressionTemplate<
      BinaryExpressionTemplate< L1, L2, LOperation >,
      Minus >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
operator-( const UnaryExpressionTemplate< L1, LOperation >& a )
{
   return UnaryExpressionTemplate<
      UnaryExpressionTemplate< L1, LOperation >,
      Minus >( a );
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
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
min( const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
min( const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
     const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
min( const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& a,
     const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
      typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
min( const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& a,
     const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
      Containers::Expressions::Min >( a, b );
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
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Min >( a, b );
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
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Min >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
min( const Containers::Expressions::UnaryExpressionTemplate< L1,LOperation >& a,
     const Containers::Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Min >( a, b );
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
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
max( const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
max( const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
     const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
max( const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& a,
     const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
      typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
max( const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& a,
     const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
      Containers::Expressions::Max >( a, b );
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
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Max >( a, b );
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
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Max >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
max( const Containers::Expressions::UnaryExpressionTemplate< L1,LOperation >& a,
     const Containers::Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Max >( a, b );
}

////
// Abs
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
abs( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Abs >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
abs( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Abs >( a );
}

////
// Sin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
sin( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Sin >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
sin( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Sin >( a );
}

////
// Cos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
cos( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Cos >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
cos( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Cos >( a );
}

////
// Tan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
tan( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Tan >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
tan( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Tan >( a );
}

////
// Sqrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
sqrt( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Sqrt >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
sqrt( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Sqrt >( a );
}

////
// Cbrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
cbrt( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Cbrt >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
cbrt( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Cbrt >( a );
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
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Real,
      Containers::Expressions::Pow >( a, exp );
}

template< typename L1,
          template< typename > class LOperation,
          typename Real >
auto
pow( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a, const Real& exp )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Real,
      Containers::Expressions::Pow >( a, exp );
}

////
// Floor
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
floor( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Floor >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
floor( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Floor >( a );
}

////
// Ceil
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
ceil( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Ceil >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
ceil( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Ceil >( a );
}

////
// Asin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
asin( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Asin >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
asin( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Asin >( a );
}

////
// Acos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
acos( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Acos >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
acos( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Acos >( a );
}

////
// Atan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
atan( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Atan >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
atan( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Atan >( a );
}

////
// Sinh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
sinh( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Sinh >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
sinh( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Sinh >( a );
}

////
// Cosh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
cosh( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Cosh >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
cosh( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Cosh >( a );
}

////
// Tanh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
tanh( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Tanh >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
tanh( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Tanh >( a );
}

////
// Log
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
log( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Log >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
log( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Log >( a );
}

////
// Log10
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
log10( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Log10 >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
log10( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Log10 >( a );
}

////
// Log2
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
log2( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Log2 >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
log2( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Log2 >( a );
}

////
// Exp
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
exp( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Exp >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
exp( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Exp >( a );
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
          template< typename, typename > class LOperation,
          typename Real >
auto
lpNorm( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a, const Real& p )
-> decltype( ExpressionLpNorm( a, p ) )
{
   if( p == 1.0 )
      return ExpressionLpNorm( a, p );
   if( p == 2.0 )
      return TNL::sqrt( ExpressionLpNorm( a, p ) );
   return TNL::pow( ExpressionLpNorm( a, p ), 1.0 / p );
}

template< typename L1,
          template< typename > class LOperation,
          typename Real >
auto
lpNorm( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a, const Real& p )
-> decltype( ExpressionLpNorm( a, p ) )
{
   if( p == 1.0 )
      return ExpressionLpNorm( a, p );
   if( p == 2.0 )
      return TNL::sqrt( ExpressionLpNorm( a, p ) );
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
   return TNL::sum( a * b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
dot( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
     const Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >& b )
{
   return TNL::sum( a * b );
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
   return TNL::sum( a * b );
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
   return TNL::sum( a * b );
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
