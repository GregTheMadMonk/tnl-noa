/***************************************************************************
                          StaticExpressionTemplates.h  -  description
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
#include <TNL/Containers/Expressions/StaticComparison.h>
#include <TNL/Containers/Expressions/VerticalOperations.h>

namespace TNL {
namespace Containers {
namespace Expressions {

template< typename T1,
          template< typename > class Operation,
          typename Parameter = void,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value >
struct StaticUnaryExpressionTemplate
{};

template< typename T1,
          template< typename > class Operation,
          typename Parameter,
          ExpressionVariableType T1Type >
struct IsExpressionTemplate< StaticUnaryExpressionTemplate< T1, Operation, Parameter, T1Type > >
: std::true_type
{};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value,
          ExpressionVariableType T2Type = ExpressionVariableTypeGetter< T2 >::value >
struct StaticBinaryExpressionTemplate
{};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          ExpressionVariableType T1Type,
          ExpressionVariableType T2Type >
struct IsExpressionTemplate< StaticBinaryExpressionTemplate< T1, T2, Operation, T1Type, T2Type > >
: std::true_type
{};


////
// Static binary expression template
template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct StaticBinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, VectorExpressionVariable >
{
   static_assert( IsStaticArrayType< T1 >::value, "Left-hand side operand of static expression is not static, i.e. based on static vector." );
   static_assert( IsStaticArrayType< T2 >::value, "Right-hand side operand of static expression is not static, i.e. based on static vector." );
   using RealType = typename T1::RealType;

   static_assert( IsStaticArrayType< T1 >::value == IsStaticArrayType< T2 >::value, "Attempt to mix static and non-static operands in binary expression templates" );
   static_assert( T1::getSize() == T2::getSize(), "Attempt to mix static operands with different sizes." );

   static constexpr int getSize() { return T1::getSize(); };

   __cuda_callable__
   StaticBinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   __cuda_callable__
   static StaticBinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return StaticBinaryExpressionTemplate( a, b );
   }

   RealType getElement( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< typename T1::RealType, typename T2::RealType >::evaluate( op1[ i ], op2[ i ] );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< typename T1::RealType, typename T2::RealType >::evaluate( op1[ i ], op2[ i ] );
   }

   __cuda_callable__
   RealType x() const
   {
      return (*this)[ 0 ];
   }

   __cuda_callable__
   RealType y() const
   {
      return (*this)[ 1 ];
   }

   __cuda_callable__
   RealType z() const
   {
      return (*this)[ 2 ];
   }

   protected:
      const T1 &op1;
      const T2 &op2;
};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct StaticBinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, ArithmeticVariable  >
{
   static_assert( IsStaticArrayType< T1 >::value, "Left-hand side operand of static expression is not static, i.e. based on static vector." );

   using RealType = typename T1::RealType;

   static constexpr int getSize() { return T1::getSize(); };


   __cuda_callable__
   StaticBinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   __cuda_callable__
   StaticBinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return StaticBinaryExpressionTemplate( a, b );
   }

   RealType getElement( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< typename T1::RealType, T2 >::evaluate( op1[ i ], op2 );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< typename T1::RealType, T2 >::evaluate( op1[ i ], op2 );
   }

   __cuda_callable__
   RealType x() const
   {
      return (*this)[ 0 ];
   }

   __cuda_callable__
   RealType y() const
   {
      return (*this)[ 1 ];
   }

   __cuda_callable__
   RealType z() const
   {
      return (*this)[ 2 ];
   }

   protected:
      const T1 &op1;
      const T2 &op2;

};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct StaticBinaryExpressionTemplate< T1, T2, Operation, ArithmeticVariable, VectorExpressionVariable  >
{
   static_assert( IsStaticArrayType< T2 >::value, "Right-hand side operand of static expression is not static, i.e. based on static vector." );

   using RealType = typename T2::RealType;

   static constexpr int getSize() { return T2::getSize(); };


   __cuda_callable__
   StaticBinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   __cuda_callable__
   StaticBinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return StaticBinaryExpressionTemplate( a, b );
   }

   RealType getElement( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< T1, typename T2::RealType >::evaluate( op1, op2[ i ] );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< T1, typename T2::RealType >::evaluate( op1, op2[ i ] );
   }

   __cuda_callable__
   RealType x() const
   {
      return (*this)[ 0 ];
   }

   __cuda_callable__
   RealType y() const
   {
      return (*this)[ 1 ];
   }

   __cuda_callable__
   RealType z() const
   {
      return (*this)[ 2 ];
   }

   protected:
      const T1& op1;
      const T2& op2;
};

////
// Static unary expression template
//
// Parameter type serves mainly for pow( base, exp ). Here exp is parameter we need
// to pass to pow.
template< typename T1,
          template< typename > class Operation,
          typename Parameter >
struct StaticUnaryExpressionTemplate< T1, Operation, Parameter, VectorExpressionVariable >
{
   static_assert( IsStaticArrayType< T1 >::value, "Operand of static expression is not static, i.e. based on static vector." );

   using RealType = typename T1::RealType;

   static constexpr int getSize() { return T1::getSize(); };

   __cuda_callable__
   StaticUnaryExpressionTemplate( const T1& a, const Parameter& p )
   : operand( a ), parameter( p ) {}

   __cuda_callable__
   static StaticUnaryExpressionTemplate evaluate( const T1& a )
   {
      return StaticUnaryExpressionTemplate( a );
   }

   RealType getElement( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< typename T1::RealType >::evaluate( operand[ i ], parameter );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< typename T1::RealType >::evaluate( operand[ i ], parameter );
   }

   __cuda_callable__
   RealType x() const
   {
      return (*this)[ 0 ];
   }

   __cuda_callable__
   RealType y() const
   {
      return (*this)[ 1 ];
   }

   __cuda_callable__
   RealType z() const
   {
      return (*this)[ 2 ];
   }

   void set( const Parameter& p ) { parameter = p; }

   const Parameter& get() { return parameter; }

   protected:
      const T1& operand;
      Parameter parameter;
};

////
// Static unary expression template with no parameter
template< typename T1,
          template< typename > class Operation >
struct StaticUnaryExpressionTemplate< T1, Operation, void, VectorExpressionVariable >
{
   using RealType = typename T1::RealType;

   static constexpr int getSize() { return T1::getSize(); };

   __cuda_callable__
   StaticUnaryExpressionTemplate( const T1& a ): operand( a ){}

   __cuda_callable__
   static StaticUnaryExpressionTemplate evaluate( const T1& a )
   {
      return StaticUnaryExpressionTemplate( a );
   }

   RealType getElement( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< typename T1::RealType >::evaluate( operand[ i ] );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< typename T1::RealType >::evaluate( operand[ i ] );
   }

   __cuda_callable__
   RealType x() const
   {
      return (*this)[ 0 ];
   }

   __cuda_callable__
   RealType y() const
   {
      return (*this)[ 1 ];
   }

   __cuda_callable__
   RealType z() const
   {
      return (*this)[ 2 ];
   }

   protected:
      const T1& operand;
};

////
// Binary expressions addition
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator+( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticBinaryExpressionTemplate<
      StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator+( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return StaticBinaryExpressionTemplate<
      StaticBinaryExpressionTemplate< T1, T2, Operation >,
      typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator+( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return StaticBinaryExpressionTemplate<
      typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      StaticBinaryExpressionTemplate< T1, T2, Operation >,
      Addition >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
auto
operator+( const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
           const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   return StaticBinaryExpressionTemplate<
      StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Addition >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
auto
operator+( const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
           const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   return StaticBinaryExpressionTemplate<
      typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      Addition >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator+( const StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticBinaryExpressionTemplate<
      StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Addition >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
auto
operator+( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticBinaryExpressionTemplate<
      StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
      Addition >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
auto
operator+( const StaticUnaryExpressionTemplate< L1,LOperation >& a,
           const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticBinaryExpressionTemplate<
      StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
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
operator-( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticBinaryExpressionTemplate<
      StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator-( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return StaticBinaryExpressionTemplate<
      StaticBinaryExpressionTemplate< T1, T2, Operation >,
      typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator-( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return StaticBinaryExpressionTemplate<
      typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      StaticBinaryExpressionTemplate< T1, T2, Operation >,
      Subtraction >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
auto
operator-( const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
           const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   return StaticBinaryExpressionTemplate<
      StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Subtraction >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
auto
operator-( const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
           const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   return StaticBinaryExpressionTemplate<
      typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      Subtraction >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator-( const StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticBinaryExpressionTemplate<
      StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Subtraction >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
auto
operator-( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticBinaryExpressionTemplate<
      StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
      Subtraction >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
auto
operator-( const StaticUnaryExpressionTemplate< L1,LOperation >& a,
           const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticBinaryExpressionTemplate<
      StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
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
operator*( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticBinaryExpressionTemplate<
      StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator*( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return StaticBinaryExpressionTemplate<
      StaticBinaryExpressionTemplate< T1, T2, Operation >,
      typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator*( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return StaticBinaryExpressionTemplate<
      typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      StaticBinaryExpressionTemplate< T1, T2, Operation >,
      Multiplication >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
auto
operator*( const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
           const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   return StaticBinaryExpressionTemplate<
      StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Multiplication >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
auto
operator*( const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
           const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   return StaticBinaryExpressionTemplate<
      typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      Multiplication >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator*( const StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticBinaryExpressionTemplate<
      StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Multiplication >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
auto
operator*( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticUnaryExpressionTemplate< R1, ROperation, RParameter >& b )
{
   return StaticBinaryExpressionTemplate<
      StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
      Multiplication >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
auto
operator*( const StaticUnaryExpressionTemplate< L1,LOperation >& a,
           const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticBinaryExpressionTemplate<
      StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
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
operator/( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticBinaryExpressionTemplate<
      StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator/( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return StaticBinaryExpressionTemplate<
      StaticBinaryExpressionTemplate< T1, T2, Operation >,
      typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator/( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return StaticBinaryExpressionTemplate<
      typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      StaticBinaryExpressionTemplate< T1, T2, Operation >,
      Division >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
auto
operator/( const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
           const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   return StaticBinaryExpressionTemplate<
      StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Division >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
auto
operator/( const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
           const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   return StaticBinaryExpressionTemplate<
      typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      Division >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator/( const StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticBinaryExpressionTemplate<
      StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Division >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
auto
operator/( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticBinaryExpressionTemplate<
      StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
      Division >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
auto
operator/( const StaticUnaryExpressionTemplate< L1,LOperation >& a,
           const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticBinaryExpressionTemplate<
      StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
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
__cuda_callable__
bool
operator==( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return StaticComparison< Left, Right >::EQ( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator==( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = StaticBinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return StaticComparison< Left, Right >::EQ( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator==( const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
            const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   using Left = StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   using Right = typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   return StaticComparison< Left, Right >::EQ( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator==( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = StaticBinaryExpressionTemplate< T1, T2, Operation >;
   return StaticComparison< Left, Right >::EQ( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator==( const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
            const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   using Left = typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   using Right = StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   return StaticComparison< Left, Right >::EQ( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator==( const StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
            const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = StaticUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return StaticComparison< Left, Right >::EQ( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
__cuda_callable__
bool
operator==( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const StaticUnaryExpressionTemplate< R1,ROperation, RParameter >& b )
{
   using Left = StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = StaticUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return StaticComparison< Left, Right >::EQ( a, b );
}

////
// Comparison operator !=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator!=( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return StaticComparison< Left, Right >::NE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator!=( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = StaticBinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return StaticComparison< Left, Right >::NE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
__cuda_callable__
bool
operator!=( const StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
            const StaticUnaryExpressionTemplate< R1, ROperation, RParameter >& b )
{
   using Left = StaticUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = StaticUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return StaticComparison< Left, Right >::NE( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator!=( const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
            const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   using Left = StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   using Right = typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   return StaticComparison< Left, Right >::NE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator!=( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = StaticBinaryExpressionTemplate< T1, T2, Operation >;
   return StaticComparison< Left, Right >::NE( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator!=( const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
            const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   using Left = typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   using Right = StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   return StaticComparison< Left, Right >::NE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator!=( const StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
            const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = StaticUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return StaticComparison< Left, Right >::NE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
__cuda_callable__
bool
operator!=( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const StaticUnaryExpressionTemplate< R1,ROperation, RParameter >& b )
{
   using Left = StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = StaticUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return StaticComparison< Left, Right >::NE( a, b );
}

////
// Comparison operator <
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator<( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return StaticComparison< Left, Right >::LT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator<( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = StaticBinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return StaticComparison< Left, Right >::LT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
__cuda_callable__
bool
operator<( const StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
           const StaticUnaryExpressionTemplate< R1, ROperation, RParameter >& b )
{
   using Left = StaticUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = StaticUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return StaticComparison< Left, Right >::LT( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator<( const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
           const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   using Left = StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   using Right = typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   return StaticComparison< Left, Right >::LT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator<( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = StaticBinaryExpressionTemplate< T1, T2, Operation >;
   return StaticComparison< Left, Right >::LT( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator<( const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
           const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   using Left = typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   using Right = StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   return StaticComparison< Left, Right >::LT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator<( const StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = StaticUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return StaticComparison< Left, Right >::LT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
__cuda_callable__
bool
operator<( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticUnaryExpressionTemplate< R1,ROperation, RParameter >& b )
{
   using Left = StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = StaticUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return StaticComparison< Left, Right >::LT( a, b );
}

////
// Comparison operator <=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator<=( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return StaticComparison< Left, Right >::LE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator<=( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = StaticBinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return StaticComparison< Left, Right >::LE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
__cuda_callable__
bool
operator<=( const StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
            const StaticUnaryExpressionTemplate< R1, ROperation, RParameter >& b )
{
   using Left = StaticUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = StaticUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return StaticComparison< Left, Right >::LE( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator<=( const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
            const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   using Left = StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   using Right = typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   return StaticComparison< Left, Right >::LE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator<=( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = StaticBinaryExpressionTemplate< T1, T2, Operation >;
   return StaticComparison< Left, Right >::LE( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator<=( const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
            const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   using Left = typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   using Right = StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   return StaticComparison< Left, Right >::LE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator<=( const StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
            const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = StaticUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return StaticComparison< Left, Right >::LE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
__cuda_callable__
bool
operator<=( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const StaticUnaryExpressionTemplate< R1,ROperation, RParameter >& b )
{
   using Left = StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = StaticUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return StaticComparison< Left, Right >::LE( a, b );
}

////
// Comparison operator >
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator>( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return StaticComparison< Left, Right >::GT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator>( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = StaticBinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return StaticComparison< Left, Right >::GT( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator>( const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
           const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   using Left = StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   using Right = typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   return StaticComparison< Left, Right >::GT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator>( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = StaticBinaryExpressionTemplate< T1, T2, Operation >;
   return StaticComparison< Left, Right >::GT( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator>( const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
           const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   using Left = typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   using Right = StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   return StaticComparison< Left, Right >::GT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator>( const StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = StaticUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return StaticComparison< Left, Right >::GT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
__cuda_callable__
bool
operator>( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticUnaryExpressionTemplate< R1,ROperation, RParameter >& b )
{
   using Left = StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = StaticUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return StaticComparison< Left, Right >::GT( a, b );
}

////
// Comparison operator >=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator>=( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return StaticComparison< Left, Right >::GE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator>=( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = StaticBinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return StaticComparison< Left, Right >::GE( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator>=( const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
            const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   using Left = StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   using Right = typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   return StaticComparison< Left, Right >::GE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator>=( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = StaticBinaryExpressionTemplate< T1, T2, Operation >;
   return StaticComparison< Left, Right >::GE( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator>=( const typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
            const StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   using Left = typename StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   using Right = StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   return StaticComparison< Left, Right >::GE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator>=( const StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
            const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = StaticUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return StaticComparison< Left, Right >::GE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
__cuda_callable__
bool
operator>=( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const StaticUnaryExpressionTemplate< R1,ROperation, RParameter >& b )
{
   using Left = StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = StaticUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return StaticComparison< Left, Right >::GE( a, b );
}

////
// Unary operations

////
// Minus
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
operator-( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return StaticUnaryExpressionTemplate<
      StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Minus >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
operator-( const StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return StaticUnaryExpressionTemplate<
      StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
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
__cuda_callable__
auto
operator,( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticExpressionSum( a * b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
auto
operator,( const StaticUnaryExpressionTemplate< L1, LOperation >& a,
           const StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return StaticExpressionSum( a * b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
auto
operator,( const StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticExpressionSum( a * b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
__cuda_callable__
auto
operator,( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticExpressionSum( a * b );
}

////
// Output stream
template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
std::ostream& operator<<( std::ostream& str, const StaticBinaryExpressionTemplate< T1, T2, Operation >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression[ i ] << ", ";
   str << expression[ expression.getSize() - 1 ] << " ]";
   return str;
}

template< typename T,
          template< typename > class Operation,
          typename Parameter >
std::ostream& operator<<( std::ostream& str, const StaticUnaryExpressionTemplate< T, Operation, Parameter >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression[ i ] << ", ";
   str << expression[ expression.getSize() - 1 ] << " ]";
   return str;
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
min( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
min( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
min( const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
     const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
auto
min( const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
     const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
auto
min( const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      Containers::Expressions::Min >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
min( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
     const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Min >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
auto
min( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
      Containers::Expressions::Min >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
auto
min( const Containers::Expressions::StaticUnaryExpressionTemplate< L1,LOperation >& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
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
max( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
max( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
max( const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
     const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
auto
max( const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
     const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
auto
max( const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      Containers::Expressions::Max >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
max( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
     const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Max >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
auto
max( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
      Containers::Expressions::Max >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
auto
max( const Containers::Expressions::StaticUnaryExpressionTemplate< L1,LOperation >& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
      Containers::Expressions::Max >( a, b );
}

////
// Abs
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
abs( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Abs >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
abs( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Abs >( a );
}

////
// Sin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
sin( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Sin >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
sin( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Sin >( a );
}

////
// Cos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
cos( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Cos >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
cos( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Cos >( a );
}

////
// Tan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
tan( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Tan >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
tan( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Tan >( a );
}

////
// Sqrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
sqrt( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Sqrt >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
sqrt( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Sqrt >( a );
}

////
// Cbrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
cbrt( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Cbrt >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
cbrt( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Cbrt >( a );
}

////
// Pow
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Real >
__cuda_callable__
auto
pow( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a, const Real& exp )
{
   auto e = Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Pow >( a );
   e.parameter.set( exp );
   return e;
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename Real >
__cuda_callable__
auto
pow( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a, const Real& exp )
{
   auto e = Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Pow >( a );
   e.parameter.set( exp );
   return e;
}

////
// Floor
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
floor( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Floor >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
floor( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Floor >( a );
}

////
// Ceil
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
ceil( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Ceil >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
ceil( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Ceil >( a );
}

////
// Asin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
asin( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Asin >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
asin( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Asin >( a );
}

////
// Acos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
acos( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Acos >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
acos( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Acos >( a );
}

////
// Atan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
atan( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Atan >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
atan( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Atan >( a );
}

////
// Sinh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
sinh( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Sinh >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
sinh( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Sinh >( a );
}

////
// Cosh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
cosh( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Cosh >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
cosh( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Cosh >( a );
}

////
// Tanh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
tanh( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Tanh >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
tanh( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Tanh >( a );
}

////
// Log
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
log( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Log >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
log( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Log >( a );
}

////
// Log10
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
log10( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Log10 >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
log10( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Log10 >( a );
}

////
// Log2
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
log2( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Log2 >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
log2( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Log2 >( a );
}

////
// Exp
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
exp( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Exp >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
exp( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Exp >( a );
}

////
// Vertical operations - min
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
min( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return StaticExpressionMin( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
min( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return StaticExpressionMin( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Index >
__cuda_callable__
auto
argMin( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a, Index& arg )
{
   return StaticExpressionArgMin( a, arg );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename Index >
__cuda_callable__
auto
argMin( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a, Index& arg )
{
   return StaticExpressionArgMin( a, arg );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
max( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return StaticExpressionMax( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
max( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return StaticExpressionMax( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Index >
__cuda_callable__
auto
argMax( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a, Index& arg )
{
   return StaticExpressionArgMax( a, arg );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename Index >
__cuda_callable__
auto
argMax( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a, Index& arg )
{
   return StaticExpressionArgMax( a, arg );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
sum( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return StaticExpressionSum( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
sum( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return StaticExpressionSum( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Real >
__cuda_callable__
auto
lpNorm( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a, const Real& p )
-> decltype(StaticExpressionLpNorm( a, p ))
{
   if( p == 1.0 )
      return StaticExpressionLpNorm( a, p );
   if( p == 2.0 )
      return TNL::sqrt( StaticExpressionLpNorm( a, p ) );
   return TNL::pow( StaticExpressionLpNorm( a, p ), 1.0 / p );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename Real >
__cuda_callable__
auto
lpNorm( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a, const Real& p )
-> decltype(StaticExpressionLpNorm( a, p ))
{
   if( p == 1.0 )
      return StaticExpressionLpNorm( a, p );
   if( p == 2.0 )
      return TNL::sqrt( StaticExpressionLpNorm( a, p ) );
   return TNL::pow( StaticExpressionLpNorm( a, p ), 1.0 / p );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
product( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return StaticExpressionProduct( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
product( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return StaticExpressionProduct( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
logicalOr( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return StaticExpressionLogicalOr( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
logicalOr( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return StaticExpressionLogicalOr( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
auto
binaryOr( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return StaticExpressionBinaryOr( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
auto
binaryOr( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return StaticExpressionBinaryOr( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
auto
dot( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return TNL::sum( a * b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
auto
operator,( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
           const Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return TNL::sum( a * b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
auto
dot( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
     const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return TNL::sum( a * b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
__cuda_callable__
auto
dot( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
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
__cuda_callable__
Result evaluateAndReduce( Vector& lhs,
   const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ )
      reduction( result, lhs[ i ] = expression[ i ] );
   return result;
}

template< typename Vector,
   typename T1,
   template< typename > class Operation,
   typename Parameter,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
__cuda_callable__
Result evaluateAndReduce( Vector& lhs,
   const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ )
      reduction( result, lhs[ i ] = expression[ i ] );
   return result;
}

////
// Addition with reduction
template< typename Vector,
   typename T1,
   typename T2,
   template< typename, typename > class Operation,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
__cuda_callable__
Result addAndReduce( Vector& lhs,
   const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ ) {
      const Result aux = expression[ i ];
      lhs[ i ] += aux;
      reduction( result, aux );
   }
   return result;
}

template< typename Vector,
   typename T1,
   template< typename > class Operation,
   typename Parameter,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
__cuda_callable__
Result addAndReduce( Vector& lhs,
   const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ ) {
      const Result aux = expression[ i ];
      lhs[ i ] += aux;
      reduction( result, aux );
   }
   return result;
}

////
// Addition with reduction of abs
template< typename Vector,
   typename T1,
   typename T2,
   template< typename, typename > class Operation,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
__cuda_callable__
Result addAndReduceAbs( Vector& lhs,
   const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ ) {
      const Result aux = expression[ i ];
      lhs[ i ] += aux;
      reduction( result, TNL::abs( aux ) );
   }
   return result;
}

template< typename Vector,
   typename T1,
   template< typename > class Operation,
   typename Parameter,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
__cuda_callable__
Result addAndReduceAbs( Vector& lhs,
   const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ ) {
      const Result aux = expression[ i ];
      lhs[ i ] += aux;
      reduction( result, TNL::abs( aux ) );
   }
   return result;
}

} // namespace TNL
