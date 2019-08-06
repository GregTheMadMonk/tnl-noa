/***************************************************************************
                          StaticExpressionTemplates.h  -  description
                             -------------------
    begin                : Apr 18, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <ostream>

#include <TNL/TypeTraits.h>
#include <TNL/Containers/Expressions/TypeTraits.h>
#include <TNL/Containers/Expressions/ExpressionTemplatesOperations.h>
#include <TNL/Containers/Expressions/ExpressionVariableType.h>
#include <TNL/Containers/Expressions/StaticComparison.h>
#include <TNL/Containers/Expressions/StaticVerticalOperations.h>

namespace TNL {
namespace Containers {
namespace Expressions {

template< typename T1,
          template< typename > class Operation,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value >
struct StaticUnaryExpressionTemplate
{};

template< typename T1,
          template< typename > class Operation,
          ExpressionVariableType T1Type >
struct IsExpressionTemplate< StaticUnaryExpressionTemplate< T1, Operation, T1Type > >
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
template< typename T1,
          template< typename > class Operation >
struct StaticUnaryExpressionTemplate< T1, Operation, VectorExpressionVariable >
{
   using RealType = typename T1::RealType;

   static constexpr int getSize() { return T1::getSize(); };

   __cuda_callable__
   StaticUnaryExpressionTemplate( const T1& a ): operand( a ){}

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
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator+( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator+( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator+( const StaticUnaryExpressionTemplate< T1, Operation >& a,
           const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator+( const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& a,
           const StaticUnaryExpressionTemplate< T1, Operation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator+( const StaticUnaryExpressionTemplate< L1, LOperation >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator+( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator+( const StaticUnaryExpressionTemplate< L1,LOperation >& a,
           const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
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
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator-( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator-( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator-( const StaticUnaryExpressionTemplate< T1, Operation >& a,
           const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator-( const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& a,
           const StaticUnaryExpressionTemplate< T1, Operation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator-( const StaticUnaryExpressionTemplate< L1, LOperation >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator-( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator-( const StaticUnaryExpressionTemplate< L1,LOperation >& a,
           const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
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
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator*( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator*( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator*( const StaticUnaryExpressionTemplate< T1, Operation >& a,
           const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator*( const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& a,
           const StaticUnaryExpressionTemplate< T1, Operation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator*( const StaticUnaryExpressionTemplate< L1, LOperation >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator*( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator*( const StaticUnaryExpressionTemplate< L1,LOperation >& a,
           const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
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
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator/( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator/( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator/( const StaticUnaryExpressionTemplate< T1, Operation >& a,
           const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator/( const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& a,
           const StaticUnaryExpressionTemplate< T1, Operation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator/( const StaticUnaryExpressionTemplate< L1, LOperation >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator/( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator/( const StaticUnaryExpressionTemplate< L1,LOperation >& a,
           const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
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
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator==( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
}

template< typename T1,
          template< typename > class Operation >
__cuda_callable__
bool
operator==( const StaticUnaryExpressionTemplate< T1, Operation >& a,
            const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator==( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
}

template< typename T1,
          template< typename > class Operation >
__cuda_callable__
bool
operator==( const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& a,
            const StaticUnaryExpressionTemplate< T1, Operation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator==( const StaticUnaryExpressionTemplate< L1, LOperation >& a,
            const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator==( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
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
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator!=( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator!=( const StaticUnaryExpressionTemplate< L1, LOperation >& a,
            const StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename T1,
          template< typename > class Operation >
__cuda_callable__
bool
operator!=( const StaticUnaryExpressionTemplate< T1, Operation >& a,
            const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator!=( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename T1,
          template< typename > class Operation >
__cuda_callable__
bool
operator!=( const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& a,
            const StaticUnaryExpressionTemplate< T1, Operation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator!=( const StaticUnaryExpressionTemplate< L1, LOperation >& a,
            const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator!=( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
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
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator<( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator<( const StaticUnaryExpressionTemplate< L1, LOperation >& a,
           const StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename T1,
          template< typename > class Operation >
__cuda_callable__
bool
operator<( const StaticUnaryExpressionTemplate< T1, Operation >& a,
           const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator<( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename T1,
          template< typename > class Operation >
__cuda_callable__
bool
operator<( const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& a,
           const StaticUnaryExpressionTemplate< T1, Operation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator<( const StaticUnaryExpressionTemplate< L1, LOperation >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator<( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
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
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator<=( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator<=( const StaticUnaryExpressionTemplate< L1, LOperation >& a,
            const StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename T1,
          template< typename > class Operation >
__cuda_callable__
bool
operator<=( const StaticUnaryExpressionTemplate< T1, Operation >& a,
            const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator<=( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename T1,
          template< typename > class Operation >
__cuda_callable__
bool
operator<=( const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& a,
            const StaticUnaryExpressionTemplate< T1, Operation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator<=( const StaticUnaryExpressionTemplate< L1, LOperation >& a,
            const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator<=( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
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
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator>( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
}

template< typename T1,
          template< typename > class Operation >
__cuda_callable__
bool
operator>( const StaticUnaryExpressionTemplate< T1, Operation >& a,
           const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator>( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
}

template< typename T1,
          template< typename > class Operation >
__cuda_callable__
bool
operator>( const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& a,
           const StaticUnaryExpressionTemplate< T1, Operation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator>( const StaticUnaryExpressionTemplate< L1, LOperation >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator>( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
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
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator>=( const StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

template< typename T1,
          template< typename > class Operation >
__cuda_callable__
bool
operator>=( const StaticUnaryExpressionTemplate< T1, Operation >& a,
            const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator>=( const typename StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

template< typename T1,
          template< typename > class Operation >
__cuda_callable__
bool
operator>=( const typename StaticUnaryExpressionTemplate< T1, Operation >::RealType& a,
            const StaticUnaryExpressionTemplate< T1, Operation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator>=( const StaticUnaryExpressionTemplate< L1, LOperation >& a,
            const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator>=( const StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return StaticComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
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
   return StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Minus >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
auto
operator-( const StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Minus >( a );
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
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
auto
operator,( const StaticUnaryExpressionTemplate< L1, LOperation >& a,
           const StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return StaticExpressionSum( a * b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
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
          template< typename > class Operation >
std::ostream& operator<<( std::ostream& str, const StaticUnaryExpressionTemplate< T, Operation >& expression )
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
   return Containers::Expressions::StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
min( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
min( const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
     const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
min( const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation >& a,
     const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
min( const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation >::RealType& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
min( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
     const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
min( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
min( const Containers::Expressions::StaticUnaryExpressionTemplate< L1,LOperation >& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
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
   return Containers::Expressions::StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
max( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
max( const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
     const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
max( const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation >& a,
     const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
max( const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation >::RealType& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
max( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
     const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
max( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
max( const Containers::Expressions::StaticUnaryExpressionTemplate< L1,LOperation >& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
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
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Abs >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
auto
abs( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Abs >( a );
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
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sin >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
auto
sin( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sin >( a );
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
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Cos >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
auto
cos( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Cos >( a );
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
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Tan >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
auto
tan( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Tan >( a );
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
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sqrt >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
auto
sqrt( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sqrt >( a );
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
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Cbrt >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
auto
cbrt( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Cbrt >( a );
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
   return Containers::Expressions::StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, Real, Containers::Expressions::Pow >( a, exp );
}

template< typename L1,
          template< typename > class LOperation,
          typename Real >
__cuda_callable__
auto
pow( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a, const Real& exp )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< std::decay_t<decltype(a)>, Real, Containers::Expressions::Pow >( a, exp );
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
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Floor >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
auto
floor( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Floor >( a );
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
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Ceil >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
auto
ceil( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Ceil >( a );
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
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Asin >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
auto
asin( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Asin >( a );
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
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Acos >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
auto
acos( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Acos >( a );
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
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Atan >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
auto
atan( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Atan >( a );
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
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sinh >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
auto
sinh( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sinh >( a );
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
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Cosh >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
auto
cosh( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Cosh >( a );
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
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Tanh >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
auto
tanh( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Tanh >( a );
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
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Log >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
auto
log( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Log >( a );
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
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Log10 >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
auto
log10( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Log10 >( a );
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
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Log2 >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
auto
log2( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Log2 >( a );
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
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Exp >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
auto
exp( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Exp >( a );
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
          template< typename > class LOperation >
__cuda_callable__
auto
min( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
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
          typename Index >
__cuda_callable__
auto
argMin( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a, Index& arg )
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
          template< typename > class LOperation >
__cuda_callable__
auto
max( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
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
          typename Index >
__cuda_callable__
auto
argMax( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a, Index& arg )
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
          template< typename > class LOperation >
__cuda_callable__
auto
sum( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
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
          typename Real >
__cuda_callable__
auto
lpNorm( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a, const Real& p )
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
          template< typename > class LOperation >
__cuda_callable__
auto
product( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
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
          template< typename > class LOperation >
__cuda_callable__
auto
logicalOr( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
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
          template< typename > class LOperation >
__cuda_callable__
auto
binaryOr( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
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
   return (a, b);
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
auto
dot( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return (a, b);
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
auto
dot( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
     const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return (a, b);
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
auto
dot( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
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
   typename Reduction,
   typename VolatileReduction,
   typename Result >
__cuda_callable__
Result evaluateAndReduce( Vector& lhs,
   const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation >& expression,
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
   typename Reduction,
   typename VolatileReduction,
   typename Result >
__cuda_callable__
Result addAndReduce( Vector& lhs,
   const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation >& expression,
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
   typename Reduction,
   typename VolatileReduction,
   typename Result >
__cuda_callable__
Result addAndReduceAbs( Vector& lhs,
   const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation >& expression,
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
