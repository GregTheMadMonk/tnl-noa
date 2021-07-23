/***************************************************************************
                          StaticExpressionTemplates.h  -  description
                             -------------------
    begin                : Apr 18, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <ostream>
#include <utility>

#include <TNL/TypeTraits.h>
#include <TNL/Containers/Expressions/TypeTraits.h>
#include <TNL/Containers/Expressions/ExpressionVariableType.h>
#include <TNL/Containers/Expressions/HorizontalOperations.h>
#include <TNL/Containers/Expressions/StaticComparison.h>
#include <TNL/Containers/Expressions/StaticVerticalOperations.h>

namespace TNL {
namespace Containers {
namespace Expressions {

template< typename T1,
          typename Operation >
struct StaticUnaryExpressionTemplate;

template< typename T1,
          typename Operation >
struct HasEnabledStaticExpressionTemplates< StaticUnaryExpressionTemplate< T1, Operation > >
: std::true_type
{};

template< typename T1,
          typename T2,
          typename Operation,
          ExpressionVariableType T1Type = getExpressionVariableType< T1, T2 >(),
          ExpressionVariableType T2Type = getExpressionVariableType< T2, T1 >() >
struct StaticBinaryExpressionTemplate;

template< typename T1,
          typename T2,
          typename Operation,
          ExpressionVariableType T1Type,
          ExpressionVariableType T2Type >
struct HasEnabledStaticExpressionTemplates< StaticBinaryExpressionTemplate< T1, T2, Operation, T1Type, T2Type > >
: std::true_type
{};


////
// Static binary expression template
template< typename T1,
          typename T2,
          typename Operation >
struct StaticBinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, VectorExpressionVariable >
{
   using VectorOperandType = T1;
   using RealType = decltype( Operation::evaluate( std::declval<T1>()[0], std::declval<T2>()[0] ) );
   using ValueType = RealType;

   static_assert( IsStaticArrayType< T1 >::value,
                  "Left-hand side operand of static expression is not static, i.e. based on static vector." );
   static_assert( IsStaticArrayType< T2 >::value,
                  "Right-hand side operand of static expression is not static, i.e. based on static vector." );
   static_assert( HasEnabledStaticExpressionTemplates< T1 >::value,
                  "Invalid operand in static binary expression templates - static expression templates are not enabled for the left operand." );
   static_assert( HasEnabledStaticExpressionTemplates< T2 >::value,
                  "Invalid operand in static binary expression templates - static expression templates are not enabled for the right operand." );
   static_assert( T1::getSize() == T2::getSize(),
                  "Attempt to mix static operands with different sizes." );

   static constexpr int getSize() { return T1::getSize(); };

   __cuda_callable__
   StaticBinaryExpressionTemplate( const T1& a, const T2& b )
   : op1( a ), op2( b ) {}

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
      return Operation::evaluate( op1[ i ], op2[ i ] );
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
   typename OperandMemberType< T1 >::type op1;
   typename OperandMemberType< T2 >::type op2;
};

template< typename T1,
          typename T2,
          typename Operation >
struct StaticBinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, ArithmeticVariable  >
{
   using VectorOperandType = T1;
   using RealType = decltype( Operation::evaluate( std::declval<T1>()[0], std::declval<T2>() ) );
   using ValueType = RealType;

   static_assert( IsStaticArrayType< T1 >::value,
                  "Left-hand side operand of static expression is not static, i.e. based on static vector." );
   static_assert( HasEnabledStaticExpressionTemplates< T1 >::value,
                  "Invalid operand in static binary expression templates - static expression templates are not enabled for the left operand." );

   static constexpr int getSize() { return T1::getSize(); };

   __cuda_callable__
   StaticBinaryExpressionTemplate( const T1& a, const T2& b )
   : op1( a ), op2( b ) {}

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
      return Operation::evaluate( op1[ i ], op2 );
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
   typename OperandMemberType< T1 >::type op1;
   typename OperandMemberType< T2 >::type op2;
};

template< typename T1,
          typename T2,
          typename Operation >
struct StaticBinaryExpressionTemplate< T1, T2, Operation, ArithmeticVariable, VectorExpressionVariable  >
{
   using VectorOperandType = T2;
   using RealType = decltype( Operation::evaluate( std::declval<T1>(), std::declval<T2>()[0] ) );
   using ValueType = RealType;

   static_assert( IsStaticArrayType< T2 >::value,
                  "Right-hand side operand of static expression is not static, i.e. based on static vector." );
   static_assert( HasEnabledStaticExpressionTemplates< T2 >::value,
                  "Invalid operand in static binary expression templates - static expression templates are not enabled for the right operand." );

   static constexpr int getSize() { return T2::getSize(); };

   __cuda_callable__
   StaticBinaryExpressionTemplate( const T1& a, const T2& b )
   : op1( a ), op2( b ) {}

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
      return Operation::evaluate( op1, op2[ i ] );
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
   typename OperandMemberType< T1 >::type op1;
   typename OperandMemberType< T2 >::type op2;
};

////
// Static unary expression template
template< typename T1,
          typename Operation >
struct StaticUnaryExpressionTemplate
{
   using VectorOperandType = T1;
   using RealType = decltype( Operation::evaluate( std::declval<T1>()[0] ) );
   using ValueType = RealType;

   static_assert( IsStaticArrayType< T1 >::value,
                  "The operand of static expression is not static, i.e. based on static vector." );
   static_assert( HasEnabledStaticExpressionTemplates< T1 >::value,
                  "Invalid operand in static unary expression templates - static expression templates are not enabled for the operand." );

   static constexpr int getSize() { return T1::getSize(); };

   __cuda_callable__
   StaticUnaryExpressionTemplate( const T1& a )
   : operand( a ) {}

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
      return Operation::evaluate( operand[ i ] );
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
   typename OperandMemberType< T1 >::type operand;
};

#ifndef DOXYGEN_ONLY

////
// Binary expressions addition
template< typename ET1, typename ET2,
          typename..., typename = EnableIfStaticBinaryExpression_t< ET1, ET2 > >
__cuda_callable__
auto
operator+( const ET1& a, const ET2& b )
{
   return StaticBinaryExpressionTemplate< ET1, ET2, Addition >( a, b );
}

////
// Binary expression subtraction
template< typename ET1, typename ET2,
          typename..., typename = EnableIfStaticBinaryExpression_t< ET1, ET2 > >
__cuda_callable__
auto
operator-( const ET1& a, const ET2& b )
{
   return StaticBinaryExpressionTemplate< ET1, ET2, Subtraction >( a, b );
}

////
// Binary expression multiplication
template< typename ET1, typename ET2,
          typename..., typename = EnableIfStaticBinaryExpression_t< ET1, ET2 > >
__cuda_callable__
auto
operator*( const ET1& a, const ET2& b )
{
   return StaticBinaryExpressionTemplate< ET1, ET2, Multiplication >( a, b );
}

////
// Binary expression division
template< typename ET1, typename ET2,
          typename..., typename = EnableIfStaticBinaryExpression_t< ET1, ET2 > >
__cuda_callable__
auto
operator/( const ET1& a, const ET2& b )
{
   return StaticBinaryExpressionTemplate< ET1, ET2, Division >( a, b );
}

////
// Comparison operator ==
template< typename ET1, typename ET2,
          typename..., typename = EnableIfStaticBinaryExpression_t< ET1, ET2 > >
__cuda_callable__
bool
operator==( const ET1& a, const ET2& b )
{
   return StaticComparison< ET1, ET2 >::EQ( a, b );
}

////
// Comparison operator !=
template< typename ET1, typename ET2,
          typename..., typename = EnableIfStaticBinaryExpression_t< ET1, ET2 > >
__cuda_callable__
bool
operator!=( const ET1& a, const ET2& b )
{
   return StaticComparison< ET1, ET2 >::NE( a, b );
}

////
// Comparison operator <
template< typename ET1, typename ET2,
          typename..., typename = EnableIfStaticBinaryExpression_t< ET1, ET2 > >
__cuda_callable__
bool
operator<( const ET1& a, const ET2& b )
{
   return StaticComparison< ET1, ET2 >::LT( a, b );
}

////
// Comparison operator <=
template< typename ET1, typename ET2,
          typename..., typename = EnableIfStaticBinaryExpression_t< ET1, ET2 > >
__cuda_callable__
bool
operator<=( const ET1& a, const ET2& b )
{
   return StaticComparison< ET1, ET2 >::LE( a, b );
}

////
// Comparison operator >
template< typename ET1, typename ET2,
          typename..., typename = EnableIfStaticBinaryExpression_t< ET1, ET2 > >
__cuda_callable__
bool
operator>( const ET1& a, const ET2& b )
{
   return StaticComparison< ET1, ET2 >::GT( a, b );
}

////
// Comparison operator >=
template< typename ET1, typename ET2,
          typename..., typename = EnableIfStaticBinaryExpression_t< ET1, ET2 > >
__cuda_callable__
bool
operator>=( const ET1& a, const ET2& b )
{
   return Expressions::StaticComparison< ET1, ET2 >::GE( a, b );
}

////
// Scalar product
template< typename ET1, typename ET2,
          typename..., typename = EnableIfStaticBinaryExpression_t< ET1, ET2 > >
__cuda_callable__
auto
operator,( const ET1& a, const ET2& b )
{
   return StaticExpressionSum( a * b );
}

template< typename ET1, typename ET2,
          typename..., typename = EnableIfStaticBinaryExpression_t< ET1, ET2 > >
__cuda_callable__
auto
dot( const ET1& a, const ET2& b )
{
   return (a, b);
}

////
// Unary expression minus
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
operator-( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Minus >( a );
}

////
// Binary expression min
template< typename ET1, typename ET2,
          typename..., typename = EnableIfStaticBinaryExpression_t< ET1, ET2 > >
__cuda_callable__
auto
min( const ET1& a, const ET2& b )
{
   return StaticBinaryExpressionTemplate< ET1, ET2, Min >( a, b );
}

////
// Binary expression max
template< typename ET1, typename ET2,
          typename..., typename = EnableIfStaticBinaryExpression_t< ET1, ET2 > >
__cuda_callable__
auto
max( const ET1& a, const ET2& b )
{
   return StaticBinaryExpressionTemplate< ET1, ET2, Max >( a, b );
}

////
// Abs
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
abs( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Abs >( a );
}

////
// Pow
template< typename ET1, typename Real,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
pow( const ET1& a, const Real& exp )
{
   return StaticBinaryExpressionTemplate< ET1, Real, Pow >( a, exp );
}

////
// Exp
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
exp( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Exp >( a );
}

////
// Sqrt
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
sqrt( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Sqrt >( a );
}

////
// Cbrt
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
cbrt( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Cbrt >( a );
}

////
// Log
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
log( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Log >( a );
}

////
// Log10
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
log10( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Log10 >( a );
}

////
// Log2
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
log2( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Log2 >( a );
}

////
// Sin
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
sin( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Sin >( a );
}

////
// Cos
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
cos( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Cos >( a );
}

////
// Tan
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
tan( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Tan >( a );
}

////
// Asin
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
asin( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Asin >( a );
}

////
// Acos
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
acos( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Acos >( a );
}

////
// Atan
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
atan( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Atan >( a );
}

////
// Sinh
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
sinh( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Sinh >( a );
}

////
// Cosh
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
cosh( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Cosh >( a );
}

////
// Tanh
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
tanh( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Tanh >( a );
}

////
// Asinh
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
asinh( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Asinh >( a );
}

////
// Acosh
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
acosh( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Acosh >( a );
}

////
// Atanh
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
atanh( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Atanh >( a );
}

////
// Floor
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
floor( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Floor >( a );
}

////
// Ceil
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
ceil( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Ceil >( a );
}

////
// Sign
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
sign( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, Sign >( a );
}

////
// Cast
template< typename ResultType,
          typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 >,
          // workaround: templated type alias cannot be declared at block level
          typename CastOperation = typename Cast< ResultType >::Operation >
__cuda_callable__
auto
cast( const ET1& a )
{
   return StaticUnaryExpressionTemplate< ET1, CastOperation >( a );
}

////
// Vertical operations
template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
min( const ET1& a )
{
   return StaticExpressionMin( a );
}

template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
argMin( const ET1& a )
{
   return StaticExpressionArgMin( a );
}

template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
max( const ET1& a )
{
   return StaticExpressionMax( a );
}

template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
argMax( const ET1& a )
{
   return StaticExpressionArgMax( a );
}

template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
sum( const ET1& a )
{
   return StaticExpressionSum( a );
}

template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
maxNorm( const ET1& a )
{
   return max( abs( a ) );
}

template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
l1Norm( const ET1& a )
{
   return sum( abs( a ) );
}

template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 >,
          std::enable_if_t< (ET1::getSize() > 1), bool > = true >
__cuda_callable__
auto
l2Norm( const ET1& a )
{
   using TNL::sqrt;
   return sqrt( sum( a * a ) );
}

template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 >,
          std::enable_if_t< ET1::getSize() == 1, bool > = true >
__cuda_callable__
auto
l2Norm( const ET1& a )
{
   // avoid sqrt for 1D vectors (l1 and l2 norms are identical in 1D)
   return l1Norm( a );
}

template< typename ET1,
          typename Real,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 >,
          std::enable_if_t< (ET1::getSize() > 1), bool > = true >
__cuda_callable__
auto
lpNorm( const ET1& a, const Real& p )
// since (1.0 / p) has type double, TNL::pow returns double
-> double
//-> RemoveET< decltype(pow( StaticExpressionLpNorm( a, p ), 1.0 / p )) >
{
   if( p == 1.0 )
      return l1Norm( a );
   if( p == 2.0 )
      return l2Norm( a );
   using TNL::pow;
   return pow( sum( pow( abs( a ), p ) ), 1.0 / p );
}

template< typename ET1,
          typename Real,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 >,
          std::enable_if_t< ET1::getSize() == 1, bool > = true >
__cuda_callable__
auto
lpNorm( const ET1& a, const Real& p )
{
   // avoid sqrt and pow for 1D vectors (all lp norms are identical in 1D)
   return l1Norm( a );
}

template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
product( const ET1& a )
{
   return StaticExpressionProduct( a );
}

template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
logicalAnd( const ET1& a )
{
   return StaticExpressionLogicalAnd( a );
}

template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
logicalOr( const ET1& a )
{
   return StaticExpressionLogicalOr( a );
}

template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
binaryAnd( const ET1& a )
{
   return StaticExpressionBinaryAnd( a );
}

template< typename ET1,
          typename..., typename = EnableIfStaticUnaryExpression_t< ET1 > >
__cuda_callable__
auto
binaryOr( const ET1& a )
{
   return StaticExpressionBinaryOr( a );
}

#endif // DOXYGEN_ONLY

////
// Output stream
template< typename T1,
          typename T2,
          typename Operation >
std::ostream& operator<<( std::ostream& str, const StaticBinaryExpressionTemplate< T1, T2, Operation >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression[ i ] << ", ";
   str << expression[ expression.getSize() - 1 ] << " ]";
   return str;
}

template< typename T,
          typename Operation >
std::ostream& operator<<( std::ostream& str, const StaticUnaryExpressionTemplate< T, Operation >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression[ i ] << ", ";
   str << expression[ expression.getSize() - 1 ] << " ]";
   return str;
}

} // namespace Expressions

// Make all operators visible in the TNL::Containers namespace to be considered
// even for StaticVector
using Expressions::operator+;
using Expressions::operator-;
using Expressions::operator*;
using Expressions::operator/;
using Expressions::operator,;
using Expressions::operator==;
using Expressions::operator!=;
using Expressions::operator<;
using Expressions::operator<=;
using Expressions::operator>;
using Expressions::operator>=;

// Make all functions visible in the TNL::Containers namespace
using Expressions::dot;
using Expressions::min;
using Expressions::max;
using Expressions::abs;
using Expressions::pow;
using Expressions::exp;
using Expressions::sqrt;
using Expressions::cbrt;
using Expressions::log;
using Expressions::log10;
using Expressions::log2;
using Expressions::sin;
using Expressions::cos;
using Expressions::tan;
using Expressions::asin;
using Expressions::acos;
using Expressions::atan;
using Expressions::sinh;
using Expressions::cosh;
using Expressions::tanh;
using Expressions::asinh;
using Expressions::acosh;
using Expressions::atanh;
using Expressions::floor;
using Expressions::ceil;
using Expressions::sign;
using Expressions::cast;
using Expressions::argMin;
using Expressions::argMax;
using Expressions::sum;
using Expressions::maxNorm;
using Expressions::l1Norm;
using Expressions::l2Norm;
using Expressions::lpNorm;
using Expressions::product;
using Expressions::logicalAnd;
using Expressions::logicalOr;
using Expressions::binaryAnd;
using Expressions::binaryOr;

} // namespace Containers

// Make all functions visible in the main TNL namespace
using Containers::dot;
using Containers::min;
using Containers::max;
using Containers::abs;
using Containers::pow;
using Containers::exp;
using Containers::sqrt;
using Containers::cbrt;
using Containers::log;
using Containers::log10;
using Containers::log2;
using Containers::sin;
using Containers::cos;
using Containers::tan;
using Containers::asin;
using Containers::acos;
using Containers::atan;
using Containers::sinh;
using Containers::cosh;
using Containers::tanh;
using Containers::asinh;
using Containers::acosh;
using Containers::atanh;
using Containers::floor;
using Containers::ceil;
using Containers::sign;
using Containers::cast;
using Containers::argMin;
using Containers::argMax;
using Containers::sum;
using Containers::maxNorm;
using Containers::l1Norm;
using Containers::l2Norm;
using Containers::lpNorm;
using Containers::product;
using Containers::logicalAnd;
using Containers::logicalOr;
using Containers::binaryAnd;
using Containers::binaryOr;

////
// Evaluation with reduction
template< typename Vector,
   typename T1,
   typename T2,
   typename Operation,
   typename Reduction,
   typename Result >
__cuda_callable__
Result evaluateAndReduce( Vector& lhs,
   const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& expression,
   const Reduction& reduction,
   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ )
      result = reduction( result, lhs[ i ] = expression[ i ] );
   return result;
}

template< typename Vector,
   typename T1,
   typename Operation,
   typename Reduction,
   typename Result >
__cuda_callable__
Result evaluateAndReduce( Vector& lhs,
   const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation >& expression,
   const Reduction& reduction,
   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ )
      result = reduction( result, lhs[ i ] = expression[ i ] );
   return result;
}

////
// Addition with reduction
template< typename Vector,
   typename T1,
   typename T2,
   typename Operation,
   typename Reduction,
   typename Result >
__cuda_callable__
Result addAndReduce( Vector& lhs,
   const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& expression,
   const Reduction& reduction,
   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ ) {
      const Result aux = expression[ i ];
      lhs[ i ] += aux;
      result = reduction( result, aux );
   }
   return result;
}

template< typename Vector,
   typename T1,
   typename Operation,
   typename Reduction,
   typename Result >
__cuda_callable__
Result addAndReduce( Vector& lhs,
   const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation >& expression,
   const Reduction& reduction,
   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ ) {
      const Result aux = expression[ i ];
      lhs[ i ] += aux;
      result = reduction( result, aux );
   }
   return result;
}

////
// Addition with reduction of abs
template< typename Vector,
   typename T1,
   typename T2,
   typename Operation,
   typename Reduction,
   typename Result >
__cuda_callable__
Result addAndReduceAbs( Vector& lhs,
   const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& expression,
   const Reduction& reduction,
   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ ) {
      const Result aux = expression[ i ];
      lhs[ i ] += aux;
      result = reduction( result, TNL::abs( aux ) );
   }
   return result;
}

template< typename Vector,
   typename T1,
   typename Operation,
   typename Reduction,
   typename Result >
__cuda_callable__
Result addAndReduceAbs( Vector& lhs,
   const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation >& expression,
   const Reduction& reduction,
   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ ) {
      const Result aux = expression[ i ];
      lhs[ i ] += aux;
      result = reduction( result, TNL::abs( aux ) );
   }
   return result;
}

} // namespace TNL
