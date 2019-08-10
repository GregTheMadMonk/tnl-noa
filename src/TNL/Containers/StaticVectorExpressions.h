/***************************************************************************
                          StaticVectorExpressions.h  -  description
                             -------------------
    begin                : Apr 19, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Expressions/StaticExpressionTemplates.h>

#include "StaticVector.h"

namespace TNL {
namespace Containers {
// operators must be defined in the same namespace as the first operand, otherwise
// they may not be considered by the compiler (e.g. inside pybind11's macros)
// namespace Containers { Overriden operators should be in namespace TNL

////
// Addition
template< int Size, typename Real, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
auto
operator+( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Addition >( a, b );
}

template< typename ET, int Size, typename Real,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
auto
operator+( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticBinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Addition >( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
auto
operator+( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Addition >( a, b );
}

////
// Subtraction
template< int Size, typename Real, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
auto
operator-( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Subtraction >( a, b );
}

template< typename ET, int Size, typename Real,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
auto
operator-( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticBinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Subtraction >( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
auto
operator-( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Subtraction >( a, b );
}

////
// Multiplication
template< int Size, typename Real, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
auto
operator*( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Multiplication >( a, b );
}

template< typename ET, int Size, typename Real,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
auto
operator*( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticBinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Multiplication >( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
auto
operator*( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Multiplication >( a, b );
}

////
// Division
template< int Size, typename Real, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
auto
operator/( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Division >( a, b );
}

template< typename ET, int Size, typename Real,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
auto
operator/( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticBinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Division >( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
auto
operator/( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Division >( a, b );
}

////
// Comparison operations - operator ==
template< int Size, typename Real, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
bool operator==( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparison< StaticVector< Size, Real >, ET >::EQ( a, b );
}

template< typename ET, int Size, typename Real,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
bool operator==( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticComparison< ET, StaticVector< Size, Real > >::EQ( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
bool operator==( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticComparison< StaticVector< Size, Real1 >, StaticVector< Size, Real2 > >::EQ( a, b );
}

////
// Comparison operations - operator !=
template< int Size, typename Real, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
bool operator!=( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparison< StaticVector< Size, Real >, ET >::NE( a, b );
}

template< typename ET, int Size, typename Real,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
bool operator!=( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticComparison< ET, StaticVector< Size, Real > >::NE( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
bool operator!=( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticComparison< StaticVector< Size, Real1 >, StaticVector< Size, Real2 > >::NE( a, b );
}

////
// Comparison operations - operator <
template< int Size, typename Real, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
bool operator<( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparison< StaticVector< Size, Real >, ET >::LT( a, b );
}

template< typename ET, int Size, typename Real,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
bool operator<( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticComparison< ET, StaticVector< Size, Real > >::LT( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
bool operator<( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticComparison< StaticVector< Size, Real1 >, StaticVector< Size, Real2 > >::LT( a, b );
}

////
// Comparison operations - operator <=
template< int Size, typename Real, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
bool operator<=( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparison< StaticVector< Size, Real >, ET >::LE( a, b );
}

template< typename ET, int Size, typename Real,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
bool operator<=( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticComparison< ET, StaticVector< Size, Real > >::LE( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
bool operator<=( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticComparison< StaticVector< Size, Real1 >, StaticVector< Size, Real2 > >::LE( a, b );
}

////
// Comparison operations - operator >
template< int Size, typename Real, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
bool operator>( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparison< StaticVector< Size, Real >, ET >::GT( a, b );
}

template< typename ET, int Size, typename Real,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
bool operator>( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticComparison< ET, StaticVector< Size, Real > >::GT( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
bool operator>( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticComparison< StaticVector< Size, Real1 >, StaticVector< Size, Real2 > >::GT( a, b );
}

////
// Comparison operations - operator >=
template< int Size, typename Real, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
bool operator>=( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparison< StaticVector< Size, Real >, ET >::GE( a, b );
}

template< typename ET, int Size, typename Real,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
bool operator>=( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticComparison< ET, StaticVector< Size, Real > >::GE( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
bool operator>=( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticComparison< StaticVector< Size, Real1 >, StaticVector< Size, Real2 > >::GE( a, b );
}

////
// Minus
template< int Size, typename Real >
__cuda_callable__
auto
operator-( const StaticVector< Size, Real >& a )
{
   return Expressions::StaticUnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Minus >( a );
}

////
// Scalar product
template< int Size, typename Real, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
auto
operator,( const StaticVector< Size, Real >& a, const ET& b )
{
   return Containers::Expressions::StaticExpressionSum( a * b );
}

template< typename ET, int Size, typename Real,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
auto
operator,( const ET& a, const StaticVector< Size, Real >& b )
{
   return Containers::Expressions::StaticExpressionSum( a * b );
}

template< typename Real1, int Size, typename Real2 >
__cuda_callable__
auto
operator,( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Containers::Expressions::StaticExpressionSum( a * b );
}

} // namespace Containers

////
// Min
template< int Size, typename Real, typename ET,
          typename..., typename = std::enable_if_t< Containers::Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
auto
min( const Containers::StaticVector< Size, Real >& a, const ET& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< Containers::StaticVector< Size, Real >, ET, Containers::Expressions::Min >( a, b );
}

template< typename ET, int Size, typename Real,
          typename..., typename = std::enable_if_t< Containers::Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
auto
min( const ET& a, const Containers::StaticVector< Size, Real >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< ET, Containers::StaticVector< Size, Real >, Containers::Expressions::Min >( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
auto
min( const Containers::StaticVector< Size, Real1 >& a, const Containers::StaticVector< Size, Real2 >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< Containers::StaticVector< Size, Real1 >, Containers::StaticVector< Size, Real2 >, Containers::Expressions::Min >( a, b );
}

////
// Max
template< int Size, typename Real, typename ET,
          typename..., typename = std::enable_if_t< Containers::Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
auto
max( const Containers::StaticVector< Size, Real >& a, const ET& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< Containers::StaticVector< Size, Real >, ET, Containers::Expressions::Max >( a, b );
}

template< typename ET, int Size, typename Real,
          typename..., typename = std::enable_if_t< Containers::Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
auto
max( const ET& a, const Containers::StaticVector< Size, Real >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< ET, Containers::StaticVector< Size, Real >, Containers::Expressions::Max >( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
auto
max( const Containers::StaticVector< Size, Real1 >& a, const Containers::StaticVector< Size, Real2 >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< Containers::StaticVector< Size, Real1 >, Containers::StaticVector< Size, Real2 >, Containers::Expressions::Max >( a, b );
}

////
// Dot product - the same as scalar product, just for convenience
template< int Size, typename Real, typename ET,
          typename..., typename = std::enable_if_t< Containers::Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
auto
dot( const Containers::StaticVector< Size, Real >& a, const ET& b )
{
   return (a, b);
}

template< typename ET, int Size, typename Real,
          typename..., typename = std::enable_if_t< Containers::Expressions::IsNumericExpression<ET>::value > >
__cuda_callable__
auto
dot( const ET& a, const Containers::StaticVector< Size, Real >& b )
{
   return (a, b);
}

template< typename Real1, int Size, typename Real2 >
__cuda_callable__
auto
dot( const Containers::StaticVector< Size, Real1 >& a, const Containers::StaticVector< Size, Real2 >& b )
{
   return (a, b);
}


////
// Abs
template< int Size, typename Real >
__cuda_callable__
auto
abs( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Abs >( a );
}

////
// Power
template< int Size, typename Real, typename ExpType >
__cuda_callable__
auto
pow( const Containers::StaticVector< Size, Real >& a, const ExpType& exp )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< Containers::StaticVector< Size, Real >, ExpType, Containers::Expressions::Pow >( a, exp );
}

////
// Exp
template< int Size, typename Real >
__cuda_callable__
auto
exp( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Exp >( a );
}

////
// Sqrt
template< int Size, typename Real >
__cuda_callable__
auto
sqrt( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Sqrt >( a );
}

////
// Cbrt
template< int Size, typename Real >
__cuda_callable__
auto
cbrt( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Cbrt >( a );
}

////
// Log
template< int Size, typename Real >
__cuda_callable__
auto
log( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Log >( a );
}

////
// Log10
template< int Size, typename Real >
__cuda_callable__
auto
log10( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Log10 >( a );
}

////
// Log2
template< int Size, typename Real >
__cuda_callable__
auto
log2( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Log2 >( a );
}

////
// Sine
template< int Size, typename Real >
__cuda_callable__
auto
sin( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Sin >( a );
}

////
// Cosine
template< int Size, typename Real >
__cuda_callable__
auto
cos( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Cos >( a );
}

////
// Tangent
template< int Size, typename Real >
__cuda_callable__
auto
tan( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Tan >( a );
}

////
// Asin
template< int Size, typename Real >
__cuda_callable__
auto
asin( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Asin >( a );
}

////
// Acos
template< int Size, typename Real >
__cuda_callable__
auto
acos( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Acos >( a );
}

////
// Atan
template< int Size, typename Real >
__cuda_callable__
auto
atan( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Atan >( a );
}

////
// Sinh
template< int Size, typename Real >
__cuda_callable__
auto
sinh( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Sinh >( a );
}

////
// Cosh
template< int Size, typename Real >
__cuda_callable__
auto
cosh( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Cosh >( a );
}

////
// Tanh
template< int Size, typename Real >
__cuda_callable__
auto
tanh( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Tanh >( a );
}

////
// Asinh
template< int Size, typename Real >
__cuda_callable__
auto
asinh( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Asinh >( a );
}

////
// Acosh
template< int Size, typename Real >
__cuda_callable__
auto
acosh( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Acosh >( a );
}

////
// Atanh
template< int Size, typename Real >
__cuda_callable__
auto
atanh( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Atanh >( a );
}

////
// Floor
template< int Size, typename Real >
__cuda_callable__
auto
floor( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Floor >( a );
}

////
// Ceil
template< int Size, typename Real >
__cuda_callable__
auto
ceil( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Ceil >( a );
}

////
// Sign
template< int Size, typename Real >
__cuda_callable__
auto
sign( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Sign >( a );
}

////
// Cast
template< typename ResultType, int Size, typename Real,
          // workaround: templated type alias cannot be declared at block level
          template<typename> class Operation = Containers::Expressions::Cast< ResultType >::template Operation >
auto
__cuda_callable__
cast( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< std::decay_t<decltype(a)>, Operation >( a );
}

////
// Vertical operations - min
template< int Size, typename Real >
__cuda_callable__
auto
min( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticExpressionMin( a );
}

template< int Size, typename Real >
__cuda_callable__
std::pair< int, Real >
argMin( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticExpressionArgMin( a );
}

template< int Size, typename Real >
__cuda_callable__
auto
max( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticExpressionMax( a );
}

template< int Size, typename Real >
__cuda_callable__
std::pair< int, Real >
argMax( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticExpressionArgMax( a );
}

template< int Size, typename Real >
__cuda_callable__
auto
sum( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticExpressionSum( a );
}

template< int Size, typename Real >
__cuda_callable__
auto
maxNorm( const Containers::StaticVector< Size, Real >& a )
{
   return max( abs( a ) );
}

template< int Size, typename Real >
__cuda_callable__
auto
l1Norm( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticExpressionL1Norm( a );
}

template< int Size, typename Real >
__cuda_callable__
auto
l2Norm( const Containers::StaticVector< Size, Real >& a )
{
   return TNL::sqrt( Containers::Expressions::StaticExpressionL2Norm( a ) );
}

template< int Size, typename Real, typename Real2 >
__cuda_callable__
auto
lpNorm( const Containers::StaticVector< Size, Real >& a, const Real2& p )
// since (1.0 / p) has type double, TNL::pow returns double
-> double
{
   if( p == 1.0 )
      return l1Norm( a );
   if( p == 2.0 )
      return l2Norm( a );
   return TNL::pow( Containers::Expressions::StaticExpressionLpNorm( a, p ), 1.0 / p );
}

template< int Size, typename Real >
__cuda_callable__
auto
product( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticExpressionProduct( a );
}

template< int Size, typename Real >
__cuda_callable__
auto
logicalOr( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticExpressionLogicalOr( a );
}

template< int Size, typename Real >
__cuda_callable__
auto
binaryOr( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticExpressionBinaryOr( a );
}

template< int Size, typename Real >
__cuda_callable__
auto
logicalAnd( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticExpressionLogicalAnd( a );
}

template< int Size, typename Real >
__cuda_callable__
auto
binaryAnd( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticExpressionBinaryAnd( a );
}

} // namespace TNL
