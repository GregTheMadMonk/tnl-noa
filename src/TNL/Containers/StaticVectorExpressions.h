/***************************************************************************
                          Containers::StaticVectorExpressions.h  -  description
                             -------------------
    begin                : Apr 19, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Expressions/StaticExpressionTemplates.h>
#include <TNL/Containers/Expressions/ExpressionTemplatesOperations.h>
#include <TNL/Containers/Expressions/StaticComparison.h>
#include <TNL/Containers/Expressions/StaticVerticalOperations.h>

namespace TNL {

// operators must be defined in the same namespace as the first operand, otherwise
// they may not be considered by the compiler (e.g. inside pybind11's macros)
namespace Containers {

////
// Addition
template< int Size, typename Real, typename ET >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Addition >
operator+( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Addition >( a, b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Addition >
operator+( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticBinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Addition >( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Addition >
operator+( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Addition >( a, b );
}

////
// Subtraction
template< int Size, typename Real, typename ET >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Subtraction >
operator-( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Subtraction >( a, b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Subtraction >
operator-( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticBinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Subtraction >( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Subtraction >
operator-( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Subtraction >( a, b );
}

////
// Multiplication
template< int Size, typename Real, typename ET >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Multiplication >
operator*( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Multiplication >( a, b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Multiplication >
operator*( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticBinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Multiplication >( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Multiplication >
operator*( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Multiplication >( a, b );
}

////
// Division
template< int Size, typename Real, typename ET >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Division >
operator/( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Division >( a, b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Division >
operator/( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticBinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Division >( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Division >
operator/( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticBinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Division >( a, b );
}

////
// Comparison operations - operator ==
template< int Size, typename Real, typename ET >
__cuda_callable__
bool operator==( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparison< StaticVector< Size, Real >, ET >::EQ( a, b );
}

template< typename ET, int Size, typename Real >
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
template< int Size, typename Real, typename ET >
__cuda_callable__
bool operator!=( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparison< StaticVector< Size, Real >, ET >::NE( a, b );
}

template< typename ET, int Size, typename Real >
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
template< int Size, typename Real, typename ET >
__cuda_callable__
bool operator<( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparison< StaticVector< Size, Real >, ET >::LT( a, b );
}

template< typename ET, int Size, typename Real >
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
template< int Size, typename Real, typename ET >
__cuda_callable__
bool operator<=( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparison< StaticVector< Size, Real >, ET >::LE( a, b );
}

template< typename ET, int Size, typename Real >
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
template< int Size, typename Real, typename ET >
__cuda_callable__
bool operator>( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparison< StaticVector< Size, Real >, ET >::GT( a, b );
}

template< typename ET, int Size, typename Real >
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
template< int Size, typename Real, typename ET >
__cuda_callable__
bool operator>=( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparison< StaticVector< Size, Real >, ET >::GE( a, b );
}

template< typename ET, int Size, typename Real >
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
const Expressions::StaticUnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Minus >
operator-( const StaticVector< Size, Real >& a )
{
   return Expressions::StaticUnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Minus >( a );
}

////
// Scalar product
template< int Size, typename Real, typename ET >
__cuda_callable__
auto operator,( const StaticVector< Size, Real >& a, const ET& b )
->decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
auto operator,( const ET& a, const StaticVector< Size, Real >& b )
->decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename Real1, int Size, typename Real2 >
__cuda_callable__
auto operator,( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
->decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

} // namespace Containers

////
// Min
template< int Size, typename Real, typename ET >
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate< Containers::StaticVector< Size, Real >, ET, Containers::Expressions::Min >
min( const Containers::StaticVector< Size, Real >& a, const ET& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< Containers::StaticVector< Size, Real >, ET, Containers::Expressions::Min >( a, b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate< ET, Containers::StaticVector< Size, Real >, Containers::Expressions::Min >
min( const ET& a, const Containers::StaticVector< Size, Real >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< ET, Containers::StaticVector< Size, Real >, Containers::Expressions::Min >( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate< Containers::StaticVector< Size, Real1 >, Containers::StaticVector< Size, Real2 >, Containers::Expressions::Min >
min( const Containers::StaticVector< Size, Real1 >& a, const Containers::StaticVector< Size, Real2 >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< Containers::StaticVector< Size, Real1 >, Containers::StaticVector< Size, Real2 >, Containers::Expressions::Min >( a, b );
}

////
// Max
template< int Size, typename Real, typename ET >
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate< Containers::StaticVector< Size, Real >, ET, Containers::Expressions::Max >
max( const Containers::StaticVector< Size, Real >& a, const ET& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< Containers::StaticVector< Size, Real >, ET, Containers::Expressions::Max >( a, b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate< ET, Containers::StaticVector< Size, Real >, Containers::Expressions::Max >
max( const ET& a, const Containers::StaticVector< Size, Real >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< ET, Containers::StaticVector< Size, Real >, Containers::Expressions::Max >( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate< Containers::StaticVector< Size, Real1 >, Containers::StaticVector< Size, Real2 >, Containers::Expressions::Max >
max( const Containers::StaticVector< Size, Real1 >& a, const Containers::StaticVector< Size, Real2 >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate< Containers::StaticVector< Size, Real1 >, Containers::StaticVector< Size, Real2 >, Containers::Expressions::Max >( a, b );
}

////
// Abs
template< int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Abs >
abs( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Abs >( a );
}

////
// Sine
template< int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Sin >
sin( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Sin >( a );
}

////
// Cosine
template< int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Cos >
cos( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Cos >( a );
}

////
// Tangent
template< int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Tan >
tan( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Tan >( a );
}

////
// Sqrt
template< int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Sqrt >
sqrt( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Sqrt >( a );
}

////
// Cbrt
template< int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Cbrt >
cbrt( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Cbrt >( a );
}

////
// Power
template< int Size, typename Real, typename ExpType >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Pow, ExpType >
pow( const Containers::StaticVector< Size, Real >& a, const ExpType& exp )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Pow, ExpType >( a, exp );
}

////
// Floor
template< int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Floor >
floor( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Floor >( a );
}

////
// Ceil
template< int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Ceil >
ceil( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Ceil >( a );
}

////
// Acos
template< int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Acos >
acos( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Acos >( a );
}

////
// Asin
template< int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Asin >
asin( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Asin >( a );
}

////
// Atan
template< int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Atan >
atan( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Atan >( a );
}

////
// Cosh
template< int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Cosh >
cosh( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Cosh >( a );
}

////
// Tanh
template< int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Tanh >
tanh( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Tanh >( a );
}

////
// Log
template< int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Log >
log( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Log >( a );
}

////
// Log10
template< int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Log10 >
log10( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Log10 >( a );
}

////
// Log2
template< int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Log2 >
log2( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Log2 >( a );
}

////
// Exp
template< int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Exp >
exp( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Exp >( a );
}

////
// Sign
template< int Size, typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Sign >
sign( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate< Containers::StaticVector< Size, Real >, Containers::Expressions::Sign >( a );
}

////
// Vertical operations - min
template< int Size, typename Real >
__cuda_callable__
auto
min( const Containers::StaticVector< Size, Real >& a )
-> decltype( Containers::Expressions::StaticExpressionMin( a ) )
{
   return Containers::Expressions::StaticExpressionMin( a );
}

template< int Size, typename Real >
__cuda_callable__
auto
argMin( const Containers::StaticVector< Size, Real >& a, int& arg )
-> decltype( Containers::Expressions::StaticExpressionArgMin( a, arg ) )
{
   return Containers::Expressions::StaticExpressionArgMin( a, arg );
}

template< int Size, typename Real >
__cuda_callable__
auto
max( const Containers::StaticVector< Size, Real >& a )
-> decltype( Containers::Expressions::StaticExpressionMax( a ) )
{
   return Containers::Expressions::StaticExpressionMax( a );
}

template< int Size, typename Real >
__cuda_callable__
auto
argMax( const Containers::StaticVector< Size, Real >& a, int& arg )
-> decltype( Containers::Expressions::StaticExpressionArgMax( a, arg ) )
{
   return Containers::Expressions::StaticExpressionArgMax( a, arg );
}

template< int Size, typename Real >
__cuda_callable__
auto
sum( const Containers::StaticVector< Size, Real >& a )
-> decltype( Containers::Expressions::StaticExpressionSum( a ) )
{
   return Containers::Expressions::StaticExpressionSum( a );
}

template< int Size, typename Real, typename Real2 >
__cuda_callable__
auto
lpNorm( const Containers::StaticVector< Size, Real >& a, const Real2& p )
-> decltype( Containers::Expressions::StaticExpressionLpNorm( a, p ) )
{
   return Containers::Expressions::StaticExpressionLpNorm( a, p );
}

template< int Size, typename Real >
__cuda_callable__
auto
product( const Containers::StaticVector< Size, Real >& a )
-> decltype( Containers::Expressions::StaticExpressionProduct( a ) )
{
   return Containers::Expressions::StaticExpressionProduct( a );
}

template< int Size, typename Real >
__cuda_callable__
bool
logicalOr( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticExpressionLogicalOr( a );
}

template< int Size, typename Real >
__cuda_callable__
auto
binaryOr( const Containers::StaticVector< Size, Real >& a )
-> decltype( Containers::Expressions::StaticExpressionBinaryOr( a ) )
{
   return Containers::Expressions::StaticExpressionBinaryOr( a );
}

template< int Size, typename Real >
__cuda_callable__
bool
logicalAnd( const Containers::StaticVector< Size, Real >& a )
{
   return Containers::Expressions::StaticExpressionLogicalAnd( a );
}

template< int Size, typename Real >
__cuda_callable__
auto
binaryAnd( const Containers::StaticVector< Size, Real >& a )
-> decltype( Containers::Expressions::StaticExpressionBinaryAnd( a ) )
{
   return Containers::Expressions::StaticExpressionBinaryAnd( a );
}

////
// Dot product - the same as scalar product, just for convenience
template< int Size, typename Real, typename ET >
__cuda_callable__
auto dot( const Containers::StaticVector< Size, Real >& a, const ET& b )
->decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
auto dot( const ET& a, const Containers::StaticVector< Size, Real >& b )
->decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename Real1, int Size, typename Real2 >
__cuda_callable__
auto dot( const Containers::StaticVector< Size, Real1 >& a, const Containers::StaticVector< Size, Real2 >& b )
->decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

////
// TODO: Replace this with multiplication when its safe
template< int Size, typename Real, typename ET >
__cuda_callable__
Containers::StaticVector< Size, Real >
Scale( const Containers::StaticVector< Size, Real >& a, const ET& b )
{
   Containers::StaticVector< Size, Real > result = Containers::Expressions::StaticBinaryExpressionTemplate< Containers::StaticVector< Size, Real >, ET, Containers::Expressions::Multiplication >( a, b );
   return result;
}

template< typename ET, int Size, typename Real >
__cuda_callable__
Containers::Expressions::StaticBinaryExpressionTemplate< ET, Containers::StaticVector< Size, Real >, Containers::Expressions::Multiplication >
Scale( const ET& a, const Containers::StaticVector< Size, Real >& b )
{
   Containers::StaticVector< Size, Real > result =  Containers::Expressions::StaticBinaryExpressionTemplate< ET, Containers::StaticVector< Size, Real >, Containers::Expressions::Multiplication >( a, b );
   return result;
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
Containers::Expressions::StaticBinaryExpressionTemplate< Containers::StaticVector< Size, Real1 >, Containers::StaticVector< Size, Real2 >, Containers::Expressions::Multiplication >
Scale( const Containers::StaticVector< Size, Real1 >& a, const Containers::StaticVector< Size, Real2 >& b )
{
   Containers::StaticVector< Size, Real1 > result =  Containers::Expressions::StaticBinaryExpressionTemplate< Containers::StaticVector< Size, Real1 >, Containers::StaticVector< Size, Real2 >, Containers::Expressions::Multiplication >( a, b );
   return result;
}

} // namespace TNL
