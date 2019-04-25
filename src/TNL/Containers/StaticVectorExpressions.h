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
#include <TNL/Containers/Expressions/ExpressionTemplatesOperations.h>
#include <TNL/Containers/Expressions/StaticComparison.h>

#include "Expressions/StaticComparison.h"

namespace TNL {
   namespace Containers {

////
// Addition
template< int Size, typename Real, typename ET >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Addition >
operator+( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Addition >( a, b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Addition >
operator+( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Addition >( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Addition >
operator+( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Addition >( a, b );
}

////
// Subtraction
template< int Size, typename Real, typename ET >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Subtraction >
operator-( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Subtraction >( a, b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Subtraction >
operator-( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Subtraction >( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Subtraction >
operator-( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Subtraction >( a, b );
}

////
// Multiplication
template< int Size, typename Real, typename ET >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Multiplication >
operator*( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Multiplication >( a, b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Multiplication >
operator*( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Multiplication >( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Multiplication >
operator*( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Multiplication >( a, b );
}

////
// Division
template< int Size, typename Real, typename ET >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Division >
operator/( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Division >( a, b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Division >
operator/( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Division >( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Division >
operator/( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Division >( a, b );
}

////
// Min
template< int Size, typename Real, typename ET >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Min >
min( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Min >( a, b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Min >
min( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Min >( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Min >
min( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Min >( a, b );
}

////
// Max
template< int Size, typename Real, typename ET >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Max >
max( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Max >( a, b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Max >
max( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Max >( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Max >
max( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Max >( a, b );
}

////
// Comparison operations - operator ==
template< int Size, typename Real, typename ET >
__cuda_callable__
bool operator==( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparisonEQ( a, b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
bool operator==( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticComparisonEQ( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
bool operator==( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticComparisonEQ( a, b );
}

////
// Comparison operations - operator !=
template< int Size, typename Real, typename ET >
__cuda_callable__
bool operator!=( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparisonNE( a, b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
bool operator!=( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticComparisonNE( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
bool operator!=( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticComparisonNE( a, b );
}

////
// Comparison operations - operator <
template< int Size, typename Real, typename ET >
__cuda_callable__
bool operator<( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparisonLT( a, b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
bool operator<( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticComparisonLT( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
bool operator<( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticComparisonLT( a, b );
}

////
// Comparison operations - operator <=
template< int Size, typename Real, typename ET >
__cuda_callable__
bool operator<=( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparisonLE( a, b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
bool operator<=( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticComparisonLE( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
bool operator<=( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticComparisonLE( a, b );
}

////
// Comparison operations - operator >
template< int Size, typename Real, typename ET >
__cuda_callable__
bool operator>( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparisonGT( a, b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
bool operator>( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticComparisonGT( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
bool operator>( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticComparisonGT( a, b );
}

////
// Comparison operations - operator >=
template< int Size, typename Real, typename ET >
__cuda_callable__
bool operator>=( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparisonGE( a, b );
}

template< typename ET, int Size, typename Real >
__cuda_callable__
bool operator>=( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticComparisonGE( a, b );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
bool operator>=( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticComparisonGE( a, b );
}

////
// Minus
template< int Size, typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Minus >
operator -( const StaticVector< Size, Real >& a )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Minus >( a );
}

////
// Abs
template< int Size, typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Abs >
abs( const StaticVector< Size, Real >& a )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Abs >( a );
}

////
// Sine
template< int Size, typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Sin >
sin( const StaticVector< Size, Real >& a )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Sin >( a );
}

////
// Cosine
template< int Size, typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Cos >
cos( const StaticVector< Size, Real >& a )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Cos >( a );
}

////
// Tangent
template< int Size, typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Tan >
tan( const StaticVector< Size, Real >& a )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Tan >( a );
}

////
// Sqrt
template< int Size, typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Sqrt >
sqrt( const StaticVector< Size, Real >& a )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Sqrt >( a );
}

////
// Cbrt
template< int Size, typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Cbrt >
cbrt( const StaticVector< Size, Real >& a )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Cbrt >( a );
}

////
// Power
template< int Size, typename Real, typename ExpType >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Pow, ExpType >
pow( const StaticVector< Size, Real >& a, const ExpType& exp )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Pow, ExpType >( a, exp );
}

////
// Floor
template< int Size, typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Floor >
floor( const StaticVector< Size, Real >& a )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Floor >( a );
}

////
// Ceil
template< int Size, typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Ceil >
ceil( const StaticVector< Size, Real >& a )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Ceil >( a );
}

////
// Acos
template< int Size, typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Acos >
acos( const StaticVector< Size, Real >& a )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Acos >( a );
}

////
// Asin
template< int Size, typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Asin >
asin( const StaticVector< Size, Real >& a )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Asin >( a );
}

////
// Atan
template< int Size, typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Atan >
atan( const StaticVector< Size, Real >& a )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Atan >( a );
}

////
// Cosh
template< int Size, typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Cosh >
cosh( const StaticVector< Size, Real >& a )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Cosh >( a );
}

////
// Tanh
template< int Size, typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Tanh >
tanh( const StaticVector< Size, Real >& a )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Tanh >( a );
}

////
// Log
template< int Size, typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Log >
log( const StaticVector< Size, Real >& a )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Log >( a );
}

////
// Log10
template< int Size, typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Log10 >
log10( const StaticVector< Size, Real >& a )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Log10 >( a );
}

////
// Log2
template< int Size, typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Log2 >
log2( const StaticVector< Size, Real >& a )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Log2 >( a );
}

////
// Exp
template< int Size, typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Exp >
exp( const StaticVector< Size, Real >& a )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Exp >( a );
}

////
// Sign
template< int Size, typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Sign >
sign( const StaticVector< Size, Real >& a )
{
   return Expressions::UnaryExpressionTemplate< StaticVector< Size, Real >, Expressions::Sign >( a );
}


////
// TODO: Replace this with multiplication when its safe
template< int Size, typename Real, typename ET >
__cuda_callable__
StaticVector< Size, Real >
Scale( const StaticVector< Size, Real >& a, const ET& b )
{
   StaticVector< Size, Real > result = Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Multiplication >( a, b );
   return result;
}

template< typename ET, int Size, typename Real >
__cuda_callable__
Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Multiplication >
Scale( const ET& a, const StaticVector< Size, Real >& b )
{
   StaticVector< Size, Real > result =  Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Multiplication >( a, b );
   return result;
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Multiplication >
Scale( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   StaticVector< Size, Real1 > result =  Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Multiplication >( a, b );
   return result;
}


   } //namespace Containers
} // namespace TNL
