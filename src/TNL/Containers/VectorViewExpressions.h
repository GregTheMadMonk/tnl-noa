/***************************************************************************
                          VectorViewExpressions.h  -  description
                             -------------------
    begin                : Apr 27, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Expressions/ExpressionTemplates.h>
#include <TNL/Containers/Expressions/ExpressionTemplatesOperations.h>
#include <TNL/Containers/Expressions/Comparison.h>

namespace TNL {
   namespace Containers {

////
// Addition
template< typename Real, typename Device, typename Index, typename ET >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< VectorView< Real, Device, Index >, ET, Expressions::Addition >
operator+( const VectorView< Real, Device, Index >& a, const ET& b )
{
   return Expressions::BinaryExpressionTemplate< VectorView< Real, Device, Index >, ET, Expressions::Addition >( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< ET, VectorView< Real, Device, Index >, Expressions::Addition >
operator+( const ET& a, const VectorView< Real, Device, Index >& b )
{
   return Expressions::BinaryExpressionTemplate< ET, VectorView< Real, Device, Index >, Expressions::Addition >( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< VectorView< Real1, Device, Index >, VectorView< Real2, Device, Index >, Expressions::Addition >
operator+( const VectorView< Real1, Device, Index >& a, const VectorView< Real2, Device, Index >& b )
{
   return Expressions::BinaryExpressionTemplate< VectorView< Real1, Device, Index >, VectorView< Real2, Device, Index >, Expressions::Addition >( a, b );
}

////
// Subtraction
template< typename Real, typename Device, typename Index, typename ET >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< VectorView< Real, Device, Index >, ET, Expressions::Subtraction >
operator-( const VectorView< Real, Device, Index >& a, const ET& b )
{
   return Expressions::BinaryExpressionTemplate< VectorView< Real, Device, Index >, ET, Expressions::Subtraction >( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< ET, VectorView< Real, Device, Index >, Expressions::Subtraction >
operator-( const ET& a, const VectorView< Real, Device, Index >& b )
{
   return Expressions::BinaryExpressionTemplate< ET, VectorView< Real, Device, Index >, Expressions::Subtraction >( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< VectorView< Real1, Device, Index >, VectorView< Real2, Device, Index >, Expressions::Subtraction >
operator-( const VectorView< Real1, Device, Index >& a, const VectorView< Real2, Device, Index >& b )
{
   return Expressions::BinaryExpressionTemplate< VectorView< Real1, Device, Index >, VectorView< Real2, Device, Index >, Expressions::Subtraction >( a, b );
}

////
// Multiplication
template< typename Real, typename Device, typename Index, typename ET >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< VectorView< Real, Device, Index >, ET, Expressions::Multiplication >
operator*( const VectorView< Real, Device, Index >& a, const ET& b )
{
   return Expressions::BinaryExpressionTemplate< VectorView< Real, Device, Index >, ET, Expressions::Multiplication >( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< ET, VectorView< Real, Device, Index >, Expressions::Multiplication >
operator*( const ET& a, const VectorView< Real, Device, Index >& b )
{
   return Expressions::BinaryExpressionTemplate< ET, VectorView< Real, Device, Index >, Expressions::Multiplication >( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< VectorView< Real1, Device, Index >, VectorView< Real2, Device, Index >, Expressions::Multiplication >
operator*( const VectorView< Real1, Device, Index >& a, const VectorView< Real2, Device, Index >& b )
{
   return Expressions::BinaryExpressionTemplate< VectorView< Real1, Device, Index >, VectorView< Real2, Device, Index >, Expressions::Multiplication >( a, b );
}

////
// Division
template< typename Real, typename Device, typename Index, typename ET >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< VectorView< Real, Device, Index >, ET, Expressions::Division >
operator/( const VectorView< Real, Device, Index >& a, const ET& b )
{
   return Expressions::BinaryExpressionTemplate< VectorView< Real, Device, Index >, ET, Expressions::Division >( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< ET, VectorView< Real, Device, Index >, Expressions::Division >
operator/( const ET& a, const VectorView< Real, Device, Index >& b )
{
   return Expressions::BinaryExpressionTemplate< ET, VectorView< Real, Device, Index >, Expressions::Division >( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< VectorView< Real1, Device, Index >, VectorView< Real2, Device, Index >, Expressions::Division >
operator/( const VectorView< Real1, Device, Index >& a, const VectorView< Real2, Device, Index >& b )
{
   return Expressions::BinaryExpressionTemplate< VectorView< Real1, Device, Index >, VectorView< Real2, Device, Index >, Expressions::Division >( a, b );
}

////
// Min
template< typename Real, typename Device, typename Index, typename ET >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< VectorView< Real, Device, Index >, ET, Expressions::Min >
min( const VectorView< Real, Device, Index >& a, const ET& b )
{
   return Expressions::BinaryExpressionTemplate< VectorView< Real, Device, Index >, ET, Expressions::Min >( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< ET, VectorView< Real, Device, Index >, Expressions::Min >
min( const ET& a, const VectorView< Real, Device, Index >& b )
{
   return Expressions::BinaryExpressionTemplate< ET, VectorView< Real, Device, Index >, Expressions::Min >( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< VectorView< Real1, Device, Index >, VectorView< Real2, Device, Index >, Expressions::Min >
min( const VectorView< Real1, Device, Index >& a, const VectorView< Real2, Device, Index >& b )
{
   return Expressions::BinaryExpressionTemplate< VectorView< Real1, Device, Index >, VectorView< Real2, Device, Index >, Expressions::Min >( a, b );
}

////
// Max
template< typename Real, typename Device, typename Index, typename ET >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< VectorView< Real, Device, Index >, ET, Expressions::Max >
max( const VectorView< Real, Device, Index >& a, const ET& b )
{
   return Expressions::BinaryExpressionTemplate< VectorView< Real, Device, Index >, ET, Expressions::Max >( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< ET, VectorView< Real, Device, Index >, Expressions::Max >
max( const ET& a, const VectorView< Real, Device, Index >& b )
{
   return Expressions::BinaryExpressionTemplate< ET, VectorView< Real, Device, Index >, Expressions::Max >( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
__cuda_callable__
const Expressions::BinaryExpressionTemplate< VectorView< Real1, Device, Index >, VectorView< Real2, Device, Index >, Expressions::Max >
max( const VectorView< Real1, Device, Index >& a, const VectorView< Real2, Device, Index >& b )
{
   return Expressions::BinaryExpressionTemplate< VectorView< Real1, Device, Index >, VectorView< Real2, Device, Index >, Expressions::Max >( a, b );
}

////
// Comparison operations - operator ==
template< typename Real, typename Device, typename Index, typename ET >
__cuda_callable__
bool operator==( const VectorView< Real, Device, Index >& a, const ET& b )
{
   return Expressions::ComparisonEQ( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
__cuda_callable__
bool operator==( const ET& a, const VectorView< Real, Device, Index >& b )
{
   return Expressions::ComparisonEQ( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
__cuda_callable__
bool operator==( const VectorView< Real1, Device, Index >& a, const VectorView< Real2, Device, Index >& b )
{
   return Expressions::ComparisonEQ( a, b );
}

////
// Comparison operations - operator !=
template< typename Real, typename Device, typename Index, typename ET >
__cuda_callable__
bool operator!=( const VectorView< Real, Device, Index >& a, const ET& b )
{
   return Expressions::ComparisonNE( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
__cuda_callable__
bool operator!=( const ET& a, const VectorView< Real, Device, Index >& b )
{
   return Expressions::ComparisonNE( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
__cuda_callable__
bool operator!=( const VectorView< Real1, Device, Index >& a, const VectorView< Real2, Device, Index >& b )
{
   return Expressions::ComparisonNE( a, b );
}

////
// Comparison operations - operator <
template< typename Real, typename Device, typename Index, typename ET >
__cuda_callable__
bool operator<( const VectorView< Real, Device, Index >& a, const ET& b )
{
   return Expressions::ComparisonLT( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
__cuda_callable__
bool operator<( const ET& a, const VectorView< Real, Device, Index >& b )
{
   return Expressions::ComparisonLT( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
__cuda_callable__
bool operator<( const VectorView< Real1, Device, Index >& a, const VectorView< Real2, Device, Index >& b )
{
   return Expressions::ComparisonLT( a, b );
}

////
// Comparison operations - operator <=
template< typename Real, typename Device, typename Index, typename ET >
__cuda_callable__
bool operator<=( const VectorView< Real, Device, Index >& a, const ET& b )
{
   return Expressions::ComparisonLE( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
__cuda_callable__
bool operator<=( const ET& a, const VectorView< Real, Device, Index >& b )
{
   return Expressions::ComparisonLE( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
__cuda_callable__
bool operator<=( const VectorView< Real1, Device, Index >& a, const VectorView< Real2, Device, Index >& b )
{
   return Expressions::ComparisonLE( a, b );
}

////
// Comparison operations - operator >
template< typename Real, typename Device, typename Index, typename ET >
__cuda_callable__
bool operator>( const VectorView< Real, Device, Index >& a, const ET& b )
{
   return Expressions::ComparisonGT( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
__cuda_callable__
bool operator>( const ET& a, const VectorView< Real, Device, Index >& b )
{
   return Expressions::ComparisonGT( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
__cuda_callable__
bool operator>( const VectorView< Real1, Device, Index >& a, const VectorView< Real2, Device, Index >& b )
{
   return Expressions::ComparisonGT( a, b );
}

////
// Comparison operations - operator >=
template< typename Real, typename Device, typename Index, typename ET >
__cuda_callable__
bool operator>=( const VectorView< Real, Device, Index >& a, const ET& b )
{
   return Expressions::ComparisonGE( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
__cuda_callable__
bool operator>=( const ET& a, const VectorView< Real, Device, Index >& b )
{
   return Expressions::ComparisonGE( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
__cuda_callable__
bool operator>=( const VectorView< Real1, Device, Index >& a, const VectorView< Real2, Device, Index >& b )
{
   return Expressions::ComparisonGE( a, b );
}

////
// Minus
template< typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Minus >
operator-( const VectorView< Real, Device, Index >& a )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Minus >( a );
}

////
// Abs
template< typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Abs >
abs( const VectorView< Real, Device, Index >& a )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Abs >( a );
}

////
// Sine
template< typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Sin >
sin( const VectorView< Real, Device, Index >& a )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Sin >( a );
}

////
// Cosine
template< typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Cos >
cos( const VectorView< Real, Device, Index >& a )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Cos >( a );
}

////
// Tangent
template< typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Tan >
tan( const VectorView< Real, Device, Index >& a )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Tan >( a );
}

////
// Sqrt
template< typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Sqrt >
sqrt( const VectorView< Real, Device, Index >& a )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Sqrt >( a );
}

////
// Cbrt
template< typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Cbrt >
cbrt( const VectorView< Real, Device, Index >& a )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Cbrt >( a );
}

////
// Power
template< typename Real, typename Device, typename Index, typename ExpType >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Pow, ExpType >
pow( const VectorView< Real, Device, Index >& a, const ExpType& exp )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Pow, ExpType >( a, exp );
}

////
// Floor
template< typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Floor >
floor( const VectorView< Real, Device, Index >& a )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Floor >( a );
}

////
// Ceil
template< typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Ceil >
ceil( const VectorView< Real, Device, Index >& a )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Ceil >( a );
}

////
// Acos
template< typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Acos >
acos( const VectorView< Real, Device, Index >& a )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Acos >( a );
}

////
// Asin
template< typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Asin >
asin( const VectorView< Real, Device, Index >& a )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Asin >( a );
}

////
// Atan
template< typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Atan >
atan( const VectorView< Real, Device, Index >& a )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Atan >( a );
}

////
// Cosh
template< typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Cosh >
cosh( const VectorView< Real, Device, Index >& a )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Cosh >( a );
}

////
// Tanh
template< typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Tanh >
tanh( const VectorView< Real, Device, Index >& a )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Tanh >( a );
}

////
// Log
template< typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Log >
log( const VectorView< Real, Device, Index >& a )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Log >( a );
}

////
// Log10
template< typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Log10 >
log10( const VectorView< Real, Device, Index >& a )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Log10 >( a );
}

////
// Log2
template< typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Log2 >
log2( const VectorView< Real, Device, Index >& a )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Log2 >( a );
}

////
// Exp
template< typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Exp >
exp( const VectorView< Real, Device, Index >& a )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Exp >( a );
}

////
// Sign
template< typename Real, typename Device, typename Index >
__cuda_callable__
const Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Sign >
sign( const VectorView< Real, Device, Index >& a )
{
   return Expressions::UnaryExpressionTemplate< VectorView< Real, Device, Index >, Expressions::Sign >( a );
}


////
// TODO: Replace this with multiplication when its safe
template< typename Real, typename Device, typename Index, typename ET >
__cuda_callable__
VectorView< Real, Device, Index >
Scale( const VectorView< Real, Device, Index >& a, const ET& b )
{
   VectorView< Real, Device, Index > result = Expressions::BinaryExpressionTemplate< VectorView< Real, Device, Index >, ET, Expressions::Multiplication >( a, b );
   return result;
}

template< typename ET, typename Real, typename Device, typename Index >
__cuda_callable__
Expressions::BinaryExpressionTemplate< ET, VectorView< Real, Device, Index >, Expressions::Multiplication >
Scale( const ET& a, const VectorView< Real, Device, Index >& b )
{
   VectorView< Real, Device, Index > result =  Expressions::BinaryExpressionTemplate< ET, VectorView< Real, Device, Index >, Expressions::Multiplication >( a, b );
   return result;
}

template< typename Real1, typename Real2, typename Device, typename Index >
__cuda_callable__
Expressions::BinaryExpressionTemplate< VectorView< Real1, Device, Index >, VectorView< Real2, Device, Index >, Expressions::Multiplication >
Scale( const VectorView< Real1, Device, Index >& a, const VectorView< Real2, Device, Index >& b )
{
   VectorView< Real1, Device, Index > result =  Expressions::BinaryExpressionTemplate< VectorView< Real1, Device, Index >, VectorView< Real2, Device, Index >, Expressions::Multiplication >( a, b );
   return result;
}


   } //namespace Containers
} // namespace TNL
