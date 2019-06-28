/***************************************************************************
                          DistributedVectorViewExpressions.h  -  description
                             -------------------
    begin                : Apr 27, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <limits>
#include <TNL/Containers/Algorithms/ArrayOperations.h>
#include <TNL/Containers/Expressions/DistributedExpressionTemplates.h>
#include <TNL/Containers/Expressions/ExpressionTemplatesOperations.h>
#include <TNL/Containers/Expressions/Comparison.h>
#include <TNL/Containers/Expressions/VerticalOperations.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {

////
// All operations are supposed to be in namespace TNL
//   namespace Containers {

////
// Addition
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, ET, Containers::Expressions::Addition >
operator+( const Containers::DistributedVectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, ET, Containers::Expressions::Addition >( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Addition >
operator+( const ET& a, const Containers::DistributedVectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Addition >( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index >, Containers::DistributedVectorView< Real2, Device, Index >, Containers::Expressions::Addition >
operator+( const Containers::DistributedVectorView< Real1, Device, Index >& a, const Containers::DistributedVectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index >, Containers::DistributedVectorView< Real2, Device, Index >, Containers::Expressions::Addition >( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Subtraction
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, ET, Containers::Expressions::Subtraction >
operator-( const Containers::DistributedVectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, ET, Containers::Expressions::Subtraction >( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Subtraction >
operator-( const ET& a, const Containers::DistributedVectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Subtraction >( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index >, Containers::DistributedVectorView< Real2, Device, Index >, Containers::Expressions::Subtraction >
operator-( const Containers::DistributedVectorView< Real1, Device, Index >& a, const Containers::DistributedVectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index >, Containers::DistributedVectorView< Real2, Device, Index >, Containers::Expressions::Subtraction >( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Multiplication
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, ET, Containers::Expressions::Multiplication >
operator*( const Containers::DistributedVectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, ET, Containers::Expressions::Multiplication >( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Multiplication >
operator*( const ET& a, const Containers::DistributedVectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Multiplication >( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index >, Containers::DistributedVectorView< Real2, Device, Index >, Containers::Expressions::Multiplication >
operator*( const Containers::DistributedVectorView< Real1, Device, Index >& a, const Containers::DistributedVectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index >, Containers::DistributedVectorView< Real2, Device, Index >, Containers::Expressions::Multiplication >( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Division
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, ET, Containers::Expressions::Division >
operator/( const Containers::DistributedVectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, ET, Containers::Expressions::Division >( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Division >
operator/( const ET& a, const Containers::DistributedVectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Division >( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index >, Containers::DistributedVectorView< Real2, Device, Index >, Containers::Expressions::Division >
operator/( const Containers::DistributedVectorView< Real1, Device, Index >& a, const Containers::DistributedVectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index >, Containers::DistributedVectorView< Real2, Device, Index >, Containers::Expressions::Division >( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Min
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, ET, Containers::Expressions::Min >
min( const Containers::DistributedVectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, ET, Containers::Expressions::Min >( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Min >
min( const ET& a, const Containers::DistributedVectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Min >( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index >, Containers::DistributedVectorView< Real2, Device, Index >, Containers::Expressions::Min >
min( const Containers::DistributedVectorView< Real1, Device, Index >& a, const Containers::DistributedVectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index >, Containers::DistributedVectorView< Real2, Device, Index >, Containers::Expressions::Min >( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Max
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, ET, Containers::Expressions::Max >
max( const Containers::DistributedVectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, ET, Containers::Expressions::Max >( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Max >
max( const ET& a, const Containers::DistributedVectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Max >( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index >, Containers::DistributedVectorView< Real2, Device, Index >, Containers::Expressions::Max >
max( const Containers::DistributedVectorView< Real1, Device, Index >& a, const Containers::DistributedVectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index >, Containers::DistributedVectorView< Real2, Device, Index >, Containers::Expressions::Max >( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Comparison operations - operator ==
template< typename Real, typename Device, typename Index, typename ET >
bool operator==( const Containers::DistributedVectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::ComparisonEQ( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator==( const ET& a, const Containers::DistributedVectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::ComparisonEQ( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index >
bool operator==( const Containers::DistributedVectorView< Real1, Device1, Index >& a, const Containers::DistributedVectorView< Real2, Device2, Index >& b )
{
   if( a.getSize() != b.getSize() )
      return false;
   if( a.getSize() == 0 )
      return true;
   return Containers::Algorithms::ArrayOperations< Device1, Device2 >::
            compareMemory( a.getData(),
                           b.getData(),
                           a.getSize() );
}

////
// Comparison operations - operator !=
template< typename Real, typename Device, typename Index, typename ET >
bool operator!=( const Containers::DistributedVectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::ComparisonNE( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator!=( const ET& a, const Containers::DistributedVectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::ComparisonNE( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index >
bool operator!=( const Containers::DistributedVectorView< Real1, Device1, Index >& a, const Containers::DistributedVectorView< Real2, Device2, Index >& b )
{
   if( a.getSize() != b.getSize() )
      return false;
   if( a.getSize() == 0 )
      return true;
   return !Containers::Algorithms::ArrayOperations< Device1, Device2 >::
            compareMemory( a.getData(),
                           b.getData(),
                           a.getSize() );
}

////
// Comparison operations - operator <
template< typename Real, typename Device, typename Index, typename ET >
bool operator<( const Containers::DistributedVectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::ComparisonLT( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator<( const ET& a, const Containers::DistributedVectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::ComparisonLT( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
bool operator<( const Containers::DistributedVectorView< Real1, Device, Index >& a, const Containers::DistributedVectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::ComparisonLT( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Comparison operations - operator <=
template< typename Real, typename Device, typename Index, typename ET >
bool operator<=( const Containers::DistributedVectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::ComparisonLE( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator<=( const ET& a, const Containers::DistributedVectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::ComparisonLE( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
bool operator<=( const Containers::DistributedVectorView< Real1, Device, Index >& a, const Containers::DistributedVectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::ComparisonLE( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Comparison operations - operator >
template< typename Real, typename Device, typename Index, typename ET >
bool operator>( const Containers::DistributedVectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::ComparisonGT( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator>( const ET& a, const Containers::DistributedVectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::ComparisonGT( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
bool operator>( const Containers::DistributedVectorView< Real1, Device, Index >& a, const Containers::DistributedVectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::ComparisonGT( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Comparison operations - operator >=
template< typename Real, typename Device, typename Index, typename ET >
bool operator>=( const Containers::DistributedVectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::ComparisonGE( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator>=( const ET& a, const Containers::DistributedVectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::ComparisonGE( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
bool operator>=( const Containers::DistributedVectorView< Real1, Device, Index >& a, const Containers::DistributedVectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::ComparisonGE( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Minus
template< typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Minus >
operator-( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Minus >( a.getLocalVectorView() );
}

////
// Abs
template< typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Abs >
abs( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Abs >( a.getLocalVectorView() );
}

////
// Sine
template< typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Sin >
sin( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Sin >( a.getLocalVectorView() );
}

////
// Cosine
template< typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Cos >
cos( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Cos >( a.getLocalVectorView() );
}

////
// Tangent
template< typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Tan >
tan( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Tan >( a.getLocalVectorView() );
}

////
// Sqrt
template< typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Sqrt >
sqrt( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Sqrt >( a.getLocalVectorView() );
}

////
// Cbrt
template< typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Cbrt >
cbrt( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Cbrt >( a.getLocalVectorView() );
}

////
// Power
template< typename Real, typename Device, typename Index, typename ExpType >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Pow, ExpType >
pow( const Containers::DistributedVectorView< Real, Device, Index >& a, const ExpType& exp )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Pow, ExpType >( a.getLocalVectorView(), exp );
}

////
// Floor
template< typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Floor >
floor( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Floor >( a.getLocalVectorView() );
}

////
// Ceil
template< typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Ceil >
ceil( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Ceil >( a.getLocalVectorView() );
}

////
// Acos
template< typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Acos >
acos( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Acos >( a.getLocalVectorView() );
}

////
// Asin
template< typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Asin >
asin( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Asin >( a.getLocalVectorView() );
}

////
// Atan
template< typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Atan >
atan( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Atan >( a.getLocalVectorView() );
}

////
// Cosh
template< typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Cosh >
cosh( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Cosh >( a.getLocalVectorView() );
}

////
// Tanh
template< typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Tanh >
tanh( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Tanh >( a.getLocalVectorView() );
}

////
// Log
template< typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Log >
log( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Log >( a.getLocalVectorView() );
}

////
// Log10
template< typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Log10 >
log10( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Log10 >( a.getLocalVectorView() );
}

////
// Log2
template< typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Log2 >
log2( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Log2 >( a.getLocalVectorView() );
}

////
// Exp
template< typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Exp >
exp( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Exp >( a.getLocalVectorView() );
}

////
// Sign
template< typename Real, typename Device, typename Index >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Sign >
sign( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Sign >( a.getLocalVectorView() );
}

////
// Vertical operations - min
template< typename Real,
          typename Device,
          typename Index >
typename Containers::DistributedVectorView< Real, Device, Index >::RealType
min( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index >::CommunicatorType;
   Real result = std::numeric_limits< Real >::max();
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionMin( a.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MIN, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
typename Containers::DistributedVectorView< Real, Device, Index >::RealType
argMin( const Containers::DistributedVectorView< Real, Device, Index >& a, Index& arg )
{
   throw Exceptions::NotImplementedError( "agrMin for distributed vector view is not implemented yet." );
   return Containers::Expressions::ExpressionArgMin( a.getLocalVectorView(), arg );
}

template< typename Real,
          typename Device,
          typename Index >
typename Containers::DistributedVectorView< Real, Device, Index >::RealType
max( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index >::CommunicatorType;
   Real result = std::numeric_limits< Real >::min();
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionMax( a.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MAX, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
typename Containers::DistributedVectorView< Real, Device, Index >::RealType
argMax( const Containers::DistributedVectorView< Real, Device, Index >& a, Index& arg )
{
   throw Exceptions::NotImplementedError( "agrMax for distributed vector view is not implemented yet." );
   return Containers::Expressions::ExpressionArgMax( a.getLocalVectorView(), arg );
}

template< typename Real,
          typename Device,
          typename Index >
typename Containers::DistributedVectorView< Real, Device, Index >::RealType
sum( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index >::CommunicatorType;
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionSum( a.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Real2 >
typename Containers::DistributedVectorView< Real, Device, Index >::RealType
lpNorm( const Containers::DistributedVectorView< Real, Device, Index >& a, const Real2& p )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index >::CommunicatorType;
   Real result = ( Real ) 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::pow( Containers::Expressions::ExpressionLpNorm( a.getLocalVectorView(), p ), p );
      CommunicatorType::template Allreduce< Real >( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return TNL::pow( result, 1.0 / p );
}

template< typename Real,
          typename Device,
          typename Index >
typename Containers::DistributedVectorView< Real, Device, Index >::RealType
product( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index >::CommunicatorType;
   Real result = ( Real ) 1.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionProduct( a.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_PROD, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
bool
logicalOr( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index >::CommunicatorType;
   bool result = false;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionLogicalOr( a.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LOR, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
typename Containers::DistributedVectorView< Real, Device, Index >::RealType
binaryOr( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index >::CommunicatorType;
   Real result = ( Real ) 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionBinaryOr( a.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_BOR, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
bool
logicalAnd( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index >::CommunicatorType;
   Real result = std::numeric_limits< Real >::max();
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionLogicalAnd( a.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
typename Containers::DistributedVectorView< Real, Device, Index >::RealType
binaryAnd( const Containers::DistributedVectorView< Real, Device, Index >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index >::CommunicatorType;
   bool result = true;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionBinaryAnd( a.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_BAND, a.getCommunicationGroup() );
   }
   return result;
}

////
// Scalar product
template< typename Real, typename Device, typename Index, typename ET >
Real operator,( const Containers::DistributedVectorView< Real, Device, Index >& a, const ET& b )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index >::CommunicatorType;
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::sum( a.getLocalVectorView() * b );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

template< typename ET, typename Real, typename Device, typename Index >
Real operator,( const ET& a, const Containers::DistributedVectorView< Real, Device, Index >& b )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index >::CommunicatorType;
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::sum( a * b.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real1, typename Real2, typename Device, typename Index >
auto operator,( const Containers::DistributedVectorView< Real1, Device, Index >& a, const Containers::DistributedVectorView< Real2, Device, Index >& b )
->decltype( TNL::sum( a.getLocalVectorView() * b.getLocalVectorView() ) )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real1, Device, Index >::CommunicatorType;
   using Real = decltype( TNL::sum( a.getLocalVectorView() * b.getLocalVectorView() ) );
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::sum( a.getLocalVectorView() * b.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

////
// Dot product - the same as scalar product, just for convenience
template< typename Real, typename Device, typename Index, typename ET >
Real dot( const Containers::DistributedVectorView< Real, Device, Index >& a, const ET& b )
{
   return ( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
Real dot( const ET& a, const Containers::DistributedVectorView< Real, Device, Index >& b )
{
   return ( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
auto dot( const Containers::DistributedVectorView< Real1, Device, Index >& a, const Containers::DistributedVectorView< Real2, Device, Index >& b )
->decltype( TNL::sum( a.getLocalVectorView() * b.getLocalVectorView() ) )
{
   return ( a, b );
}

////
// TODO: Replace this with multiplication when its safe
template< typename Real, typename Device, typename Index, typename ET >
Containers::DistributedVectorView< Real, Device, Index >
Scale( const Containers::DistributedVectorView< Real, Device, Index >& a, const ET& b )
{
   Containers::DistributedVectorView< Real, Device, Index > result = Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index >, ET, Containers::Expressions::Multiplication >( a, b );
   return result;
}

template< typename ET, typename Real, typename Device, typename Index >
Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Multiplication >
Scale( const ET& a, const Containers::DistributedVectorView< Real, Device, Index >& b )
{
   Containers::DistributedVectorView< Real, Device, Index > result =  Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index >, Containers::Expressions::Multiplication >( a, b );
   return result;
}

template< typename Real1, typename Real2, typename Device, typename Index >
Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index >, Containers::DistributedVectorView< Real2, Device, Index >, Containers::Expressions::Multiplication >
Scale( const Containers::DistributedVectorView< Real1, Device, Index >& a, const Containers::DistributedVectorView< Real2, Device, Index >& b )
{
   Containers::DistributedVectorView< Real1, Device, Index > result =  Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index >, Containers::DistributedVectorView< Real2, Device, Index >, Containers::Expressions::Multiplication >( a, b );
   return result;
}

} // namespace TNL
