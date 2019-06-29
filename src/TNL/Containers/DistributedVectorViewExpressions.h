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
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, ET, Containers::Expressions::Addition >
operator+( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, ET, Containers::Expressions::Addition >( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Addition >
operator+( const ET& a, const Containers::DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Addition >( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index, Communicator >, Containers::DistributedVectorView< Real2, Device, Index, Communicator >, Containers::Expressions::Addition >
operator+( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index, Communicator >, Containers::DistributedVectorView< Real2, Device, Index, Communicator >, Containers::Expressions::Addition >( a.getView(), b.getView() );
}

////
// Subtraction
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, ET, Containers::Expressions::Subtraction >
operator-( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, ET, Containers::Expressions::Subtraction >( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Subtraction >
operator-( const ET& a, const Containers::DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Subtraction >( a.getView(), b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index, Communicator >, Containers::DistributedVectorView< Real2, Device, Index, Communicator >, Containers::Expressions::Subtraction >
operator-( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index, Communicator >, Containers::DistributedVectorView< Real2, Device, Index, Communicator >, Containers::Expressions::Subtraction >( a.getView(), b.getView() );
}

////
// Multiplication
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, ET, Containers::Expressions::Multiplication >
operator*( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, ET, Containers::Expressions::Multiplication >( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Multiplication >
operator*( const ET& a, const Containers::DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Multiplication >( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index, Communicator >, Containers::DistributedVectorView< Real2, Device, Index, Communicator >, Containers::Expressions::Multiplication >
operator*( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index, Communicator >, Containers::DistributedVectorView< Real2, Device, Index, Communicator >, Containers::Expressions::Multiplication >( a.getView(), b.getView() );
}

////
// Division
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, ET, Containers::Expressions::Division >
operator/( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, ET, Containers::Expressions::Division >( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Division >
operator/( const ET& a, const Containers::DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Division >( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index, Communicator >, Containers::DistributedVectorView< Real2, Device, Index, Communicator >, Containers::Expressions::Division >
operator/( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index, Communicator >, Containers::DistributedVectorView< Real2, Device, Index, Communicator >, Containers::Expressions::Division >( a.getView(), b.getView() );
}

////
// Min
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, ET, Containers::Expressions::Min >
min( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, ET, Containers::Expressions::Min >( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Min >
min( const ET& a, const Containers::DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Min >( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index, Communicator >, Containers::DistributedVectorView< Real2, Device, Index, Communicator >, Containers::Expressions::Min >
min( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index, Communicator >, Containers::DistributedVectorView< Real2, Device, Index, Communicator >, Containers::Expressions::Min >( a.getView(), b.getView() );
}

////
// Max
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, ET, Containers::Expressions::Max >
max( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, ET, Containers::Expressions::Max >( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Max >
max( const ET& a, const Containers::DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Max >( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index, Communicator >, Containers::DistributedVectorView< Real2, Device, Index, Communicator >, Containers::Expressions::Max >
max( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index, Communicator >, Containers::DistributedVectorView< Real2, Device, Index, Communicator >, Containers::Expressions::Max >( a.getView(), b.getView() );
}

////
// Comparison operations - operator ==
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
bool operator==( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   return Containers::Expressions::ComparisonEQ( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
bool operator==( const ET& a, const Containers::DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonEQ( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index, typename Communicator >
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
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
bool operator!=( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   return Containers::Expressions::ComparisonNE( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
bool operator!=( const ET& a, const Containers::DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonNE( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index, typename Communicator >
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
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
bool operator<( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   return Containers::Expressions::ComparisonLT( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
bool operator<( const ET& a, const Containers::DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonLT( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator<( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonLT( a.getView(), b.getView() );
}

////
// Comparison operations - operator <=
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
bool operator<=( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   return Containers::Expressions::ComparisonLE( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
bool operator<=( const ET& a, const Containers::DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonLE( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator<=( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonLE( a.getView(), b.getView() );
}

////
// Comparison operations - operator >
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
bool operator>( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   return Containers::Expressions::ComparisonGT( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
bool operator>( const ET& a, const Containers::DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonGT( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator>( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonGT( a.getView(), b.getView() );
}

////
// Comparison operations - operator >=
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
bool operator>=( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   return Containers::Expressions::ComparisonGE( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
bool operator>=( const ET& a, const Containers::DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonGE( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator>=( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonGE( a.getView(), b.getView() );
}

////
// Minus
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Minus >
operator-( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Minus >( a.getView() );
}

////
// Abs
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Abs >
abs( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Abs >( a.getView() );
}

////
// Sine
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Sin >
sin( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Sin >( a.getView() );
}

////
// Cosine
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Cos >
cos( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Cos >( a.getView() );
}

////
// Tangent
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Tan >
tan( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Tan >( a.getView() );
}

////
// Sqrt
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Sqrt >
sqrt( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Sqrt >( a.getView() );
}

////
// Cbrt
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Cbrt >
cbrt( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Cbrt >( a.getView() );
}

////
// Power
template< typename Real, typename Device, typename Index, typename Communicator, typename ExpType >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Pow, ExpType >
pow( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ExpType& exp )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Pow, ExpType >( a.getView(), exp );
}

////
// Floor
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Floor >
floor( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Floor >( a.getView() );
}

////
// Ceil
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Ceil >
ceil( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Ceil >( a.getView() );
}

////
// Acos
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Acos >
acos( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Acos >( a.getView() );
}

////
// Asin
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Asin >
asin( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Asin >( a.getView() );
}

////
// Atan
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Atan >
atan( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Atan >( a.getView() );
}

////
// Cosh
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Cosh >
cosh( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Cosh >( a.getView() );
}

////
// Tanh
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Tanh >
tanh( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Tanh >( a.getView() );
}

////
// Log
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Log >
log( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Log >( a.getView() );
}

////
// Log10
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Log10 >
log10( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Log10 >( a.getView() );
}

////
// Log2
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Log2 >
log2( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Log2 >( a.getView() );
}

////
// Exp
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Exp >
exp( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Exp >( a.getView() );
}

////
// Sign
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Sign >
sign( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Sign >( a.getView() );
}

////
// Vertical operations - min
template< typename Real,
          typename Device,
          typename Index, typename Communicator >
typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::RealType
min( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = std::numeric_limits< Real >::max();
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionMin( a.getView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MIN, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::RealType
argMin( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, Index& arg )
{
   throw Exceptions::NotImplementedError( "agrMin for distributed vector view is not implemented yet." );
   return Containers::Expressions::ExpressionArgMin( a.getView(), arg );
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::RealType
max( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = std::numeric_limits< Real >::min();
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionMax( a.getView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MAX, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::RealType
argMax( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, Index& arg )
{
   throw Exceptions::NotImplementedError( "agrMax for distributed vector view is not implemented yet." );
   return Containers::Expressions::ExpressionArgMax( a.getView(), arg );
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::RealType
sum( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionSum( a.getView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator,
          typename Real2 >
typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::RealType
lpNorm( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const Real2& p )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = ( Real ) 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::pow( Containers::Expressions::ExpressionLpNorm( a.getView(), p ), p );
      CommunicatorType::template Allreduce< Real >( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return TNL::pow( result, 1.0 / p );
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::RealType
product( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = ( Real ) 1.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionProduct( a.getView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_PROD, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
bool
logicalOr( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
   bool result = false;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionLogicalOr( a.getView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LOR, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::RealType
binaryOr( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = ( Real ) 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionBinaryOr( a.getView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_BOR, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
bool
logicalAnd( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = std::numeric_limits< Real >::max();
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionLogicalAnd( a.getView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::RealType
binaryAnd( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
   bool result = true;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionBinaryAnd( a.getView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_BAND, a.getCommunicationGroup() );
   }
   return result;
}

////
// Scalar product
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
Real operator,( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::sum( a.getView() * b );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
Real operator,( const ET& a, const Containers::DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::sum( a * b.getView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto operator,( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
->decltype( TNL::sum( a.getView() * b.getView() ) )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real1, Device, Index, Communicator >::CommunicatorType;
   using Real = decltype( TNL::sum( a.getView() * b.getView() ) );
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::sum( a.getView() * b.getView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

////
// Dot product - the same as scalar product, just for convenience
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
Real dot( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   return ( a, b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
Real dot( const ET& a, const Containers::DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   return ( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto dot( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
->decltype( TNL::sum( a.getView() * b.getView() ) )
{
   return ( a, b );
}

////
// TODO: Replace this with multiplication when its safe
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
Containers::DistributedVectorView< Real, Device, Index, Communicator >
Scale( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   Containers::DistributedVectorView< Real, Device, Index, Communicator > result = Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, ET, Containers::Expressions::Multiplication >( a, b );
   return result;
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Multiplication >
Scale( const ET& a, const Containers::DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   Containers::DistributedVectorView< Real, Device, Index, Communicator > result =  Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Multiplication >( a, b );
   return result;
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index, Communicator >, Containers::DistributedVectorView< Real2, Device, Index, Communicator >, Containers::Expressions::Multiplication >
Scale( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   Containers::DistributedVectorView< Real1, Device, Index, Communicator > result =  Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index, Communicator >, Containers::DistributedVectorView< Real2, Device, Index, Communicator >, Containers::Expressions::Multiplication >( a, b );
   return result;
}

} // namespace TNL
