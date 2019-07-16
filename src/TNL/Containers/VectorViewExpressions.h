/***************************************************************************
                          VectorViewExpressions.h  -  description
                             -------------------
    begin                : Apr 27, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Algorithms/ArrayOperations.h>
#include <TNL/Containers/Expressions/ExpressionTemplates.h>
#include <TNL/Containers/Expressions/ExpressionTemplatesOperations.h>
#include <TNL/Containers/Expressions/Comparison.h>
#include <TNL/Containers/Expressions/VerticalOperations.h>

namespace TNL {
   namespace Containers {
////
// All operations are supposed to be in namespace TNL


////
// Addition
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, ET, Containers::Expressions::Addition >
operator+( const Containers::VectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, ET, Containers::Expressions::Addition >( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Addition >
operator+( const ET& a, const Containers::VectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Addition >( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Addition >
operator+( const Containers::VectorView< Real1, Device, Index >& a, const Containers::VectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Addition >( a, b );
}

////
// Subtraction
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, ET, Containers::Expressions::Subtraction >
operator-( const Containers::VectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, ET, Containers::Expressions::Subtraction >( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Subtraction >
operator-( const ET& a, const Containers::VectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Subtraction >( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Subtraction >
operator-( const Containers::VectorView< Real1, Device, Index >& a, const Containers::VectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Subtraction >( a, b );
}

////
// Multiplication
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, ET, Containers::Expressions::Multiplication >
operator*( const Containers::VectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, ET, Containers::Expressions::Multiplication >( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Multiplication >
operator*( const ET& a, const Containers::VectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Multiplication >( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Multiplication >
operator*( const Containers::VectorView< Real1, Device, Index >& a, const Containers::VectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Multiplication >( a, b );
}

////
// Division
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, ET, Containers::Expressions::Division >
operator/( const Containers::VectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, ET, Containers::Expressions::Division >( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Division >
operator/( const ET& a, const Containers::VectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Division >( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Division >
operator/( const Containers::VectorView< Real1, Device, Index >& a, const Containers::VectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Division >( a, b );
}

////
// Comparison operations - operator ==
template< typename Real, typename Device, typename Index, typename ET >
bool operator==( const Containers::VectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::Comparison< Containers::VectorView< Real, Device, Index >, ET >::EQ( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator==( const ET& a, const Containers::VectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::Comparison< ET, Containers::VectorView< Real, Device, Index > >::EQ( a, b );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index >
bool operator==( const Containers::VectorView< Real1, Device1, Index >& a, const Containers::VectorView< Real2, Device2, Index >& b )
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
bool operator!=( const Containers::VectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::Comparison< Containers::VectorView< Real, Device, Index >, ET >::NE( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator!=( const ET& a, const Containers::VectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::Comparison< ET, Containers::VectorView< Real, Device, Index > >::NE( a, b );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index >
bool operator!=( const Containers::VectorView< Real1, Device1, Index >& a, const Containers::VectorView< Real2, Device2, Index >& b )
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
bool operator<( const Containers::VectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::Comparison< Containers::VectorView< Real, Device, Index >, ET >::LT( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator<( const ET& a, const Containers::VectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::Comparison< ET, Containers::VectorView< Real, Device, Index > >::LT( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
bool operator<( const Containers::VectorView< Real1, Device, Index >& a, const Containers::VectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::Comparison< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index > >::LT( a, b );
}

////
// Comparison operations - operator <=
template< typename Real, typename Device, typename Index, typename ET >
bool operator<=( const Containers::VectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::Comparison< Containers::VectorView< Real, Device, Index >, ET >::LE( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator<=( const ET& a, const Containers::VectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::Comparison< ET, Containers::VectorView< Real, Device, Index > >::LE( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
bool operator<=( const Containers::VectorView< Real1, Device, Index >& a, const Containers::VectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::Comparison< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index > >::LE( a, b );
}

////
// Comparison operations - operator >
template< typename Real, typename Device, typename Index, typename ET >
bool operator>( const Containers::VectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::Comparison< Containers::VectorView< Real, Device, Index >, ET >::GT( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator>( const ET& a, const Containers::VectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::Comparison< ET, Containers::VectorView< Real, Device, Index > >::GT( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
bool operator>( const Containers::VectorView< Real1, Device, Index >& a, const Containers::VectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::Comparison< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index > >::GT( a, b );
}

////
// Comparison operations - operator >=
template< typename Real, typename Device, typename Index, typename ET >
bool operator>=( const Containers::VectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::Comparison< Containers::VectorView< Real, Device, Index >, ET >::GE( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator>=( const ET& a, const Containers::VectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::Comparison< ET, Containers::VectorView< Real, Device, Index > >::GE( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
bool operator>=( const Containers::VectorView< Real1, Device, Index >& a, const Containers::VectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::Comparison< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index > >::GE( a, b );
}

////
// Minus
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Minus >
operator-( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Minus >( a );
}

////
// Scalar product
template< typename Real, typename Device, typename Index, typename ET >
//Real 
auto
operator,( const Containers::VectorView< Real, Device, Index >& a, const ET& b )
->decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename ET, typename Real, typename Device, typename Index >
//Real
auto
operator,( const ET& a, const Containers::VectorView< Real, Device, Index >& b )
->decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
auto operator,( const Containers::VectorView< Real1, Device, Index >& a, const Containers::VectorView< Real2, Device, Index >& b )
->decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

} //namespace Containers

////
// All functions are supposed to be in namespace TNL

////
// Min
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, ET, Containers::Expressions::Min >
min( const Containers::VectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, ET, Containers::Expressions::Min >( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Min >
min( const ET& a, const Containers::VectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Min >( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Min >
min( const Containers::VectorView< Real1, Device, Index >& a, const Containers::VectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Min >( a, b );
}

////
// Max
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, ET, Containers::Expressions::Max >
max( const Containers::VectorView< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, ET, Containers::Expressions::Max >( a, b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Max >
max( const ET& a, const Containers::VectorView< Real, Device, Index >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Max >( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Max >
max( const Containers::VectorView< Real1, Device, Index >& a, const Containers::VectorView< Real2, Device, Index >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Max >( a, b );
}

////
// Abs
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Abs >
abs( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Abs >( a );
}

////
// Sine
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Sin >
sin( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Sin >( a );
}

////
// Cosine
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Cos >
cos( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Cos >( a );
}

////
// Tangent
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Tan >
tan( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Tan >( a );
}

////
// Sqrt
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Sqrt >
sqrt( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Sqrt >( a );
}

////
// Cbrt
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Cbrt >
cbrt( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Cbrt >( a );
}

////
// Power
template< typename Real, typename Device, typename Index, typename ExpType >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Pow, ExpType >
pow( const Containers::VectorView< Real, Device, Index >& a, const ExpType& exp )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Pow, ExpType >( a, exp );
}

////
// Floor
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Floor >
floor( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Floor >( a );
}

////
// Ceil
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Ceil >
ceil( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Ceil >( a );
}

////
// Acos
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Acos >
acos( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Acos >( a );
}

////
// Asin
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Asin >
asin( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Asin >( a );
}

////
// Atan
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Atan >
atan( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Atan >( a );
}

////
// Cosh
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Cosh >
cosh( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Cosh >( a );
}

////
// Tanh
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Tanh >
tanh( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Tanh >( a );
}

////
// Log
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Log >
log( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Log >( a );
}

////
// Log10
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Log10 >
log10( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Log10 >( a );
}

////
// Log2
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Log2 >
log2( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Log2 >( a );
}

////
// Exp
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Exp >
exp( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Exp >( a );
}

////
// Sign
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Sign >
sign( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Sign >( a );
}

////
// Vertical operations - min
template< typename Real,
          typename Device,
          typename Index >
typename Containers::VectorView< Real, Device, Index >::RealType
min( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::ExpressionMin( a );
}

template< typename Real,
          typename Device,
          typename Index >
typename Containers::VectorView< Real, Device, Index >::RealType
argMin( const Containers::VectorView< Real, Device, Index >& a, Index& arg )
{
   return Containers::Expressions::ExpressionArgMin( a, arg );
}

template< typename Real,
          typename Device,
          typename Index >
typename Containers::VectorView< Real, Device, Index >::RealType
max( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::ExpressionMax( a );
}

template< typename Real,
          typename Device,
          typename Index >
typename Containers::VectorView< Real, Device, Index >::RealType
argMax( const Containers::VectorView< Real, Device, Index >& a, Index& arg )
{
   return Containers::Expressions::ExpressionArgMax( a, arg );
}

template< typename Real,
          typename Device,
          typename Index >
auto
sum( const Containers::VectorView< Real, Device, Index >& a )
-> decltype( Containers::Expressions::ExpressionSum( a ) )
{
   return Containers::Expressions::ExpressionSum( a );
}

template< typename Real,
          typename Device,
          typename Index,
          typename Real2 >
auto
lpNorm( const Containers::VectorView< Real, Device, Index >& a, const Real2& p )
-> decltype( Containers::Expressions::ExpressionLpNorm( a, p ) )
{
   if( p == 1.0 )
      return Containers::Expressions::ExpressionLpNorm( a, p );
   if( p == 2.0 )
      return TNL::sqrt( Containers::Expressions::ExpressionLpNorm( a, p ) );
   return TNL::pow( Containers::Expressions::ExpressionLpNorm( a, p ), 1.0 / p );
}

template< typename Real,
          typename Device,
          typename Index >
auto
product( const Containers::VectorView< Real, Device, Index >& a )
-> decltype( Containers::Expressions::ExpressionProduct( a ) )
{
   return Containers::Expressions::ExpressionProduct( a );
}

template< typename Real,
          typename Device,
          typename Index >
bool
logicalOr( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::ExpressionLogicalOr( a );
}

template< typename Real,
          typename Device,
          typename Index >
auto
binaryOr( const Containers::VectorView< Real, Device, Index >& a )
-> decltype( Containers::Expressions::ExpressionBinaryOr( a ) )
{
   return Containers::Expressions::ExpressionBinaryOr( a );
}

template< typename Real,
          typename Device,
          typename Index >
bool
logicalAnd( const Containers::VectorView< Real, Device, Index >& a )
{
   return Containers::Expressions::ExpressionLogicalAnd( a );
}

template< typename Real,
          typename Device,
          typename Index >
auto
binaryAnd( const Containers::VectorView< Real, Device, Index >& a )
-> decltype( Containers::Expressions::ExpressionBinaryAnd( a ) )
{
   return Containers::Expressions::ExpressionBinaryAnd( a );
}

////
// Dot product - the same as scalar product, just for convenience
template< typename Real, typename Device, typename Index, typename ET >
auto
dot( const Containers::VectorView< Real, Device, Index >& a, const ET& b )
->decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename ET, typename Real, typename Device, typename Index >
auto
dot( const ET& a, const Containers::VectorView< Real, Device, Index >& b )
->decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename Real1, typename Real2, typename Device, typename Index1, typename Index2 >
//auto
Real1 dot( const Containers::VectorView< Real1, Device, Index1 >& a, const Containers::VectorView< Real2, Device, Index2 >& b )
//->decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

////
// TODO: Replace this with multiplication when its safe
template< typename Real, typename Device, typename Index, typename ET >
Containers::VectorView< Real, Device, Index >
Scale( const Containers::VectorView< Real, Device, Index >& a, const ET& b )
{
   Containers::VectorView< Real, Device, Index > result = Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, ET, Containers::Expressions::Multiplication >( a, b );
   return result;
}

template< typename ET, typename Real, typename Device, typename Index >
Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Multiplication >
Scale( const ET& a, const Containers::VectorView< Real, Device, Index >& b )
{
   Containers::VectorView< Real, Device, Index > result =  Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Multiplication >( a, b );
   return result;
}

template< typename Real1, typename Real2, typename Device, typename Index >
Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Multiplication >
Scale( const Containers::VectorView< Real1, Device, Index >& a, const Containers::VectorView< Real2, Device, Index >& b )
{
   Containers::VectorView< Real1, Device, Index > result =  Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Multiplication >( a, b );
   return result;
}

} // namespace TNL
