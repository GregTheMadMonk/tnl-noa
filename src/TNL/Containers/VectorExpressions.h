/***************************************************************************
                          VectorExpressions.h  -  description
                             -------------------
    begin                : Jun 27, 2019
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

#include "VectorView.h"

namespace TNL {

////
// All operations are supposed to be in namespace TNL
//   namespace Containers {

////
// Addition
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::BinaryExpressionTemplate< 
   typename Containers::VectorView< Real, Device, Index >::ConstViewType,
   ET,
   Containers::Expressions::Addition >
operator+( const Containers::Vector< Real, Device, Index >& a, const ET& b )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ConstView, ET, Containers::Expressions::Addition >( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< 
   ET,
   typename Containers::VectorView< Real, Device, Index >::ConstViewType,
   Containers::Expressions::Addition >
operator+( const ET& a, const Containers::Vector< Real, Device, Index >& b )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ET, ConstView, Containers::Expressions::Addition >( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< 
   typename Containers::VectorView< Real1, Device, Index >::ConstViewType,
   typename Containers::VectorView< Real2, Device, Index >::ConstViewType,
   Containers::Expressions::Addition >
operator+( const Containers::Vector< Real1, Device, Index >& a, const Containers::Vector< Real2, Device, Index >& b )
{
   using ConstView1 = typename Containers::Vector< Real1, Device, Index >::ConstViewType;
   using ConstView2 = typename Containers::Vector< Real2, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Addition >( a.getView(), b.getView() );
}

////
// Subtraction
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, ET, Containers::Expressions::Subtraction >
operator-( const Containers::Vector< Real, Device, Index >& a, const ET& b )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ConstView, ET, Containers::Expressions::Subtraction >( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Subtraction >
operator-( const ET& a, const Containers::Vector< Real, Device, Index >& b )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ET, ConstView, Containers::Expressions::Subtraction >( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< 
   typename Containers::VectorView< Real1, Device, Index >::ConstViewType,
   typename Containers::VectorView< Real2, Device, Index >::ConstViewType,
   Containers::Expressions::Subtraction >
operator-( const Containers::Vector< Real1, Device, Index >& a, const Containers::Vector< Real2, Device, Index >& b )
{
   using ConstView1 = typename Containers::Vector< Real1, Device, Index >::ConstViewType;
   using ConstView2 = typename Containers::Vector< Real2, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Subtraction >( a.getView(), b.getView() );
}

////
// Multiplication
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, ET, Containers::Expressions::Multiplication >
operator*( const Containers::Vector< Real, Device, Index >& a, const ET& b )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ConstView, ET, Containers::Expressions::Multiplication >( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Multiplication >
operator*( const ET& a, const Containers::Vector< Real, Device, Index >& b )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ET, ConstView, Containers::Expressions::Multiplication >( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Multiplication >
operator*( const Containers::Vector< Real1, Device, Index >& a, const Containers::Vector< Real2, Device, Index >& b )
{
   using ConstView1 = typename Containers::Vector< Real1, Device, Index >::ConstViewType;
   using ConstView2 = typename Containers::Vector< Real2, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Multiplication >( a.getView(), b.getView() );
}

////
// Division
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, ET, Containers::Expressions::Division >
operator/( const Containers::Vector< Real, Device, Index >& a, const ET& b )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ConstView, ET, Containers::Expressions::Division >( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Division >
operator/( const ET& a, const Containers::Vector< Real, Device, Index >& b )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ET, ConstView, Containers::Expressions::Division >( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Division >
operator/( const Containers::Vector< Real1, Device, Index >& a, const Containers::Vector< Real2, Device, Index >& b )
{
   using ConstView1 = typename Containers::Vector< Real1, Device, Index >::ConstViewType;
   using ConstView2 = typename Containers::Vector< Real2, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Division >( a.getView(), b.getView() );
}

////
// Min
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, ET, Containers::Expressions::Min >
min( const Containers::Vector< Real, Device, Index >& a, const ET& b )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ConstView, ET, Containers::Expressions::Min >( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Min >
min( const ET& a, const Containers::Vector< Real, Device, Index >& b )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ET, ConstView, Containers::Expressions::Min >( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Min >
min( const Containers::Vector< Real1, Device, Index >& a, const Containers::Vector< Real2, Device, Index >& b )
{
   using ConstView1 = typename Containers::Vector< Real1, Device, Index >::ConstViewType;
   using ConstView2 = typename Containers::Vector< Real2, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Min >( a.getView(), b.getView() );
}

////
// Max
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, ET, Containers::Expressions::Max >
max( const Containers::Vector< Real, Device, Index >& a, const ET& b )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ConstView, ET, Containers::Expressions::Max >( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Max >
max( const ET& a, const Containers::Vector< Real, Device, Index >& b )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ET, ConstView, Containers::Expressions::Max >( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Max >
max( const Containers::Vector< Real1, Device, Index >& a, const Containers::Vector< Real2, Device, Index >& b )
{
   using ConstView1 = typename Containers::Vector< Real1, Device, Index >::ConstViewType;
   using ConstView2 = typename Containers::Vector< Real2, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Max >( a.getView(), b.getView() );
}

////
// Comparison operations - operator ==
template< typename Real, typename Device, typename Index, typename ET >
bool operator==( const Containers::Vector< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::ComparisonEQ( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator==( const ET& a, const Containers::Vector< Real, Device, Index >& b )
{
   return Containers::Expressions::ComparisonEQ( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index >
bool operator==( const Containers::Vector< Real1, Device1, Index >& a, const Containers::Vector< Real2, Device2, Index >& b )
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
bool operator!=( const Containers::Vector< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::ComparisonNE( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator!=( const ET& a, const Containers::Vector< Real, Device, Index >& b )
{
   return Containers::Expressions::ComparisonNE( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index >
bool operator!=( const Containers::Vector< Real1, Device1, Index >& a, const Containers::Vector< Real2, Device2, Index >& b )
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
bool operator<( const Containers::Vector< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::ComparisonLT( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator<( const ET& a, const Containers::Vector< Real, Device, Index >& b )
{
   return Containers::Expressions::ComparisonLT( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
bool operator<( const Containers::Vector< Real1, Device, Index >& a, const Containers::Vector< Real2, Device, Index >& b )
{
   return Containers::Expressions::ComparisonLT( a.getView(), b.getView() );
}

////
// Comparison operations - operator <=
template< typename Real, typename Device, typename Index, typename ET >
bool operator<=( const Containers::Vector< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::ComparisonLE( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator<=( const ET& a, const Containers::Vector< Real, Device, Index >& b )
{
   return Containers::Expressions::ComparisonLE( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
bool operator<=( const Containers::Vector< Real1, Device, Index >& a, const Containers::Vector< Real2, Device, Index >& b )
{
   return Containers::Expressions::ComparisonLE( a.getView(), b.getView() );
}

////
// Comparison operations - operator >
template< typename Real, typename Device, typename Index, typename ET >
bool operator>( const Containers::Vector< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::ComparisonGT( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator>( const ET& a, const Containers::Vector< Real, Device, Index >& b )
{
   return Containers::Expressions::ComparisonGT( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
bool operator>( const Containers::Vector< Real1, Device, Index >& a, const Containers::Vector< Real2, Device, Index >& b )
{
   return Containers::Expressions::ComparisonGT( a.getView(), b.getView() );
}

////
// Comparison operations - operator >=
template< typename Real, typename Device, typename Index, typename ET >
bool operator>=( const Containers::Vector< Real, Device, Index >& a, const ET& b )
{
   return Containers::Expressions::ComparisonGE( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator>=( const ET& a, const Containers::Vector< Real, Device, Index >& b )
{
   return Containers::Expressions::ComparisonGE( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
bool operator>=( const Containers::Vector< Real1, Device, Index >& a, const Containers::Vector< Real2, Device, Index >& b )
{
   return Containers::Expressions::ComparisonGE( a.getView(), b.getView() );
}

////
// Minus
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Minus >
operator-( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Minus >( a.getView() );
}

////
// Abs
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Abs >
abs( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Abs >( a.getView() );
}

////
// Sine
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Sin >
sin( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Sin >( a.getView() );
}

////
// Cosine
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Cos >
cos( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Cos >( a.getView() );
}

////
// Tangent
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Tan >
tan( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Tan >( a.getView() );
}

////
// Sqrt
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Sqrt >
sqrt( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Sqrt >( a.getView() );
}

////
// Cbrt
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Cbrt >
cbrt( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Cbrt >( a.getView() );
}

////
// Power
template< typename Real, typename Device, typename Index, typename ExpType >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Pow, ExpType >
pow( const Containers::Vector< Real, Device, Index >& a, const ExpType& exp )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Pow, ExpType >( a.getView(), exp );
}

////
// Floor
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Floor >
floor( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Floor >( a.getView() );
}

////
// Ceil
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Ceil >
ceil( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Ceil >( a.getView() );
}

////
// Acos
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Acos >
acos( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Acos >( a.getView() );
}

////
// Asin
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Asin >
asin( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Asin >( a.getView() );
}

////
// Atan
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Atan >
atan( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Atan >( a.getView() );
}

////
// Cosh
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Cosh >
cosh( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Cosh >( a.getView() );
}

////
// Tanh
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Tanh >
tanh( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Tanh >( a.getView() );
}

////
// Log
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Log >
log( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Log >( a.getView() );
}

////
// Log10
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Log10 >
log10( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Log10 >( a.getView() );
}

////
// Log2
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Log2 >
log2( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Log2 >( a.getView() );
}

////
// Exp
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Exp >
exp( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Exp >( a.getView() );
}

////
// Sign
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, Containers::Expressions::Sign >
sign( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Sign >( a.getView() );
}

////
// Vertical operations - min
template< typename Real,
          typename Device,
          typename Index >
typename Containers::VectorView< Real, Device, Index >::RealType
min( const Containers::Vector< Real, Device, Index >& a )
{
   return Containers::Expressions::ExpressionMin( a.getView() );
}

template< typename Real,
          typename Device,
          typename Index >
typename Containers::VectorView< Real, Device, Index >::RealType
argMin( const Containers::Vector< Real, Device, Index >& a, Index& arg )
{
   return Containers::Expressions::ExpressionArgMin( a.getView(), arg );
}

template< typename Real,
          typename Device,
          typename Index >
typename Containers::VectorView< Real, Device, Index >::RealType
max( const Containers::Vector< Real, Device, Index >& a )
{
   return Containers::Expressions::ExpressionMax( a.getView() );
}

template< typename Real,
          typename Device,
          typename Index >
typename Containers::VectorView< Real, Device, Index >::RealType
argMax( const Containers::Vector< Real, Device, Index >& a, Index& arg )
{
   return Containers::Expressions::ExpressionArgMax( a.getView(), arg );
}

template< typename Real,
          typename Device,
          typename Index >
auto
sum( const Containers::Vector< Real, Device, Index >& a ) -> decltype( Containers::Expressions::ExpressionSum( a.getView() ) )
{
   return Containers::Expressions::ExpressionSum( a.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename Real2 >
auto
lpNorm( const Containers::Vector< Real, Device, Index >& a, const Real2& p ) -> decltype( Containers::Expressions::ExpressionLpNorm( a.getView(), p ) )
{
   return Containers::Expressions::ExpressionLpNorm( a.getView(), p );
}

template< typename Real,
          typename Device,
          typename Index >
auto
product( const Containers::Vector< Real, Device, Index >& a ) -> decltype( Containers::Expressions::ExpressionProduct( a.getView() ) )
{
   return Containers::Expressions::ExpressionProduct( a.getView() );
}

template< typename Real,
          typename Device,
          typename Index >
bool
logicalOr( const Containers::Vector< Real, Device, Index >& a )
{
   return Containers::Expressions::ExpressionLogicalOr( a.getView() );
}

template< typename Real,
          typename Device,
          typename Index >
auto
binaryOr( const Containers::Vector< Real, Device, Index >& a ) -> decltype( Containers::Expressions::ExpressionBinaryOr( a.getView() ) )
{
   return Containers::Expressions::ExpressionBinaryOr( a.getView() );
}

template< typename Real,
          typename Device,
          typename Index >
bool
logicalAnd( const Containers::Vector< Real, Device, Index >& a )
{
   return Containers::Expressions::ExpressionLogicalAnd( a.getView() );
}

template< typename Real,
          typename Device,
          typename Index >
auto
binaryAnd( const Containers::Vector< Real, Device, Index >& a ) -> decltype( Containers::Expressions::ExpressionBinaryAnd( a.getView() ) )
{
   return Containers::Expressions::ExpressionBinaryAnd( a.getView() );
}

////
// Scalar product
template< typename Real, typename Device, typename Index, typename ET >
Real operator,( const Containers::Vector< Real, Device, Index >& a, const ET& b )
{
   return TNL::sum( a.getView() * b );
}

template< typename ET, typename Real, typename Device, typename Index >
Real operator,( const ET& a, const Containers::Vector< Real, Device, Index >& b )
{
   return TNL::sum( a * b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
auto operator,( const Containers::Vector< Real1, Device, Index >& a, const Containers::Vector< Real2, Device, Index >& b )
->decltype( TNL::sum( a.getView() * b.getView() ) )
{
   return TNL::sum( a.getView() * b.getView() );
}

////
// Dot product - the same as scalar product, just for convenience
template< typename Real, typename Device, typename Index, typename ET >
auto dot( const Containers::Vector< Real, Device, Index >& a, const ET& b )->decltype( TNL::sum( a.getView() * b ) )
{
   return TNL::sum( a.getView() * b );
}

template< typename ET, typename Real, typename Device, typename Index >
auto dot( const ET& a, const Containers::Vector< Real, Device, Index >& b )->decltype( TNL::sum( a * b.getView() ) )
{
   return TNL::sum( a * b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
auto dot( const Containers::Vector< Real1, Device, Index >& a, const Containers::Vector< Real2, Device, Index >& b )
->decltype( TNL::sum( a.getView() * b.getView() ) )
{
   return TNL::sum( a.getView() * b.getView() );
}

////
// TODO: Replace this with multiplication when its safe
template< typename Real, typename Device, typename Index, typename ET >
Containers::VectorView< Real, Device, Index >
Scale( const Containers::Vector< Real, Device, Index >& a, const ET& b )
{
   Containers::VectorView< Real, Device, Index > result = Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real, Device, Index >, ET, Containers::Expressions::Multiplication >( a.getView(), b );
   return result;
}

template< typename ET, typename Real, typename Device, typename Index >
Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Multiplication >
Scale( const ET& a, const Containers::Vector< Real, Device, Index >& b )
{
   Containers::VectorView< Real, Device, Index > result =  Containers::Expressions::BinaryExpressionTemplate< ET, Containers::VectorView< Real, Device, Index >, Containers::Expressions::Multiplication >( a, b.getView() );
   return result;
}

template< typename Real1, typename Real2, typename Device, typename Index >
Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Multiplication >
Scale( const Containers::Vector< Real1, Device, Index >& a, const Containers::Vector< Real2, Device, Index >& b )
{
   Containers::VectorView< Real1, Device, Index > result =  Containers::Expressions::BinaryExpressionTemplate< Containers::VectorView< Real1, Device, Index >, Containers::VectorView< Real2, Device, Index >, Containers::Expressions::Multiplication >( a.getView(), b.getView() );
   return result;
}

} // namespace TNL
