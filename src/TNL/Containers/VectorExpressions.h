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
#include "Vector.h"

namespace TNL {
   namespace Containers {

////
// Addition
template< typename Real, typename Device, typename Index, typename ET >
const Expressions::BinaryExpressionTemplate< 
   typename VectorView< Real, Device, Index >::ConstViewType,
   ET,
   Expressions::Addition >
operator+( const Vector< Real, Device, Index >& a, const ET& b )
{
   using ConstView = typename Vector< Real, Device, Index >::ConstViewType;
   return Expressions::BinaryExpressionTemplate< ConstView, ET, Expressions::Addition >( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Expressions::BinaryExpressionTemplate< 
   ET,
   typename VectorView< Real, Device, Index >::ConstViewType,
   Expressions::Addition >
operator+( const ET& a, const Vector< Real, Device, Index >& b )
{
   using ConstView = typename Vector< Real, Device, Index >::ConstViewType;
   return Expressions::BinaryExpressionTemplate< ET, ConstView, Expressions::Addition >( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Expressions::BinaryExpressionTemplate< 
   typename Vector< Real1, Device, Index >::ConstViewType,
   typename VectorView< Real2, Device, Index >::ConstViewType,
   Expressions::Addition >
operator+( const Vector< Real1, Device, Index >& a, const Vector< Real2, Device, Index >& b )
{
   using ConstView1 = typename Vector< Real1, Device, Index >::ConstViewType;
   using ConstView2 = typename Vector< Real2, Device, Index >::ConstViewType;
   return Expressions::BinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Addition >( a.getView(), b.getView() );
}

////
// Subtraction
template< typename Real, typename Device, typename Index, typename ET >
const Expressions::BinaryExpressionTemplate< typename Vector< Real, Device, Index >::ConstViewType, ET, Expressions::Subtraction >
operator-( const Vector< Real, Device, Index >& a, const ET& b )
{
   using ConstView = typename Vector< Real, Device, Index >::ConstViewType;
   return Expressions::BinaryExpressionTemplate< ConstView, ET, Expressions::Subtraction >( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Expressions::BinaryExpressionTemplate< ET, typename Vector< Real, Device, Index >::ConstViewType, Expressions::Subtraction >
operator-( const ET& a, const Vector< Real, Device, Index >& b )
{
   using ConstView = typename Vector< Real, Device, Index >::ConstViewType;
   return Expressions::BinaryExpressionTemplate< ET, ConstView, Expressions::Subtraction >( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Expressions::BinaryExpressionTemplate< 
   typename Vector< Real1, Device, Index >::ConstViewType,
   typename VectorView< Real2, Device, Index >::ConstViewType,
   Expressions::Subtraction >
operator-( const Vector< Real1, Device, Index >& a, const Vector< Real2, Device, Index >& b )
{
   using ConstView1 = typename Vector< Real1, Device, Index >::ConstViewType;
   using ConstView2 = typename Vector< Real2, Device, Index >::ConstViewType;
   return Expressions::BinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Subtraction >( a.getView(), b.getView() );
}

////
// Multiplication
template< typename Real, typename Device, typename Index, typename ET >
const Expressions::BinaryExpressionTemplate< typename Vector< Real, Device, Index >::ConstViewType, ET, Expressions::Multiplication >
operator*( const Vector< Real, Device, Index >& a, const ET& b )
{
   using ConstView = typename Vector< Real, Device, Index >::ConstViewType;
   return Expressions::BinaryExpressionTemplate< ConstView, ET, Expressions::Multiplication >( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Expressions::BinaryExpressionTemplate< ET, typename Vector< Real, Device, Index >::ConstViewType, Expressions::Multiplication >
operator*( const ET& a, const Vector< Real, Device, Index >& b )
{
   using ConstView = typename Vector< Real, Device, Index >::ConstViewType;
   return Expressions::BinaryExpressionTemplate< ET, ConstView, Expressions::Multiplication >( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Expressions::BinaryExpressionTemplate< typename Vector< Real1, Device, Index >::ConstViewType, typename Vector< Real2, Device, Index >::ConstViewType, Expressions::Multiplication >
operator*( const Vector< Real1, Device, Index >& a, const Vector< Real2, Device, Index >& b )
{
   using ConstView1 = typename Vector< Real1, Device, Index >::ConstViewType;
   using ConstView2 = typename Vector< Real2, Device, Index >::ConstViewType;
   return Expressions::BinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Multiplication >( a.getView(), b.getView() );
}

////
// Division
template< typename Real, typename Device, typename Index, typename ET >
const Expressions::BinaryExpressionTemplate< typename Vector< Real, Device, Index >::ConstViewType, ET, Expressions::Division >
operator/( const Vector< Real, Device, Index >& a, const ET& b )
{
   using ConstView = typename Vector< Real, Device, Index >::ConstViewType;
   return Expressions::BinaryExpressionTemplate< ConstView, ET, Expressions::Division >( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Expressions::BinaryExpressionTemplate< ET, typename Vector< Real, Device, Index >::ConstViewType, Expressions::Division >
operator/( const ET& a, const Vector< Real, Device, Index >& b )
{
   using ConstView = typename Vector< Real, Device, Index >::ConstViewType;
   return Expressions::BinaryExpressionTemplate< ET, ConstView, Expressions::Division >( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Expressions::BinaryExpressionTemplate< typename Vector< Real1, Device, Index >::ConstViewType, typename Vector< Real2, Device, Index >::ConstViewType, Expressions::Division >
operator/( const Vector< Real1, Device, Index >& a, const Vector< Real2, Device, Index >& b )
{
   using ConstView1 = typename Vector< Real1, Device, Index >::ConstViewType;
   using ConstView2 = typename Vector< Real2, Device, Index >::ConstViewType;
   return Expressions::BinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Division >( a.getView(), b.getView() );
}

////
// Comparison operations - operator ==
template< typename Real, typename Device, typename Index, typename ET >
bool operator==( const Vector< Real, Device, Index >& a, const ET& b )
{
   using ConstView = typename VectorView< Real, Device, Index >::ConstViewType;
   return Expressions::Comparison< ConstView, ET >::EQ( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator==( const ET& a, const Vector< Real, Device, Index >& b )
{
   using ConstView = typename VectorView< Real, Device, Index >::ConstViewType;
   return Expressions::Comparison< ET, ConstView >::EQ( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index >
bool operator==( const Vector< Real1, Device1, Index >& a, const Vector< Real2, Device2, Index >& b )
{
   if( a.getSize() != b.getSize() )
      return false;
   if( a.getSize() == 0 )
      return true;
   return Algorithms::ArrayOperations< Device1, Device2 >::
            compareMemory( a.getData(),
                           b.getData(),
                           a.getSize() );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index >
bool operator==( const VectorView< Real1, Device1, Index >& a, const Vector< Real2, Device2, Index >& b )
{
   if( a.getSize() != b.getSize() )
      return false;
   if( a.getSize() == 0 )
      return true;
   return Algorithms::ArrayOperations< Device1, Device2 >::
            compareMemory( a.getData(),
                           b.getData(),
                           a.getSize() );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index >
bool operator==( const Vector< Real1, Device1, Index >& a, const VectorView< Real2, Device2, Index >& b )
{
   if( a.getSize() != b.getSize() )
      return false;
   if( a.getSize() == 0 )
      return true;
   return Algorithms::ArrayOperations< Device1, Device2 >::
            compareMemory( a.getData(),
                           b.getData(),
                           a.getSize() );
}

////
// Comparison operations - operator !=
template< typename Real, typename Device, typename Index, typename ET >
bool operator!=( const Vector< Real, Device, Index >& a, const ET& b )
{
   using ConstView = typename VectorView< Real, Device, Index >::ConstViewType;
   return Expressions::Comparison< ConstView, ET >::NE( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator!=( const ET& a, const Vector< Real, Device, Index >& b )
{
   using ConstView = typename VectorView< Real, Device, Index >::ConstViewType;
   return Expressions::Comparison< ET, ConstView >::NE( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index >
bool operator!=( const Vector< Real1, Device1, Index >& a, const Vector< Real2, Device2, Index >& b )
{
   if( a.getSize() != b.getSize() )
      return false;
   if( a.getSize() == 0 )
      return true;
   return !Algorithms::ArrayOperations< Device1, Device2 >::
            compareMemory( a.getData(),
                           b.getData(),
                           a.getSize() );
}

////
// Comparison operations - operator <
template< typename Real, typename Device, typename Index, typename ET >
bool operator<( const Vector< Real, Device, Index >& a, const ET& b )
{
   using ConstView = typename VectorView< Real, Device, Index >::ConstViewType;
   return Expressions::Comparison< ConstView, ET >::LT( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator<( const ET& a, const Vector< Real, Device, Index >& b )
{
   using ConstView = typename VectorView< Real, Device, Index >::ConstViewType;
   return Expressions::Comparison< ET, ConstView >::LT( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
bool operator<( const Vector< Real1, Device, Index >& a, const Vector< Real2, Device, Index >& b )
{
   using ConstView1 = typename Vector< Real1, Device, Index >::ConstViewType;
   using ConstView2 = typename VectorView< Real2, Device, Index >::ConstViewType;
   return Expressions::Comparison< ConstView1, ConstView2 >::LT( a.getView(), b.getView() );
}

////
// Comparison operations - operator <=
template< typename Real, typename Device, typename Index, typename ET >
bool operator<=( const Vector< Real, Device, Index >& a, const ET& b )
{
   using ConstView = typename VectorView< Real, Device, Index >::ConstViewType;
   return Expressions::Comparison< ConstView, ET >::LE( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator<=( const ET& a, const Vector< Real, Device, Index >& b )
{
   using ConstView = typename VectorView< Real, Device, Index >::ConstViewType;
   return Expressions::Comparison< ET, ConstView >::LE( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
bool operator<=( const Vector< Real1, Device, Index >& a, const Vector< Real2, Device, Index >& b )
{
   using ConstView1 = typename Vector< Real1, Device, Index >::ConstViewType;
   using ConstView2 = typename VectorView< Real2, Device, Index >::ConstViewType;
   return Expressions::Comparison< ConstView1, ConstView2 >::LE( a.getView(), b.getView() );
}

////
// Comparison operations - operator >
template< typename Real, typename Device, typename Index, typename ET >
bool operator>( const Vector< Real, Device, Index >& a, const ET& b )
{
   using ConstView = typename VectorView< Real, Device, Index >::ConstViewType;
   return Expressions::Comparison< ConstView, ET >::GT( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator>( const ET& a, const Vector< Real, Device, Index >& b )
{
   using ConstView = typename VectorView< Real, Device, Index >::ConstViewType;
   return Expressions::Comparison< ET, ConstView >::GT( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
bool operator>( const Vector< Real1, Device, Index >& a, const Vector< Real2, Device, Index >& b )
{
   using ConstView1 = typename Vector< Real1, Device, Index >::ConstViewType;
   using ConstView2 = typename VectorView< Real2, Device, Index >::ConstViewType;
   return Expressions::Comparison< ConstView1, ConstView2 >::GT( a.getView(), b.getView() );
}

////
// Comparison operations - operator >=
template< typename Real, typename Device, typename Index, typename ET >
bool operator>=( const Vector< Real, Device, Index >& a, const ET& b )
{
   using ConstView = typename VectorView< Real, Device, Index >::ConstViewType;
   return Expressions::Comparison< ConstView, ET >::GE( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
bool operator>=( const ET& a, const Vector< Real, Device, Index >& b )
{
   using ConstView = typename VectorView< Real, Device, Index >::ConstViewType;
   return Expressions::Comparison< ET, ConstView >::GE( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
bool operator>=( const Vector< Real1, Device, Index >& a, const Vector< Real2, Device, Index >& b )
{
   using ConstView1 = typename Vector< Real1, Device, Index >::ConstViewType;
   using ConstView2 = typename VectorView< Real2, Device, Index >::ConstViewType;
   return Expressions::Comparison< ConstView1, ConstView2 >::GE( a.getView(), b.getView() );
}

////
// Minus
template< typename Real, typename Device, typename Index >
const Expressions::UnaryExpressionTemplate< typename Vector< Real, Device, Index >::ConstViewType, Expressions::Minus >
operator-( const Vector< Real, Device, Index >& a )
{
   using ConstView = typename Vector< Real, Device, Index >::ConstViewType;
   return Expressions::UnaryExpressionTemplate< ConstView, Expressions::Minus >( a.getView() );
}

////
// Scalar product
template< typename Real, typename Device, typename Index, typename ET >
Real operator,( const Vector< Real, Device, Index >& a, const ET& b )
{
   return TNL::sum( a.getView() * b );
}

template< typename ET, typename Real, typename Device, typename Index >
Real operator,( const ET& a, const Vector< Real, Device, Index >& b )
{
   return TNL::sum( a * b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
//auto 
// TODO: auto+decltype does not work with NVCC 10.1
Real1 operator,( const Vector< Real1, Device, Index >& a, const Vector< Real2, Device, Index >& b )
//->decltype( TNL::sum( a.getView() * b.getView() ) )
{
   return TNL::sum( a.getView() * b.getView() );
}

} //namespace Containers

////
// All operations are supposed to be in namespace TNL

////
// Min
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::BinaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, ET, Containers::Expressions::Min >
min( const Containers::Vector< Real, Device, Index >& a, const ET& b )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ConstView, ET, Containers::Expressions::Min >( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< ET, typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Min >
min( const ET& a, const Containers::Vector< Real, Device, Index >& b )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ET, ConstView, Containers::Expressions::Min >( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< typename Containers::Vector< Real1, Device, Index >::ConstViewType, typename Containers::Vector< Real2, Device, Index >::ConstViewType, Containers::Expressions::Min >
min( const Containers::Vector< Real1, Device, Index >& a, const Containers::Vector< Real2, Device, Index >& b )
{
   using ConstView1 = typename Containers::Vector< Real1, Device, Index >::ConstViewType;
   using ConstView2 = typename Containers::Vector< Real2, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Min >( a.getView(), b.getView() );
}

////
// Max
template< typename Real, typename Device, typename Index, typename ET >
const Containers::Expressions::BinaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, ET, Containers::Expressions::Max >
max( const Containers::Vector< Real, Device, Index >& a, const ET& b )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ConstView, ET, Containers::Expressions::Max >( a.getView(), b );
}

template< typename ET, typename Real, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< ET, typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Max >
max( const ET& a, const Containers::Vector< Real, Device, Index >& b )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ET, ConstView, Containers::Expressions::Max >( a, b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
const Containers::Expressions::BinaryExpressionTemplate< typename Containers::Vector< Real1, Device, Index >::ConstViewType, typename Containers::Vector< Real2, Device, Index >::ConstViewType, Containers::Expressions::Max >
max( const Containers::Vector< Real1, Device, Index >& a, const Containers::Vector< Real2, Device, Index >& b )
{
   using ConstView1 = typename Containers::Vector< Real1, Device, Index >::ConstViewType;
   using ConstView2 = typename Containers::Vector< Real2, Device, Index >::ConstViewType;
   return Containers::Expressions::BinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Max >( a.getView(), b.getView() );
}


////
// Abs
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Abs >
abs( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Abs >( a.getView() );
}

////
// Sine
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Sin >
sin( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Sin >( a.getView() );
}

////
// Cosine
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Cos >
cos( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Cos >( a.getView() );
}

////
// Tangent
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Tan >
tan( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Tan >( a.getView() );
}

////
// Sqrt
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Sqrt >
sqrt( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Sqrt >( a.getView() );
}

////
// Cbrt
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Cbrt >
cbrt( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Cbrt >( a.getView() );
}

////
// Power
template< typename Real, typename Device, typename Index, typename ExpType >
const Containers::Expressions::UnaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Pow, ExpType >
pow( const Containers::Vector< Real, Device, Index >& a, const ExpType& exp )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Pow, ExpType >( a.getView(), exp );
}

////
// Floor
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Floor >
floor( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Floor >( a.getView() );
}

////
// Ceil
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Ceil >
ceil( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Ceil >( a.getView() );
}

////
// Acos
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Acos >
acos( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Acos >( a.getView() );
}

////
// Asin
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Asin >
asin( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Asin >( a.getView() );
}

////
// Atan
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Atan >
atan( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Atan >( a.getView() );
}

////
// Cosh
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Cosh >
cosh( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Cosh >( a.getView() );
}

////
// Tanh
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Tanh >
tanh( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Tanh >( a.getView() );
}

////
// Log
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Log >
log( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Log >( a.getView() );
}

////
// Log10
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Log10 >
log10( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Log10 >( a.getView() );
}

////
// Log2
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Log2 >
log2( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Log2 >( a.getView() );
}

////
// Exp
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Exp >
exp( const Containers::Vector< Real, Device, Index >& a )
{
   using ConstView = typename Containers::Vector< Real, Device, Index >::ConstViewType;
   return Containers::Expressions::UnaryExpressionTemplate< ConstView, Containers::Expressions::Exp >( a.getView() );
}

////
// Sign
template< typename Real, typename Device, typename Index >
const Containers::Expressions::UnaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Sign >
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
   if( p == 1.0 )
      return Containers::Expressions::ExpressionLpNorm( a.getView(), p );
   if( p == 2.0 )
      return TNL::sqrt( Containers::Expressions::ExpressionLpNorm( a.getView(), p ) );
   return TNL::pow( Containers::Expressions::ExpressionLpNorm( a.getView(), p ), 1.0 / p );
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
// Dot product - the same as scalar product, just for convenience
// TODO: auto+decltype does not work with NVCC 10.1
template< typename Real, typename Device, typename Index, typename ET >
//auto
Real dot( const Containers::Vector< Real, Device, Index >& a, const ET& b )//->decltype( TNL::sum( a.getView() * b ) )
{
   return TNL::sum( a.getView() * b );
}

template< typename ET, typename Real, typename Device, typename Index >
//auto
Real dot( const ET& a, const Containers::Vector< Real, Device, Index >& b )//->decltype( TNL::sum( a * b.getView() ) )
{
   return TNL::sum( a * b.getView() );
}

template< typename Real1, typename Real2, typename Device, typename Index >
//auto
Real1 dot( const Containers::Vector< Real1, Device, Index >& a, const Containers::Vector< Real2, Device, Index >& b )
//->decltype( TNL::sum( a.getView() * b.getView() ) )
{
   return TNL::sum( a.getView() * b.getView() );
}

////
// TODO: Replace this with multiplication when its safe
template< typename Real, typename Device, typename Index, typename ET >
Containers::VectorView< Real, Device, Index >
Scale( const Containers::Vector< Real, Device, Index >& a, const ET& b )
{
   Containers::VectorView< Real, Device, Index > result = Containers::Expressions::BinaryExpressionTemplate< typename Containers::Vector< Real, Device, Index >::ConstViewType, ET, Containers::Expressions::Multiplication >( a.getView(), b );
   return result;
}

template< typename ET, typename Real, typename Device, typename Index >
Containers::Expressions::BinaryExpressionTemplate< ET, typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Multiplication >
Scale( const ET& a, const Containers::Vector< Real, Device, Index >& b )
{
   Containers::VectorView< Real, Device, Index > result =  Containers::Expressions::BinaryExpressionTemplate< ET, typename Containers::Vector< Real, Device, Index >::ConstViewType, Containers::Expressions::Multiplication >( a, b.getView() );
   return result;
}

template< typename Real1, typename Real2, typename Device, typename Index >
Containers::Expressions::BinaryExpressionTemplate< typename Containers::Vector< Real1, Device, Index >::ConstViewType, typename Containers::Vector< Real2, Device, Index >::ConstViewType, Containers::Expressions::Multiplication >
Scale( const Containers::Vector< Real1, Device, Index >& a, const Containers::Vector< Real2, Device, Index >& b )
{
   Containers::VectorView< Real1, Device, Index > result =  Containers::Expressions::BinaryExpressionTemplate< typename Containers::Vector< Real1, Device, Index >::ConstViewType, typename Containers::Vector< Real2, Device, Index >::ConstViewType, Containers::Expressions::Multiplication >( a.getView(), b.getView() );
   return result;
}

} // namespace TNL
