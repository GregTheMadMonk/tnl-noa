/***************************************************************************
                          DistributedVectorExpressions.h  -  description
                             -------------------
    begin                : Jul 1, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <limits>
#include <TNL/Containers/DistributedVector.h>
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
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   ET,
   Containers::Expressions::Addition >
operator+( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView, ET, Containers::Expressions::Addition >( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< 
   ET,
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Addition >
operator+( const ET& a, const Containers::DistributedVector< Real, Device, Index, Communicator >& b )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, ConstView, Containers::Expressions::Addition >( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< 
   typename Containers::DistributedVector< Real1, Device, Index, Communicator >::ConstLocalVectorViewType,
   typename Containers::DistributedVector< Real2, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Addition >
operator+( const Containers::DistributedVector< Real1, Device, Index, Communicator >& a, const Containers::DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename Containers::DistributedVector< Real1, Device, Index >::ConstLocalVectorViewType;
   using ConstView2 = typename Containers::DistributedVector< Real2, Device, Index >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Addition >( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Subtraction
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
const Containers::Expressions::DistributedBinaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   ET,
   Containers::Expressions::Subtraction >
operator-( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView, ET, Containers::Expressions::Subtraction >( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   ET,
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Subtraction >
operator-( const ET& a, const Containers::DistributedVector< Real, Device, Index, Communicator >& b )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, ConstView, Containers::Expressions::Subtraction >( a.getLocalVectorView(), b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< 
   typename Containers::DistributedVector< Real1, Device, Index, Communicator >::ConstLocalVectorViewType,
   typename Containers::DistributedVector< Real2, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Subtraction >
operator-( const Containers::DistributedVector< Real1, Device, Index, Communicator >& a, const Containers::DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename Containers::DistributedVector< Real1, Device, Index >::ConstLocalVectorViewType;
   using ConstView2 = typename Containers::DistributedVector< Real2, Device, Index >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Subtraction >( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Multiplication
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
const Containers::Expressions::DistributedBinaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   ET, 
   Containers::Expressions::Multiplication >
operator*( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView, ET, Containers::Expressions::Multiplication >( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< 
   ET,
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Multiplication >
operator*( const ET& a, const Containers::DistributedVector< Real, Device, Index, Communicator >& b )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, ConstView, Containers::Expressions::Multiplication >( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< 
   typename Containers::DistributedVector< Real1, Device, Index, Communicator >::ConstLocalVectorViewType,
   typename Containers::DistributedVector< Real2, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Multiplication >
operator*( const Containers::DistributedVector< Real1, Device, Index, Communicator >& a, const Containers::DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename Containers::DistributedVector< Real1, Device, Index >::ConstLocalVectorViewType;
   using ConstView2 = typename Containers::DistributedVector< Real2, Device, Index >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Multiplication >( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Division
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
const Containers::Expressions::DistributedBinaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   ET, 
   Containers::Expressions::Division >
operator/( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView, ET, Containers::Expressions::Division >( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   ET,
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Division >
operator/( const ET& a, const Containers::DistributedVector< Real, Device, Index, Communicator >& b )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, ConstView, Containers::Expressions::Division >( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< 
   typename Containers::DistributedVector< Real1, Device, Index, Communicator >::ConstLocalVectorViewType,
   typename Containers::DistributedVector< Real2, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Division >
operator/( const Containers::DistributedVector< Real1, Device, Index, Communicator >& a, const Containers::DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename Containers::DistributedVector< Real1, Device, Index >::ConstLocalVectorViewType;
   using ConstView2 = typename Containers::DistributedVector< Real2, Device, Index >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Division >( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Min
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
const Containers::Expressions::DistributedBinaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   ET,
   Containers::Expressions::Min >
min( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView, ET, Containers::Expressions::Min >( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< 
   ET,
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Min >
min( const ET& a, const Containers::DistributedVector< Real, Device, Index, Communicator >& b )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, ConstView, Containers::Expressions::Min >( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::DistributedVector< Real1, Device, Index, Communicator >::ConstLocalVectorViewType,
   typename Containers::DistributedVector< Real2, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Min >
min( const Containers::DistributedVector< Real1, Device, Index, Communicator >& a, const Containers::DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename Containers::DistributedVector< Real1, Device, Index >::ConstLocalVectorViewType;
   using ConstView2 = typename Containers::DistributedVector< Real2, Device, Index >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Min >( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Max
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
const Containers::Expressions::DistributedBinaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   ET,
   Containers::Expressions::Max >
max( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView, ET, Containers::Expressions::Max >( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< 
   ET,
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Max >
max( const ET& a, const Containers::DistributedVector< Real, Device, Index, Communicator >& b )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, ConstView, Containers::Expressions::Max >( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< 
   typename Containers::DistributedVector< Real1, Device, Index, Communicator >::ConstLocalVectorViewType,
   typename Containers::DistributedVector< Real2, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Max >
max( const Containers::DistributedVector< Real1, Device, Index, Communicator >& a, const Containers::DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename Containers::DistributedVector< Real1, Device, Index >::ConstLocalVectorViewType;
   using ConstView2 = typename Containers::DistributedVector< Real2, Device, Index >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Max >( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Comparison operations - operator ==
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
bool operator==( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   return Containers::Expressions::ComparisonEQ( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
bool operator==( const ET& a, const Containers::DistributedVector< Real, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonEQ( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index, typename Communicator >
bool operator==( const Containers::DistributedVector< Real1, Device1, Index >& a, const Containers::DistributedVector< Real2, Device2, Index >& b )
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
bool operator!=( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   return Containers::Expressions::ComparisonNE( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
bool operator!=( const ET& a, const Containers::DistributedVector< Real, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonNE( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index, typename Communicator >
bool operator!=( const Containers::DistributedVector< Real1, Device1, Index >& a, const Containers::DistributedVector< Real2, Device2, Index >& b )
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
bool operator<( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   return Containers::Expressions::ComparisonLT( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
bool operator<( const ET& a, const Containers::DistributedVector< Real, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonLT( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator<( const Containers::DistributedVector< Real1, Device, Index, Communicator >& a, const Containers::DistributedVector< Real2, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonLT( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Comparison operations - operator <=
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
bool operator<=( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   return Containers::Expressions::ComparisonLE( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
bool operator<=( const ET& a, const Containers::DistributedVector< Real, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonLE( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator<=( const Containers::DistributedVector< Real1, Device, Index, Communicator >& a, const Containers::DistributedVector< Real2, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonLE( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Comparison operations - operator >
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
bool operator>( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   return Containers::Expressions::ComparisonGT( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
bool operator>( const ET& a, const Containers::DistributedVector< Real, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonGT( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator>( const Containers::DistributedVector< Real1, Device, Index, Communicator >& a, const Containers::DistributedVector< Real2, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonGT( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Comparison operations - operator >=
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
bool operator>=( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   return Containers::Expressions::ComparisonGE( a.getLocalVectorView(), b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
bool operator>=( const ET& a, const Containers::DistributedVector< Real, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonGE( a, b.getLocalVectorView() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator>=( const Containers::DistributedVector< Real1, Device, Index, Communicator >& a, const Containers::DistributedVector< Real2, Device, Index, Communicator >& b )
{
   return Containers::Expressions::ComparisonGE( a.getLocalVectorView(), b.getLocalVectorView() );
}

////
// Minus
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >,
   Containers::Expressions::Minus >
operator-( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Minus >( a.getLocalVectorView() );
}

////
// Abs
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >, 
   Containers::Expressions::Abs >
abs( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Abs >( a.getLocalVectorView() );
}

////
// Sine
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >, 
   Containers::Expressions::Sin >
sin( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Sin >( a.getLocalVectorView() );
}

////
// Cosine
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >, 
   Containers::Expressions::Cos >
cos( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Cos >( a.getLocalVectorView() );
}

////
// Tangent
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >, 
   Containers::Expressions::Tan >
tan( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Tan >( a.getLocalVectorView() );
}

////
// Sqrt
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >, 
   Containers::Expressions::Sqrt >
sqrt( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Sqrt >( a.getLocalVectorView() );
}

////
// Cbrt
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >, 
   Containers::Expressions::Cbrt >
cbrt( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Cbrt >( a.getLocalVectorView() );
}

////
// Power
template< typename Real, typename Device, typename Index, typename Communicator, typename ExpType >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >,
   Containers::Expressions::Pow, ExpType >
pow( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ExpType& exp )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Pow, ExpType >( a.getLocalVectorView(), exp );
}

////
// Floor
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >,
   Containers::Expressions::Floor >
floor( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Floor >( a.getLocalVectorView() );
}

////
// Ceil
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >, 
   Containers::Expressions::Ceil >
ceil( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Ceil >( a.getLocalVectorView() );
}

////
// Acos
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >, 
   Containers::Expressions::Acos >
acos( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Acos >( a.getLocalVectorView() );
}

////
// Asin
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >, 
   Containers::Expressions::Asin >
asin( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Asin >( a.getLocalVectorView() );
}

////
// Atan
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >, 
   Containers::Expressions::Atan >
atan( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Atan >( a.getLocalVectorView() );
}

////
// Cosh
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >, 
   Containers::Expressions::Cosh >
cosh( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Cosh >( a.getLocalVectorView() );
}

////
// Tanh
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >,
   Containers::Expressions::Tanh >
tanh( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Tanh >( a.getLocalVectorView() );
}

////
// Log
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >,
   Containers::Expressions::Log >
log( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Log >( a.getLocalVectorView() );
}

////
// Log10
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >,
   Containers::Expressions::Log10 >
log10( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Log10 >( a.getLocalVectorView() );
}

////
// Log2
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >,
   Containers::Expressions::Log2 >
log2( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Log2 >( a.getLocalVectorView() );
}

////
// Exp
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >,
   Containers::Expressions::Exp >
exp( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Exp >( a.getLocalVectorView() );
}

////
// Sign
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   Containers::DistributedVector< Real, Device, Index, Communicator >,
   Containers::Expressions::Sign >
sign( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Sign >( a.getLocalVectorView() );
}

////
// Vertical operations - min
template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
min( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
-> decltype( Containers::Expressions::ExpressionMin( a.getLocalVectorView() ) )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = std::numeric_limits< Real >::max();
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionMin( a.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MIN, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
argMin( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, Index& arg )
-> decltype( Containers::Expressions::ExpressionArgMin( a.getLocalVectorView(), arg ) )
{
   throw Exceptions::NotImplementedError( "agrMin for distributed vector view is not implemented yet." );
   return Containers::Expressions::ExpressionArgMin( a.getLocalVectorView(), arg );
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
max( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
-> decltype( Containers::Expressions::ExpressionMax( a.getLocalVectorView() ) )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = std::numeric_limits< Real >::min();
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionMax( a.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MAX, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
argMax( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, Index& arg )
-> decltype( Containers::Expressions::ExpressionArgMax( a.getLocalVectorView(), arg ) )
{
   throw Exceptions::NotImplementedError( "agrMax for distributed vector view is not implemented yet." );
   return Containers::Expressions::ExpressionArgMax( a.getLocalVectorView(), arg );
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
sum( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
-> decltype( Containers::Expressions::ExpressionSum( a.getLocalVectorView() ) )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionSum( a.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator,
          typename Real2 >
auto
lpNorm( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const Real2& p )
-> decltype( TNL::pow( Containers::Expressions::ExpressionLpNorm( a.getLocalVectorView(), p ), p ) )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = ( Real ) 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::pow( Containers::Expressions::ExpressionLpNorm( a.getLocalVectorView(), p ), p );
      CommunicatorType::template Allreduce< Real >( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return TNL::pow( result, 1.0 / p );
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
product( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
-> decltype( Containers::Expressions::ExpressionProduct( a.getLocalVectorView() ) )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = ( Real ) 1.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionProduct( a.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_PROD, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
bool
logicalOr( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   bool result = false;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionLogicalOr( a.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LOR, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
typename Containers::DistributedVector< Real, Device, Index, Communicator >::RealType
binaryOr( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = ( Real ) 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionBinaryOr( a.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_BOR, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
bool
logicalAnd( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = std::numeric_limits< Real >::max();
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionLogicalAnd( a.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
typename Containers::DistributedVector< Real, Device, Index, Communicator >::RealType
binaryAnd( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   bool result = true;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionBinaryAnd( a.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_BAND, a.getCommunicationGroup() );
   }
   return result;
}

////
// Scalar product
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
auto
operator,( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
-> decltype( TNL::sum( a.getLocalVectorView() * b ) )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::sum( a.getLocalVectorView() * b );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
//Real
auto
operator,( const ET& a, const Containers::DistributedVector< Real, Device, Index, Communicator >& b ) 
-> decltype( TNL::sum( a * b.getLocalVectorView() ) )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::sum( a * b.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto operator,( const Containers::DistributedVector< Real1, Device, Index, Communicator >& a, const Containers::DistributedVector< Real2, Device, Index, Communicator >& b )
->decltype( TNL::sum( a.getLocalVectorView() * b.getLocalVectorView() ) )
{
   using CommunicatorType = typename Containers::DistributedVector< Real1, Device, Index, Communicator >::CommunicatorType;
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
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
auto
dot( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
-> decltype( ( a, b ) )
{
   return ( a, b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
auto
dot( const ET& a, const Containers::DistributedVector< Real, Device, Index, Communicator >& b )
-> decltype( ( a, b ) )
{
   return ( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
dot( const Containers::DistributedVector< Real1, Device, Index, Communicator >& a, const Containers::DistributedVector< Real2, Device, Index, Communicator >& b )
-> decltype( ( a, b ) )
{
   return ( a, b );
}

////
// TODO: Replace this with multiplication when its safe
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
Containers::DistributedVector< Real, Device, Index, Communicator >
Scale( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   Containers::DistributedVector< Real, Device, Index, Communicator > result = Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVector< Real, Device, Index, Communicator >, ET, Containers::Expressions::Multiplication >( a, b );
   return result;
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Multiplication >
Scale( const ET& a, const Containers::DistributedVector< Real, Device, Index, Communicator >& b )
{
   Containers::DistributedVector< Real, Device, Index, Communicator > result =  Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVector< Real, Device, Index, Communicator >, Containers::Expressions::Multiplication >( a, b );
   return result;
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVector< Real1, Device, Index, Communicator >, Containers::DistributedVector< Real2, Device, Index, Communicator >, Containers::Expressions::Multiplication >
Scale( const Containers::DistributedVector< Real1, Device, Index, Communicator >& a, const Containers::DistributedVector< Real2, Device, Index, Communicator >& b )
{
   Containers::DistributedVector< Real1, Device, Index, Communicator > result =  Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVector< Real1, Device, Index, Communicator >, Containers::DistributedVector< Real2, Device, Index, Communicator >, Containers::Expressions::Multiplication >( a, b );
   return result;
}

} // namespace TNL
