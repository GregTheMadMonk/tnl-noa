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
#include <TNL/Containers/DistributedVectorView.h>

namespace TNL {
   namespace Containers {

////
// All operations are supposed to be in namespace TNL

////
// Addition
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
const Expressions::DistributedBinaryExpressionTemplate<
   typename DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   ET,
   Expressions::Addition, Communicator >
operator+( const DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   using ConstView = typename DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView, ET, Expressions::Addition, Communicator >( a.getLocalVectorView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
const Expressions::DistributedBinaryExpressionTemplate< 
   ET,
   typename DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Expressions::Addition, Communicator >
operator+( const ET& a, const DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   using ConstView = typename DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ET, ConstView, Expressions::Addition, Communicator >( a, b.getLocalVectorView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
const Expressions::DistributedBinaryExpressionTemplate< 
   typename DistributedVectorView< Real1, Device, Index, Communicator >::ConstLocalVectorViewType,
   typename DistributedVectorView< Real2, Device, Index, Communicator >::ConstLocalVectorViewType,
   Expressions::Addition, Communicator >
operator+( const DistributedVectorView< Real1, Device, Index, Communicator >& a, const DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename DistributedVectorView< Real1, Device, Index, Communicator >::ConstLocalVectorViewType;
   using ConstView2 = typename DistributedVectorView< Real2, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Addition, Communicator >( a.getLocalVectorView(), b.getLocalVectorView(), b.getCommunicationGroup() );
}

////
// Subtraction
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
const Expressions::DistributedBinaryExpressionTemplate< 
   typename DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   ET,
   Expressions::Subtraction, Communicator >
operator-( const DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   using ConstView = typename DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView, ET, Expressions::Subtraction, Communicator >( a.getLocalVectorView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
const Expressions::DistributedBinaryExpressionTemplate<
   ET,
   typename DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Expressions::Subtraction, Communicator >
operator-( const ET& a, const DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   using ConstView = typename DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ET, ConstView, Expressions::Subtraction, Communicator >( a.getLocalVectorView(), b.getLocalVectorView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
const Expressions::DistributedBinaryExpressionTemplate< 
   typename DistributedVectorView< Real1, Device, Index, Communicator >::ConstLocalVectorViewType,
   typename DistributedVectorView< Real2, Device, Index, Communicator >::ConstLocalVectorViewType,
   Expressions::Subtraction, Communicator >
operator-( const DistributedVectorView< Real1, Device, Index, Communicator >& a, const DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename DistributedVectorView< Real1, Device, Index, Communicator >::ConstLocalVectorViewType;
   using ConstView2 = typename DistributedVectorView< Real2, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Subtraction, Communicator >( a.getLocalVectorView(), b.getLocalVectorView(), b.getCommunicationGroup() );
}

////
// Multiplication
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
const Expressions::DistributedBinaryExpressionTemplate< 
   typename DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   ET, 
   Expressions::Multiplication, Communicator >
operator*( const DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   using ConstView = typename DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView, ET, Expressions::Multiplication, Communicator >( a.getLocalVectorView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
const Expressions::DistributedBinaryExpressionTemplate< 
   ET,
   typename DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Expressions::Multiplication, Communicator >
operator*( const ET& a, const DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   using ConstView = typename DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ET, ConstView, Expressions::Multiplication, Communicator >( a, b.getLocalVectorView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
const Expressions::DistributedBinaryExpressionTemplate< 
   typename DistributedVectorView< Real1, Device, Index, Communicator >::ConstLocalVectorViewType,
   typename DistributedVectorView< Real2, Device, Index, Communicator >::ConstLocalVectorViewType,
   Expressions::Multiplication, Communicator >
operator*( const DistributedVectorView< Real1, Device, Index, Communicator >& a, const DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename DistributedVectorView< Real1, Device, Index, Communicator >::ConstLocalVectorViewType;
   using ConstView2 = typename DistributedVectorView< Real2, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Multiplication, Communicator >( a.getLocalVectorView(), b.getLocalVectorView(), b.getCommunicationGroup() );
}

////
// Division
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
const Expressions::DistributedBinaryExpressionTemplate< 
   typename DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   ET, 
   Expressions::Division, Communicator >
operator/( const DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   using ConstView = typename DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView, ET, Expressions::Division, Communicator >( a.getLocalVectorView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
const Expressions::DistributedBinaryExpressionTemplate<
   ET,
   typename DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Expressions::Division, Communicator >
operator/( const ET& a, const DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   using ConstView = typename DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ET, ConstView, Expressions::Division, Communicator >( a, b.getLocalVectorView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
const Expressions::DistributedBinaryExpressionTemplate< 
   typename DistributedVectorView< Real1, Device, Index, Communicator >::ConstLocalVectorViewType,
   typename DistributedVectorView< Real2, Device, Index, Communicator >::ConstLocalVectorViewType,
   Expressions::Division, Communicator >
operator/( const DistributedVectorView< Real1, Device, Index, Communicator >& a, const DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename DistributedVectorView< Real1, Device, Index, Communicator >::ConstLocalVectorViewType;
   using ConstView2 = typename DistributedVectorView< Real2, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Division, Communicator >( a.getLocalVectorView(), b.getLocalVectorView(), b.getCommunicationGroup() );
}

////
// Comparison operations - operator ==
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
bool operator==( const DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   using Left = DistributedVectorView< Real, Device, Index, Communicator >;
   using Right = ET;
   return Expressions::DistributedComparison< Left, Right >::template EQ< Communicator >( a.getLocalVectorView(), b, a.getCommunicatorGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
bool operator==( const ET& a, const DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   using Left = ET;
   using Right = DistributedVectorView< Real, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template EQ< Communicator >( a, b.getLocalVectorView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index1, typename Index2, typename Communicator >
bool operator==( const DistributedVectorView< Real1, Device1, Index1, Communicator >& a, const DistributedVectorView< Real2, Device2, Index2, Communicator >& b )
{
   if( a.getCommunicationGroup() != b.getCommunicationGroup() )
      return false;
   const bool localResult =
         a.getLocalRange() == b.getLocalRange() &&
         a.getSize() == b.getSize() &&
         a.getLocalArrayView() == b.getLocalArrayView();
   bool result = true;
   if( a.getCommunicationGroup() != Communicator::NullGroup )
      Communicator::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicationGroup() );
   return result;
}

////
// Comparison operations - operator !=
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
bool operator!=( const DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   using Left = DistributedVectorView< Real, Device, Index, Communicator >;
   using Right = ET;
   return Expressions::DistributedComparison< Left, Right >::template NE< Communicator >( a.getLocalVectorView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
bool operator!=( const ET& a, const DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   using Left = ET;
   using Right = DistributedVectorView< Real, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template NE< Communicator >( a, b.getLocalVectorView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index, typename Communicator >
bool operator!=( const DistributedVectorView< Real1, Device1, Index >& a, const DistributedVectorView< Real2, Device2, Index >& b )
{
   return ! operator==( a, b );
}

////
// Comparison operations - operator <
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
bool operator<( const DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   using Left = DistributedVectorView< Real, Device, Index, Communicator >;
   using Right = ET;
   return Expressions::DistributedComparison< Left, Right >::template LT< Communicator >( a.getLocalVectorView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
bool operator<( const ET& a, const DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   using Left = ET;
   using Right = DistributedVectorView< Real, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template LT< Communicator >( a, b.getLocalVectorView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator<( const DistributedVectorView< Real1, Device, Index, Communicator >& a, const DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using Left = DistributedVectorView< Real1, Device, Index, Communicator >;
   using Right = DistributedVectorView< Real2, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template LT< Communicator >( a.getLocalVectorView(), b.getLocalVectorView(), b.getCommunicationGroup() );
}

////
// Comparison operations - operator <=
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
bool operator<=( const DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   using Left = DistributedVectorView< Real, Device, Index, Communicator >;
   using Right = ET;
   return Expressions::DistributedComparison< Left, Right >::template LE< Communicator >( a.getLocalVectorView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
bool operator<=( const ET& a, const DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   using Left = ET;
   using Right = DistributedVectorView< Real, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template LE< Communicator >( a, b.getLocalVectorView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator<=( const DistributedVectorView< Real1, Device, Index, Communicator >& a, const DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using Left = DistributedVectorView< Real1, Device, Index, Communicator >;
   using Right = DistributedVectorView< Real2, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template LE< Communicator >( a.getLocalVectorView(), b.getLocalVectorView(), b.getCommunicationGroup() );
}

////
// Comparison operations - operator >
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
bool operator>( const DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   using Left = DistributedVectorView< Real, Device, Index, Communicator >;
   using Right = ET;
   return Expressions::DistributedComparison< Left, Right >::template GT< Communicator >( a.getLocalVectorView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
bool operator>( const ET& a, const DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   using Left = ET;
   using Right = DistributedVectorView< Real, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template GT< Communicator >( a, b.getLocalVectorView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator>( const DistributedVectorView< Real1, Device, Index, Communicator >& a, const DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using Left = DistributedVectorView< Real1, Device, Index, Communicator >;
   using Right = DistributedVectorView< Real2, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template GT< Communicator >( a.getLocalVectorView(), b.getLocalVectorView(), b.getCommunicationGroup() );
}

////
// Comparison operations - operator >=
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
bool operator>=( const DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   using Left = DistributedVectorView< Real, Device, Index, Communicator >;
   using Right = ET;
   return Expressions::DistributedComparison< Left, Right >::template GE< Communicator >( a.getLocalVectorView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
bool operator>=( const ET& a, const DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   using Left = ET;
   using Right = DistributedVectorView< Real, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template GE< Communicator >( a, b.getLocalVectorView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator>=( const DistributedVectorView< Real1, Device, Index, Communicator >& a, const DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using Left = DistributedVectorView< Real1, Device, Index, Communicator >;
   using Right = DistributedVectorView< Real2, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template GE< Communicator >( a.getLocalVectorView(), b.getLocalVectorView(), b.getCommunicationGroup() );
}

////
// Minus
template< typename Real, typename Device, typename Index, typename Communicator >
const Expressions::DistributedUnaryExpressionTemplate< 
   typename DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Expressions::Minus, void, Communicator >
operator-( const DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Expressions::DistributedUnaryExpressionTemplate< ConstView, Expressions::Minus, void, Communicator >( a.getLocalVectorView() );
}

////
// Scalar product
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
auto
operator,( const DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
-> decltype( TNL::sum( a.getLocalVectorView() * b ) )
{
   using CommunicatorType = typename DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
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
operator,( const ET& a, const DistributedVectorView< Real, Device, Index, Communicator >& b ) 
-> decltype( TNL::sum( a * b.getLocalVectorView(), b.getCommunicationGroup() ) )
{
   using CommunicatorType = typename DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::sum( a * b.getLocalVectorView(), b.getCommunicationGroup() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto operator,( const DistributedVectorView< Real1, Device, Index, Communicator >& a, const DistributedVectorView< Real2, Device, Index, Communicator >& b )
->decltype( TNL::sum( a.getLocalVectorView() * b.getLocalVectorView(), b.getCommunicationGroup() ) )
{
   using CommunicatorType = typename DistributedVectorView< Real1, Device, Index, Communicator >::CommunicatorType;
   using Real = decltype( TNL::sum( a.getLocalVectorView() * b.getLocalVectorView(), b.getCommunicationGroup() ) );
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::sum( a.getLocalVectorView() * b.getLocalVectorView(), b.getCommunicationGroup() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

}//namespace Containers

////
// Functions are supposed to be in namespace TNL

////
// Min
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
const Containers::Expressions::DistributedBinaryExpressionTemplate< 
   typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   ET,
   Containers::Expressions::Min, Communicator >
min( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   using ConstView = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView, ET, Containers::Expressions::Min, Communicator >( a.getLocalVectorView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< 
   ET,
   typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Min, Communicator >
min( const ET& a, const Containers::DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   using ConstView = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, ConstView, Containers::Expressions::Min, Communicator >( a, b.getLocalVectorView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::DistributedVectorView< Real1, Device, Index, Communicator >::ConstLocalVectorViewType,
   typename Containers::DistributedVectorView< Real2, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Min, Communicator >
min( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename Containers::DistributedVectorView< Real1, Device, Index, Communicator >::ConstLocalVectorViewType;
   using ConstView2 = typename Containers::DistributedVectorView< Real2, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Min, Communicator >( a.getLocalVectorView(), b.getLocalVectorView(), b.getCommunicationGroup() );
}

////
// Max
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
const Containers::Expressions::DistributedBinaryExpressionTemplate< 
   typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   ET,
   Containers::Expressions::Max, Communicator >
max( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   using ConstView = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView, ET, Containers::Expressions::Max, Communicator >( a.getLocalVectorView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< 
   ET,
   typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Max, Communicator >
max( const ET& a, const Containers::DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   using ConstView = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, ConstView, Containers::Expressions::Max, Communicator >( a, b.getLocalVectorView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate< 
   typename Containers::DistributedVectorView< Real1, Device, Index, Communicator >::ConstLocalVectorViewType,
   typename Containers::DistributedVectorView< Real2, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Max, Communicator >
max( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename Containers::DistributedVectorView< Real1, Device, Index, Communicator >::ConstLocalVectorViewType;
   using ConstView2 = typename Containers::DistributedVectorView< Real2, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Max, Communicator >( a.getLocalVectorView(), b.getLocalVectorView(), b.getCommunicationGroup() );
}

////
// Abs
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType, 
   Containers::Expressions::Abs, void, Communicator >
abs( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Abs, void, Communicator >( a.getLocalVectorView(), a.getCommunicationGroup() );
}

////
// Sine
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType, 
   Containers::Expressions::Sin, void, Communicator >
sin( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Sin, void, Communicator >( a.getLocalVectorView() );
}

////
// Cosine
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType, 
   Containers::Expressions::Cos, void, Communicator >
cos( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Cos, void, Communicator >( a.getLocalVectorView() );
}

////
// Tangent
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType, 
   Containers::Expressions::Tan, void, Communicator >
tan( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Tan, void, Communicator >( a.getLocalVectorView() );
}

////
// Sqrt
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType, 
   Containers::Expressions::Sqrt, void, Communicator >
sqrt( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Sqrt, void, Communicator >( a.getLocalVectorView() );
}

////
// Cbrt
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType, 
   Containers::Expressions::Cbrt, void, Communicator >
cbrt( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Cbrt, void, Communicator >( a.getLocalVectorView() );
}

////
// Power
template< typename Real, typename Device, typename Index, typename Communicator, typename ExpType >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Pow, ExpType, Communicator >
pow( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ExpType& exp )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Pow, ExpType, Communicator >( a.getLocalVectorView(), exp );
}

////
// Floor
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Floor, void, Communicator >
floor( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Floor, void, Communicator >( a.getLocalVectorView() );
}

////
// Ceil
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType, 
   Containers::Expressions::Ceil, void, Communicator >
ceil( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Ceil, void, Communicator >( a.getLocalVectorView() );
}

////
// Acos
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType, 
   Containers::Expressions::Acos, void, Communicator >
acos( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Acos, void, Communicator >( a.getLocalVectorView() );
}

////
// Asin
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType, 
   Containers::Expressions::Asin, void, Communicator >
asin( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Asin, void, Communicator >( a.getLocalVectorView() );
}

////
// Atan
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType, 
   Containers::Expressions::Atan, void, Communicator >
atan( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Atan, void, Communicator >( a.getLocalVectorView() );
}

////
// Cosh
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType, 
   Containers::Expressions::Cosh, void, Communicator >
cosh( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Cosh, void, Communicator >( a.getLocalVectorView() );
}

////
// Tanh
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Tanh, void, Communicator >
tanh( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Tanh, void, Communicator >( a.getLocalVectorView() );
}

////
// Log
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Log, void, Communicator >
log( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Log, void, Communicator >( a.getLocalVectorView() );
}

////
// Log10
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Log10, void, Communicator >
log10( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Log10, void, Communicator >( a.getLocalVectorView() );
}

////
// Log2
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Log2, void, Communicator >
log2( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Log2, void, Communicator >( a.getLocalVectorView() );
}

////
// Exp
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Exp, void, Communicator >
exp( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Exp, void, Communicator >( a.getLocalVectorView() );
}

////
// Sign
template< typename Real, typename Device, typename Index, typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate< 
   typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType,
   Containers::Expressions::Sign, void, Communicator >
sign( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalVectorViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Sign, void, Communicator >( a.getLocalVectorView() );
}

////
// Vertical operations - min
template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
min( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
-> decltype( Containers::Expressions::ExpressionMin( a.getLocalVectorView() ) )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
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
argMin( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, Index& arg )
-> decltype( Containers::Expressions::ExpressionArgMin( a.getLocalVectorView(), arg ) )
{
   throw Exceptions::NotImplementedError( "agrMin for distributed vector view is not implemented yet." );
   return Containers::Expressions::ExpressionArgMin( a.getLocalVectorView(), arg );
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
max( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
-> decltype( Containers::Expressions::ExpressionMax( a.getLocalVectorView() ) )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
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
argMax( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, Index& arg )
-> decltype( Containers::Expressions::ExpressionArgMax( a.getLocalVectorView(), arg ) )
{
   throw Exceptions::NotImplementedError( "agrMax for distributed vector view is not implemented yet." );
   return Containers::Expressions::ExpressionArgMax( a.getLocalVectorView(), arg );
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
sum( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
-> decltype( Containers::Expressions::ExpressionSum( a.getLocalVectorView() ) )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
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
lpNorm( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const Real2& p )
-> decltype( TNL::pow( Containers::Expressions::ExpressionLpNorm( a.getLocalVectorView(), p ), p ) )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = ( Real ) 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionLpNorm( a.getLocalVectorView(), p );
      CommunicatorType::template Allreduce< Real >( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return TNL::pow( result, 1.0 / p );
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
product( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
-> decltype( Containers::Expressions::ExpressionProduct( a.getLocalVectorView() ) )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
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
logicalOr( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
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
typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::RealType
binaryOr( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
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
logicalAnd( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
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
typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::RealType
binaryAnd( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVectorView< Real, Device, Index, Communicator >::CommunicatorType;
   bool result = true;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionBinaryAnd( a.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_BAND, a.getCommunicationGroup() );
   }
   return result;
}

////
// Dot product - the same as scalar product, just for convenience
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
auto
dot( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
-> decltype( ( a, b ) )
{
   return ( a, b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
auto
dot( const ET& a, const Containers::DistributedVectorView< Real, Device, Index, Communicator >& b )
-> decltype( ( a, b ) )
{
   return ( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
dot( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
-> decltype( ( a, b ) )
{
   return ( a, b );
}

////
// TODO: Replace this with multiplication when its safe
template< typename Real, typename Device, typename Index, typename Communicator, typename ET >
Containers::DistributedVectorView< Real, Device, Index, Communicator >
Scale( const Containers::DistributedVectorView< Real, Device, Index, Communicator >& a, const ET& b )
{
   Containers::DistributedVectorView< Real, Device, Index, Communicator > result = Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real, Device, Index, Communicator >, ET, Containers::Expressions::Multiplication, Communicator >( a, b );
   return result;
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator >
Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Multiplication, Communicator >
Scale( const ET& a, const Containers::DistributedVectorView< Real, Device, Index, Communicator >& b )
{
   Containers::DistributedVectorView< Real, Device, Index, Communicator > result =  Containers::Expressions::DistributedBinaryExpressionTemplate< ET, Containers::DistributedVectorView< Real, Device, Index, Communicator >, Containers::Expressions::Multiplication, Communicator >( a, b );
   return result;
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index, Communicator >, Containers::DistributedVectorView< Real2, Device, Index, Communicator >, Containers::Expressions::Multiplication, Communicator >
Scale( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   Containers::DistributedVectorView< Real1, Device, Index, Communicator > result =  Containers::Expressions::DistributedBinaryExpressionTemplate< Containers::DistributedVectorView< Real1, Device, Index, Communicator >, Containers::DistributedVectorView< Real2, Device, Index, Communicator >, Containers::Expressions::Multiplication, Communicator >( a, b );
   return result;
}

} // namespace TNL
