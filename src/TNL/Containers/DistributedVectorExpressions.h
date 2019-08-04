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
#include <TNL/Containers/Expressions/DistributedComparison.h>
#include <TNL/Containers/Expressions/VerticalOperations.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Containers {

////
// Addition
template< typename Real, typename Device, typename Index, typename Communicator, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
auto
operator+( const DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   using ConstView = typename DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView, ET, Expressions::Addition, Communicator >( a.getConstLocalView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
auto
operator+( const ET& a, const DistributedVector< Real, Device, Index, Communicator >& b )
{
   using ConstView = typename DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ET, ConstView, Expressions::Addition, Communicator >( a, b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
operator+( const DistributedVector< Real1, Device, Index, Communicator >& a, const DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename DistributedVector< Real1, Device, Index, Communicator >::ConstLocalViewType;
   using ConstView2 = typename DistributedVector< Real2, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Addition, Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
operator+( const DistributedVectorView< Real1, Device, Index, Communicator >& a, const DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename DistributedVector< Real1, Device, Index, Communicator >::ConstLocalViewType;
   using ConstView2 = typename DistributedVector< Real2, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Addition, Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
operator+( const DistributedVector< Real1, Device, Index, Communicator >& a, const DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename DistributedVector< Real1, Device, Index, Communicator >::ConstLocalViewType;
   using ConstView2 = typename DistributedVector< Real2, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Addition, Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

////
// Subtraction
template< typename Real, typename Device, typename Index, typename Communicator, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
auto
operator-( const DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   using ConstView = typename DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView, ET, Expressions::Subtraction, Communicator >( a.getConstLocalView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
auto
operator-( const ET& a, const DistributedVector< Real, Device, Index, Communicator >& b )
{
   using ConstView = typename DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ET, ConstView, Expressions::Subtraction, Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
operator-( const DistributedVector< Real1, Device, Index, Communicator >& a, const DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename DistributedVector< Real1, Device, Index, Communicator >::ConstLocalViewType;
   using ConstView2 = typename DistributedVector< Real2, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Subtraction, Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
operator-( const DistributedVectorView< Real1, Device, Index, Communicator >& a, const DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename DistributedVector< Real1, Device, Index, Communicator >::ConstLocalViewType;
   using ConstView2 = typename DistributedVector< Real2, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Subtraction, Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
operator-( const DistributedVector< Real1, Device, Index, Communicator >& a, const DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename DistributedVector< Real1, Device, Index, Communicator >::ConstLocalViewType;
   using ConstView2 = typename DistributedVector< Real2, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Subtraction, Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

////
// Multiplication
template< typename Real, typename Device, typename Index, typename Communicator, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
auto
operator*( const DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   using ConstView = typename DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView, ET, Expressions::Multiplication, Communicator >( a.getConstLocalView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
auto
operator*( const ET& a, const DistributedVector< Real, Device, Index, Communicator >& b )
{
   using ConstView = typename DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ET, ConstView, Expressions::Multiplication, Communicator >( a, b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
operator*( const DistributedVector< Real1, Device, Index, Communicator >& a, const DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename DistributedVector< Real1, Device, Index, Communicator >::ConstLocalViewType;
   using ConstView2 = typename DistributedVector< Real2, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Multiplication, Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
operator*( const DistributedVectorView< Real1, Device, Index, Communicator >& a, const DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename DistributedVector< Real1, Device, Index, Communicator >::ConstLocalViewType;
   using ConstView2 = typename DistributedVector< Real2, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Multiplication, Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
operator*( const DistributedVector< Real1, Device, Index, Communicator >& a, const DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename DistributedVector< Real1, Device, Index, Communicator >::ConstLocalViewType;
   using ConstView2 = typename DistributedVector< Real2, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Multiplication, Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

////
// Division
template< typename Real, typename Device, typename Index, typename Communicator, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
auto
operator/( const DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   using ConstView = typename DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView, ET, Expressions::Division, Communicator >( a.getConstLocalView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
auto
operator/( const ET& a, const DistributedVector< Real, Device, Index, Communicator >& b )
{
   using ConstView = typename DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ET, ConstView, Expressions::Division, Communicator >( a, b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
operator/( const DistributedVector< Real1, Device, Index, Communicator >& a, const DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename DistributedVector< Real1, Device, Index, Communicator >::ConstLocalViewType;
   using ConstView2 = typename DistributedVector< Real2, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Division, Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
operator/( const DistributedVectorView< Real1, Device, Index, Communicator >& a, const DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename DistributedVector< Real1, Device, Index, Communicator >::ConstLocalViewType;
   using ConstView2 = typename DistributedVector< Real2, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Division, Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
operator/( const DistributedVector< Real1, Device, Index, Communicator >& a, const DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename DistributedVector< Real1, Device, Index, Communicator >::ConstLocalViewType;
   using ConstView2 = typename DistributedVector< Real2, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Expressions::Division, Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

////
// Comparison operations - operator ==
template< typename Real, typename Device, typename Index, typename Communicator, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
bool operator==( const DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   using Left = DistributedVectorView< Real, Device, Index, Communicator >;
   using Right = ET;
   return Expressions::DistributedComparison< Left, Right >::template EQ< Communicator >( a.getConstLocalView(), b, a.getCommunicatorGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
bool operator==( const ET& a, const DistributedVector< Real, Device, Index, Communicator >& b )
{
   using Left = ET;
   using Right = DistributedVectorView< Real, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template EQ< Communicator >( a, b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index1, typename Index2, typename Communicator >
bool operator==( const DistributedVector< Real1, Device1, Index1, Communicator >& a, const DistributedVector< Real2, Device2, Index2, Communicator >& b )
{
   if( a.getCommunicationGroup() != b.getCommunicationGroup() )
      return false;
   const bool localResult =
         a.getLocalRange() == b.getLocalRange() &&
         a.getSize() == b.getSize() &&
         a.getConstLocalView() == b.getConstLocalView();
   bool result = true;
   if( a.getCommunicationGroup() != Communicator::NullGroup )
      Communicator::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicationGroup() );
   return result;
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index1, typename Index2, typename Communicator >
bool operator==( const DistributedVectorView< Real1, Device1, Index1, Communicator >& a, const DistributedVector< Real2, Device2, Index2, Communicator >& b )
{
   if( a.getCommunicationGroup() != b.getCommunicationGroup() )
      return false;
   const bool localResult =
         a.getLocalRange() == b.getLocalRange() &&
         a.getSize() == b.getSize() &&
         a.getConstLocalView() == b.getConstLocalView();
   bool result = true;
   if( a.getCommunicationGroup() != Communicator::NullGroup )
      Communicator::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicationGroup() );
   return result;
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index1, typename Index2, typename Communicator >
bool operator==( const DistributedVector< Real1, Device1, Index1, Communicator >& a, const DistributedVectorView< Real2, Device2, Index2, Communicator >& b )
{
   if( a.getCommunicationGroup() != b.getCommunicationGroup() )
      return false;
   const bool localResult =
         a.getLocalRange() == b.getLocalRange() &&
         a.getSize() == b.getSize() &&
         a.getConstLocalView() == b.getConstLocalView();
   bool result = true;
   if( a.getCommunicationGroup() != Communicator::NullGroup )
      Communicator::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicationGroup() );
   return result;
}

////
// Comparison operations - operator !=
template< typename Real, typename Device, typename Index, typename Communicator, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
bool operator!=( const DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   using Left = DistributedVectorView< Real, Device, Index, Communicator >;
   using Right = ET;
   return Expressions::DistributedComparison< Left, Right >::template NE< Communicator >( a.getConstLocalView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
bool operator!=( const ET& a, const DistributedVector< Real, Device, Index, Communicator >& b )
{
   using Left = ET;
   using Right = DistributedVectorView< Real, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template NE< Communicator >( a, b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index, typename Communicator >
bool operator!=( const DistributedVector< Real1, Device1, Index >& a, const DistributedVector< Real2, Device2, Index >& b )
{
   return ! operator==( a, b );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index, typename Communicator >
bool operator!=( const DistributedVectorView< Real1, Device1, Index >& a, const DistributedVector< Real2, Device2, Index >& b )
{
   return ! operator==( a, b );
}

template< typename Real1, typename Real2, typename Device1, typename Device2, typename Index, typename Communicator >
bool operator!=( const DistributedVector< Real1, Device1, Index >& a, const DistributedVectorView< Real2, Device2, Index >& b )
{
   return ! operator==( a, b );
}

////
// Comparison operations - operator <
template< typename Real, typename Device, typename Index, typename Communicator, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
bool operator<( const DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   using Left = DistributedVectorView< Real, Device, Index, Communicator >;
   using Right = ET;
   return Expressions::DistributedComparison< Left, Right >::template LT< Communicator >( a.getConstLocalView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
bool operator<( const ET& a, const DistributedVector< Real, Device, Index, Communicator >& b )
{
   using Left = ET;
   using Right = DistributedVectorView< Real, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template LT< Communicator >( a, b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator<( const DistributedVector< Real1, Device, Index, Communicator >& a, const DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using Left = DistributedVectorView< Real1, Device, Index, Communicator >;
   using Right = DistributedVectorView< Real2, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template LT< Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator<( const DistributedVectorView< Real1, Device, Index, Communicator >& a, const DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using Left = DistributedVectorView< Real1, Device, Index, Communicator >;
   using Right = DistributedVectorView< Real2, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template LT< Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator<( const DistributedVector< Real1, Device, Index, Communicator >& a, const DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using Left = DistributedVectorView< Real1, Device, Index, Communicator >;
   using Right = DistributedVectorView< Real2, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template LT< Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

////
// Comparison operations - operator <=
template< typename Real, typename Device, typename Index, typename Communicator, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
bool operator<=( const DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   using Left = DistributedVectorView< Real, Device, Index, Communicator >;
   using Right = ET;
   return Expressions::DistributedComparison< Left, Right >::template LE< Communicator >( a.getConstLocalView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
bool operator<=( const ET& a, const DistributedVector< Real, Device, Index, Communicator >& b )
{
   using Left = ET;
   using Right = DistributedVectorView< Real, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template LE< Communicator >( a, b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator<=( const DistributedVector< Real1, Device, Index, Communicator >& a, const DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using Left = DistributedVectorView< Real1, Device, Index, Communicator >;
   using Right = DistributedVectorView< Real2, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template LE< Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator<=( const DistributedVectorView< Real1, Device, Index, Communicator >& a, const DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using Left = DistributedVectorView< Real1, Device, Index, Communicator >;
   using Right = DistributedVectorView< Real2, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template LE< Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator<=( const DistributedVector< Real1, Device, Index, Communicator >& a, const DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using Left = DistributedVectorView< Real1, Device, Index, Communicator >;
   using Right = DistributedVectorView< Real2, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template LE< Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

////
// Comparison operations - operator >
template< typename Real, typename Device, typename Index, typename Communicator, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
bool operator>( const DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   using Left = DistributedVectorView< Real, Device, Index, Communicator >;
   using Right = ET;
   return Expressions::DistributedComparison< Left, Right >::template GT< Communicator >( a.getConstLocalView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
bool operator>( const ET& a, const DistributedVector< Real, Device, Index, Communicator >& b )
{
   using Left = ET;
   using Right = DistributedVectorView< Real, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template GT< Communicator >( a, b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator>( const DistributedVector< Real1, Device, Index, Communicator >& a, const DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using Left = DistributedVectorView< Real1, Device, Index, Communicator >;
   using Right = DistributedVectorView< Real2, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template GT< Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator>( const DistributedVectorView< Real1, Device, Index, Communicator >& a, const DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using Left = DistributedVectorView< Real1, Device, Index, Communicator >;
   using Right = DistributedVectorView< Real2, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template GT< Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator>( const DistributedVector< Real1, Device, Index, Communicator >& a, const DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using Left = DistributedVectorView< Real1, Device, Index, Communicator >;
   using Right = DistributedVectorView< Real2, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template GT< Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

////
// Comparison operations - operator >=
template< typename Real, typename Device, typename Index, typename Communicator, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
bool operator>=( const DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   using Left = DistributedVectorView< Real, Device, Index, Communicator >;
   using Right = ET;
   return Expressions::DistributedComparison< Left, Right >::template GE< Communicator >( a.getConstLocalView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
bool operator>=( const ET& a, const DistributedVector< Real, Device, Index, Communicator >& b )
{
   using Left = ET;
   using Right = DistributedVectorView< Real, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template GE< Communicator >( a, b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator>=( const DistributedVector< Real1, Device, Index, Communicator >& a, const DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using Left = DistributedVectorView< Real1, Device, Index, Communicator >;
   using Right = DistributedVectorView< Real2, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template GE< Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator>=( const DistributedVectorView< Real1, Device, Index, Communicator >& a, const DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using Left = DistributedVectorView< Real1, Device, Index, Communicator >;
   using Right = DistributedVectorView< Real2, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template GE< Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
bool operator>=( const DistributedVector< Real1, Device, Index, Communicator >& a, const DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using Left = DistributedVectorView< Real1, Device, Index, Communicator >;
   using Right = DistributedVectorView< Real2, Device, Index, Communicator >;
   return Expressions::DistributedComparison< Left, Right >::template GE< Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

////
// Minus
template< typename Real, typename Device, typename Index, typename Communicator >
auto
operator-( const DistributedVector< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Expressions::DistributedUnaryExpressionTemplate< ConstView, Expressions::Minus, void, Communicator >( a.getConstLocalView(), a.getCommunicationGroup() );
}

////
// Scalar product
template< typename Real, typename Device, typename Index, typename Communicator, typename ET,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
auto
operator,( const DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   using CommunicatorType = typename DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::sum( a.getConstLocalView() * b );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator,
          typename..., typename = std::enable_if_t< Expressions::IsNumericExpression<ET>::value > >
auto
operator,( const ET& a, const DistributedVector< Real, Device, Index, Communicator >& b )
{
   using CommunicatorType = typename DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::sum( a * b.getConstLocalView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
operator,( const DistributedVector< Real1, Device, Index, Communicator >& a, const DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using CommunicatorType = typename DistributedVector< Real1, Device, Index, Communicator >::CommunicatorType;
   using Real = decltype( TNL::sum( a.getConstLocalView() * b.getConstLocalView() ) );
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::sum( a.getConstLocalView() * b.getConstLocalView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
operator,( const DistributedVectorView< Real1, Device, Index, Communicator >& a, const DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using CommunicatorType = typename DistributedVector< Real1, Device, Index, Communicator >::CommunicatorType;
   using Real = decltype( TNL::sum( a.getConstLocalView() * b.getConstLocalView() ) );
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::sum( a.getConstLocalView() * b.getConstLocalView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
operator,( const DistributedVector< Real1, Device, Index, Communicator >& a, const DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using CommunicatorType = typename DistributedVector< Real1, Device, Index, Communicator >::CommunicatorType;
   using Real = decltype( TNL::sum( a.getConstLocalView() * b.getConstLocalView() ) );
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::sum( a.getConstLocalView() * b.getConstLocalView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

} // namespace Containers

/////
// Functions are supposed to be in namespace TNL

////
// Min
template< typename Real, typename Device, typename Index, typename Communicator, typename ET,
          typename..., typename = std::enable_if_t< Containers::Expressions::IsNumericExpression<ET>::value > >
auto
min( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView, ET, Containers::Expressions::Min, Communicator >( a.getConstLocalView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator,
          typename..., typename = std::enable_if_t< Containers::Expressions::IsNumericExpression<ET>::value > >
auto
min( const ET& a, const Containers::DistributedVector< Real, Device, Index, Communicator >& b )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, ConstView, Containers::Expressions::Min, Communicator >( a, b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
min( const Containers::DistributedVector< Real1, Device, Index, Communicator >& a, const Containers::DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename Containers::DistributedVector< Real1, Device, Index, Communicator >::ConstLocalViewType;
   using ConstView2 = typename Containers::DistributedVector< Real2, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Min, Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
min( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename Containers::DistributedVector< Real1, Device, Index, Communicator >::ConstLocalViewType;
   using ConstView2 = typename Containers::DistributedVector< Real2, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Min, Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
min( const Containers::DistributedVector< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename Containers::DistributedVector< Real1, Device, Index, Communicator >::ConstLocalViewType;
   using ConstView2 = typename Containers::DistributedVector< Real2, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Min, Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

////
// Max
template< typename Real, typename Device, typename Index, typename Communicator, typename ET,
          typename..., typename = std::enable_if_t< Containers::Expressions::IsNumericExpression<ET>::value > >
auto
max( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView, ET, Containers::Expressions::Max, Communicator >( a.getConstLocalView(), b, a.getCommunicationGroup() );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator,
          typename..., typename = std::enable_if_t< Containers::Expressions::IsNumericExpression<ET>::value > >
auto
max( const ET& a, const Containers::DistributedVector< Real, Device, Index, Communicator >& b )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ET, ConstView, Containers::Expressions::Max, Communicator >( a, b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
max( const Containers::DistributedVector< Real1, Device, Index, Communicator >& a, const Containers::DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename Containers::DistributedVector< Real1, Device, Index, Communicator >::ConstLocalViewType;
   using ConstView2 = typename Containers::DistributedVector< Real2, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Max, Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
max( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVector< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename Containers::DistributedVector< Real1, Device, Index, Communicator >::ConstLocalViewType;
   using ConstView2 = typename Containers::DistributedVector< Real2, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Max, Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
max( const Containers::DistributedVector< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   using ConstView1 = typename Containers::DistributedVector< Real1, Device, Index, Communicator >::ConstLocalViewType;
   using ConstView2 = typename Containers::DistributedVector< Real2, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedBinaryExpressionTemplate< ConstView1, ConstView2, Containers::Expressions::Max, Communicator >( a.getConstLocalView(), b.getConstLocalView(), b.getCommunicationGroup() );
}

////
// Dot product - the same as scalar product, just for convenience
template< typename Real, typename Device, typename Index, typename Communicator, typename ET,
          typename..., typename = std::enable_if_t< Containers::Expressions::IsNumericExpression<ET>::value > >
auto
dot( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ET& b )
{
   return ( a, b );
}

template< typename ET, typename Real, typename Device, typename Index, typename Communicator,
          typename..., typename = std::enable_if_t< Containers::Expressions::IsNumericExpression<ET>::value > >
auto
dot( const ET& a, const Containers::DistributedVector< Real, Device, Index, Communicator >& b )
{
   return ( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
dot( const Containers::DistributedVector< Real1, Device, Index, Communicator >& a, const Containers::DistributedVector< Real2, Device, Index, Communicator >& b )
{
   return ( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
dot( const Containers::DistributedVectorView< Real1, Device, Index, Communicator >& a, const Containers::DistributedVector< Real2, Device, Index, Communicator >& b )
{
   return ( a, b );
}

template< typename Real1, typename Real2, typename Device, typename Index, typename Communicator >
auto
dot( const Containers::DistributedVector< Real1, Device, Index, Communicator >& a, const Containers::DistributedVectorView< Real2, Device, Index, Communicator >& b )
{
   return ( a, b );
}


////
// Abs
template< typename Real, typename Device, typename Index, typename Communicator >
auto
abs( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Abs, void, Communicator >( a.getConstLocalView(), a.getCommunicationGroup() );
}

////
// Sine
template< typename Real, typename Device, typename Index, typename Communicator >
auto
sin( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Sin, void, Communicator >( a.getConstLocalView(), a.getCommunicationGroup() );
}

////
// Cosine
template< typename Real, typename Device, typename Index, typename Communicator >
auto
cos( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Cos, void, Communicator >( a.getConstLocalView(), a.getCommunicationGroup() );
}

////
// Tangent
template< typename Real, typename Device, typename Index, typename Communicator >
auto
tan( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Tan, void, Communicator >( a.getConstLocalView(), a.getCommunicationGroup() );
}

////
// Sqrt
template< typename Real, typename Device, typename Index, typename Communicator >
auto
sqrt( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Sqrt, void, Communicator >( a.getConstLocalView(), a.getCommunicationGroup() );
}

////
// Cbrt
template< typename Real, typename Device, typename Index, typename Communicator >
auto
cbrt( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Cbrt, void, Communicator >( a.getConstLocalView(), a.getCommunicationGroup() );
}

////
// Power
template< typename Real, typename Device, typename Index, typename Communicator, typename ExpType >
auto
pow( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, const ExpType& exp )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Pow, ExpType, Communicator >( a.getConstLocalView(), exp, a.getCommunicationGroup() );
}

////
// Floor
template< typename Real, typename Device, typename Index, typename Communicator >
auto
floor( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Floor, void, Communicator >( a.getConstLocalView(), a.getCommunicationGroup() );
}

////
// Ceil
template< typename Real, typename Device, typename Index, typename Communicator >
auto
ceil( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Ceil, void, Communicator >( a.getConstLocalView(), a.getCommunicationGroup() );
}

////
// Acos
template< typename Real, typename Device, typename Index, typename Communicator >
auto
acos( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Acos, void, Communicator >( a.getConstLocalView(), a.getCommunicationGroup() );
}

////
// Asin
template< typename Real, typename Device, typename Index, typename Communicator >
auto
asin( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Asin, void, Communicator >( a.getConstLocalView(), a.getCommunicationGroup() );
}

////
// Atan
template< typename Real, typename Device, typename Index, typename Communicator >
auto
atan( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Atan, void, Communicator >( a.getConstLocalView(), a.getCommunicationGroup() );
}

////
// Cosh
template< typename Real, typename Device, typename Index, typename Communicator >
auto
cosh( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Cosh, void, Communicator >( a.getConstLocalView(), a.getCommunicationGroup() );
}

////
// Tanh
template< typename Real, typename Device, typename Index, typename Communicator >
auto
tanh( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Tanh, void, Communicator >( a.getConstLocalView(), a.getCommunicationGroup() );
}

////
// Log
template< typename Real, typename Device, typename Index, typename Communicator >
auto
log( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Log, void, Communicator >( a.getConstLocalView(), a.getCommunicationGroup() );
}

////
// Log10
template< typename Real, typename Device, typename Index, typename Communicator >
auto
log10( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Log10, void, Communicator >( a.getConstLocalView(), a.getCommunicationGroup() );
}

////
// Log2
template< typename Real, typename Device, typename Index, typename Communicator >
auto
log2( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Log2, void, Communicator >( a.getConstLocalView(), a.getCommunicationGroup() );
}

////
// Exp
template< typename Real, typename Device, typename Index, typename Communicator >
auto
exp( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Exp, void, Communicator >( a.getConstLocalView(), a.getCommunicationGroup() );
}

////
// Sign
template< typename Real, typename Device, typename Index, typename Communicator >
auto
sign( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using ConstView = typename Containers::DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType;
   return Containers::Expressions::DistributedUnaryExpressionTemplate< ConstView, Containers::Expressions::Sign, void, Communicator >( a.getConstLocalView(), a.getCommunicationGroup() );
}

////
// Vertical operations - min
template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
min( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = std::numeric_limits< Real >::max();
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionMin( a.getConstLocalView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MIN, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
argMin( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, Index& arg )
{
   throw Exceptions::NotImplementedError( "agrMin for distributed vector view is not implemented yet." );
   return Containers::Expressions::ExpressionArgMin( a.getConstLocalView(), arg );
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
max( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = std::numeric_limits< Real >::min();
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionMax( a.getConstLocalView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MAX, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
argMax( const Containers::DistributedVector< Real, Device, Index, Communicator >& a, Index& arg )
{
   throw Exceptions::NotImplementedError( "agrMax for distributed vector view is not implemented yet." );
   return Containers::Expressions::ExpressionArgMax( a.getConstLocalView(), arg );
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
sum( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionSum( a.getConstLocalView() );
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
-> decltype( TNL::pow( Containers::Expressions::ExpressionLpNorm( a.getConstLocalView(), p ), p ) )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = ( Real ) 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionLpNorm( a.getConstLocalView(), p );
      CommunicatorType::template Allreduce< Real >( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return TNL::pow( result, 1.0 / p );
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
product( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = ( Real ) 1.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionProduct( a.getConstLocalView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_PROD, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
logicalOr( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   bool result = false;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionLogicalOr( a.getConstLocalView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LOR, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
binaryOr( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = ( Real ) 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionBinaryOr( a.getConstLocalView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_BOR, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
logicalAnd( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   Real result = std::numeric_limits< Real >::max();
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionLogicalAnd( a.getConstLocalView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicationGroup() );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index, typename Communicator >
auto
binaryAnd( const Containers::DistributedVector< Real, Device, Index, Communicator >& a )
{
   using CommunicatorType = typename Containers::DistributedVector< Real, Device, Index, Communicator >::CommunicatorType;
   bool result = true;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionBinaryAnd( a.getConstLocalView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_BAND, a.getCommunicationGroup() );
   }
   return result;
}

} // namespace TNL
