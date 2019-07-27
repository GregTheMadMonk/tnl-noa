/***************************************************************************
                          DistributedComparison.h  -  description
                             -------------------
    begin                : Jul 4, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>
#include <TNL/Containers/Algorithms/Reduction.h>

namespace TNL {
   namespace Containers {
      namespace Expressions {

////
// Non-static comparison
template< typename T1,
          typename T2,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value,
          ExpressionVariableType T2Type = ExpressionVariableTypeGetter< T2 >::value >
struct DistributedComparison
{
};

/////
// Distributed comparison of two vector expressions
template< typename T1,
          typename T2 >
struct DistributedComparison< T1, T2, VectorExpressionVariable, VectorExpressionVariable >
{
   template< typename Communicator >
   static bool EQ( const T1& a, const T2& b, const typename Communicator::CommunicationGroup& communicationGroup )
   {
      if( a.getSize() != b.getSize() )
         return false;
      if( a.getSize() == 0 )
         return true;

      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a[ i ] == b[ i ] ); };
      auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
      auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
      bool localResult = Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );

      bool result = localResult;
      if( communicationGroup != Communicator::NullGroup ) {
         Communicator::Allreduce( &localResult, &result, 1, MPI_LAND, communicationGroup );
      }
      return result;
   }

   template< typename Communicator >
   static bool NE( const T1& a, const T2& b, const typename Communicator::CommunicationGroup& communicationGroup )
   {
      return ! EQ< Communicator >( a, b, communicationGroup );
   }

   template< typename Communicator >
   static bool GT( const T1& a, const T2& b, const typename Communicator::CommunicationGroup& communicationGroup )
   {
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );

      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a[ i ] > b[ i ] ); };
      auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
      auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
      bool localResult = Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );

      bool result = localResult;
      if( communicationGroup != Communicator::NullGroup ) {
         Communicator::Allreduce( &localResult, &result, 1, MPI_LAND, communicationGroup );
      }
      return result;
   }

   template< typename Communicator >
   static bool LE( const T1& a, const T2& b, const typename Communicator::CommunicationGroup& communicationGroup )
   {
      return ! GT( a, b, communicationGroup );
   }

   template< typename Communicator >
   static bool LT( const T1& a, const T2& b, const typename Communicator::CommunicationGroup& communicationGroup )
   {
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );

      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a[ i ] < b[ i ] ); };
      auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
      auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
      bool localResult = Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );

      bool result = localResult;
      if( communicationGroup != Communicator::NullGroup ) {
         Communicator::Allreduce( &localResult, &result, 1, MPI_LAND, communicationGroup );
      }
      return result;
   }

   template< typename Communicator >
   static bool GE( const T1& a, const T2& b, const typename Communicator::CommunicationGroup& communicationGroup )
   {
      return ! LT( a, b, communicationGroup );
   }
};

/////
// Distributed comparison of number and vector expression
template< typename T1,
          typename T2 >
struct DistributedComparison< T1, T2, ArithmeticVariable, VectorExpressionVariable >
{

   template< typename Communicator >
   static bool EQ( const T1& a, const T2& b, const typename Communicator::CommunicationGroup& communicationGroup )
   {
      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a == b[ i ] ); };
      auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
      auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
      bool localResult = Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );

      bool result = localResult;
      if( communicationGroup != Communicator::NullGroup ) {
         Communicator::Allreduce( &localResult, &result, 1, MPI_LAND, communicationGroup );
      }
      return result;
   }

   template< typename Communicator >
   static bool NE( const T1& a, const T2& b, const typename Communicator::CommunicationGroup& communicationGroup )
   {
      return ! EQ( a, b, communicationGroup );
   }

   template< typename Communicator >
   static bool GT( const T1& a, const T2& b, const typename Communicator::CommunicationGroup& communicationGroup )
   {
      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a > b[ i ] ); };
      auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
      auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
      bool localResult = Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );

      bool result = localResult;
      if( communicationGroup != Communicator::NullGroup ) {
         Communicator::Allreduce( &localResult, &result, 1, MPI_LAND, communicationGroup );
      }
      return result;
   }

   template< typename Communicator >
   static bool LE( const T1& a, const T2& b, const typename Communicator::CommunicationGroup& communicationGroup )
   {
      return ! GT( a, b, communicationGroup );
   }

   template< typename Communicator >
   static bool LT( const T1& a, const T2& b, const typename Communicator::CommunicationGroup& communicationGroup )
   {
      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a < b[ i ] ); };
      auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
      auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
      bool localResult = Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );

      bool result = localResult;
      if( communicationGroup != Communicator::NullGroup ) {
         Communicator::Allreduce( &localResult, &result, 1, MPI_LAND, communicationGroup );
      }
      return result;
   }

   template< typename Communicator >
   static bool GE( const T1& a, const T2& b, const typename Communicator::CommunicationGroup& communicationGroup )
   {
      return ! LT( a, b, communicationGroup );
   }
};

/////
// Distributed comparison of vector expressions and number
template< typename T1,
          typename T2 >
struct DistributedComparison< T1, T2, VectorExpressionVariable, ArithmeticVariable >
{

   template< typename Communicator >
   static bool EQ( const T1& a, const T2& b, const typename Communicator::CommunicationGroup& communicationGroup )
   {
      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a[ i ] == b ); };
      auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
      auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
      bool localResult = Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );

      bool result = localResult;
      if( communicationGroup != Communicator::NullGroup ) {
         Communicator::Allreduce( &localResult, &result, 1, MPI_LAND, communicationGroup );
      }
      return result;
   }

   template< typename Communicator >
   static bool NE( const T1& a, const T2& b, const typename Communicator::CommunicationGroup& communicationGroup )
   {
      return ! EQ( a, b, communicationGroup );
   }

   template< typename Communicator >
   static bool GT( const T1& a, const T2& b, const typename Communicator::CommunicationGroup& communicationGroup )
   {
      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a[ i ] > b ); };
      auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
      auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
      bool localResult = Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );

      bool result = localResult;
      if( communicationGroup != Communicator::NullGroup ) {
         Communicator::Allreduce( &localResult, &result, 1, MPI_LAND, communicationGroup );
      }
      return result;
   }

   template< typename Communicator >
   static bool LE( const T1& a, const T2& b, const typename Communicator::CommunicationGroup& communicationGroup )
   {
      return ! GT( a, b, communicationGroup );
   }

   template< typename Communicator >
   static bool LT( const T1& a, const T2& b, const typename Communicator::CommunicationGroup& communicationGroup )
   {
      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a[ i ] < b ); };
      auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
      auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
      bool localResult = Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );

      bool result = localResult;
      if( communicationGroup != Communicator::NullGroup ) {
         Communicator::Allreduce( &localResult, &result, 1, MPI_LAND, communicationGroup );
      }
      return result;
   }

   template< typename Communicator >
   static bool GE( const T1& a, const T2& b, const typename Communicator::CommunicationGroup& communicationGroup )
   {
      return ! LT( a, b, communicationGroup );
   }
};

      } //namespace Expressions
   } // namespace Containers
} // namespace TNL