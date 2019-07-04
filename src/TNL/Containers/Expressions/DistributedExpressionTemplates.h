/***************************************************************************
                          DistributedExpressionTemplates.h  -  description
                             -------------------
    begin                : Jun 28, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>
#include <TNL/Containers/Expressions/ExpressionTemplatesOperations.h>
#include <TNL/Containers/Expressions/ExpressionVariableType.h>
#include <TNL/Containers/Expressions/DistributedComparison.h>
#include <TNL/Containers/Expressions/IsStatic.h>

namespace TNL {
   namespace Containers {
      namespace Expressions {

////
// Distributed unary expression template
template< typename T1,
          template< typename > class Operation,
          typename Parameter = void,
          typename Communicator = Communicators::MpiCommunicator,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value >
struct DistributedUnaryExpressionTemplate
{
};

////
// Distributed binary expression template
template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator = Communicators::MpiCommunicator,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value,
          ExpressionVariableType T2Type = ExpressionVariableTypeGetter< T2 >::value >
struct DistributedBinaryExpressionTemplate
{
};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
struct DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator, VectorVariable, VectorVariable >
{
   using RealType = typename T1::RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using IsExpressionTemplate = bool;
   using CommunicatorType = Communicator;
   using CommunicationGroup = typename CommunicatorType::CommunicationGroup;

   static_assert( std::is_same< typename T1::DeviceType, typename T2::DeviceType >::value, "Attempt to mix operands allocated on different device types." );
   static_assert( IsStaticType< T1 >::value == IsStaticType< T2 >::value, "Attempt to mix static and non-static operands in binary expression templates." );
   static constexpr bool isStatic() { return false; }

   DistributedBinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   static DistributedBinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return DistributedBinaryExpressionTemplate( a, b );
   }

   RealType getElement( const IndexType i ) const
   {
       return Operation< typename T1::RealType, typename T2::RealType >::evaluate( op1.getElement( i ), op2.getElement( i ) );
   }

   __cuda_callable__
   RealType operator[]( const IndexType i ) const
   {
       return Operation< typename T1::RealType, typename T2::RealType >::evaluate( op1[ i ], op2[ i ] );
   }

   __cuda_callable__
   int getSize() const
   {
       return op1.getSize();
   }

   CommunicationGroup getCommunicationGroup() const
   {
      TNL_ASSERT_EQ( op1.getCommunicationGroup(), op2.getCommunicationGroup(), "Cannot create expression from operands using different communication groups." );
      return op1.getCommunicationGroup();
   }

   protected:
      const T1 op1;
      const T2 op2;
      //typename OperandType< T1, DeviceType >::type op1;
      //typename OperandType< T2, DeviceType >::type op2;
};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
struct DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator, VectorVariable, ArithmeticVariable >
{
   using RealType = typename T1::RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using CommunicatorType = Communicator;
   using CommunicationGroup = typename CommunicatorType::CommunicationGroup;

   using IsExpressionTemplate = bool;
   static constexpr bool isStatic() { return false; }

   DistributedBinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   DistributedBinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return DistributedBinaryExpressionTemplate( a, b );
   }

   RealType getElement( const IndexType i ) const
   {
       return Operation< typename T1::RealType, T2 >::evaluate( op1.getElement( i ), op2 );
   }

   __cuda_callable__
   RealType operator[]( const IndexType i ) const
   {
       return Operation< typename T1::RealType, T2 >::evaluate( op1[ i ], op2 );
   }

   __cuda_callable__
   int getSize() const
   {
       return op1.getSize();
   }

   CommunicationGroup getCommunicationGroup() const
   {
      return op1.getCommunicationGroup();
   }

   protected:
      const T1 op1;
      const T2 op2;
      //typename OperandType< T1, DeviceType >::type op1;
      //typename OperandType< T2, DeviceType >::type op2;
};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
struct DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator, ArithmeticVariable, VectorVariable >
{
   using RealType = typename T2::RealType;
   using DeviceType = typename T2::DeviceType;
   using IndexType = typename T2::IndexType;
   using CommunicatorType = Communicator;
   using CommunicationGroup = typename CommunicatorType::CommunicationGroup;

   using IsExpressionTemplate = bool;
   static constexpr bool isStatic() { return false; }

   DistributedBinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   DistributedBinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return DistributedBinaryExpressionTemplate( a, b );
   }

   RealType getElement( const IndexType i ) const
   {
       return Operation< T1, typename T2::RealType >::evaluate( op1, op2.getElement( i ) );
   }

   __cuda_callable__
   RealType operator[]( const IndexType i ) const
   {
       return Operation< T1, typename T2::RealType >::evaluate( op1, op2[ i ] );
   }

   __cuda_callable__
   int getSize() const
   {
       return op2.getSize();
   }

   CommunicationGroup getCommunicationGroup() const
   {
      return op2.getCommunicationGroup();
   }

   protected:
      const T1 op1;
      const T2 op2;
      //typename OperandType< T1, DeviceType >::type op1;
      //typename OperandType< T2, DeviceType >::type op2;
};

////
// Distributed unary expression template
//
// Parameter type serves mainly for pow( base, exp ). Here exp is parameter we need
// to pass to pow.
template< typename T1,
          template< typename > class Operation,
          typename Parameter,
          typename Communicator >
struct DistributedUnaryExpressionTemplate< T1, Operation, Parameter, Communicator, VectorVariable >
{
   using RealType = typename T1::RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using IsExpressionTemplate = bool;
   using CommunicatorType = Communicator;
   using CommunicationGroup = typename CommunicatorType::CommunicationGroup;
   static constexpr bool isStatic() { return false; }

   DistributedUnaryExpressionTemplate( const T1& a, const Parameter& p )
   : operand( a ), parameter( p ) {}

   static DistributedUnaryExpressionTemplate evaluate( const T1& a )
   {
      return DistributedUnaryExpressionTemplate( a );
   }

   RealType getElement( const IndexType i ) const
   {
       return Operation< typename T1::RealType >::evaluate( operand.getElement( i ), parameter );
   }

   __cuda_callable__
   RealType operator[]( const IndexType i ) const
   {
       return Operation< typename T1::RealType >::evaluate( operand[ i ], parameter );
   }

   __cuda_callable__
   int getSize() const
   {
       return operand.getSize();
   }

   void set( const Parameter& p ) { parameter = p; }

   const Parameter& get() { return parameter; }

   CommunicationGroup getCommunicationGroup() const
   {
      return operand.getCommunicationGroup();
   }

   protected:
      const T1 operand;
      //typename OperandType< T1, DeviceType >::type operand;
      Parameter parameter;
};

////
// Distributed unary expression template with no parameter
template< typename T1,
          template< typename > class Operation,
          typename Communicator >
struct DistributedUnaryExpressionTemplate< T1, Operation, void, Communicator, VectorVariable >
{
   using RealType = typename T1::RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using IsExpressionTemplate = bool;
   using CommunicatorType = Communicator;
   using CommunicationGroup = typename CommunicatorType::CommunicationGroup;
   static constexpr bool isStatic() { return false; }

   DistributedUnaryExpressionTemplate( const T1& a ): operand( a ){}

   static DistributedUnaryExpressionTemplate evaluate( const T1& a )
   {
      return DistributedUnaryExpressionTemplate( a );
   }

   RealType getElement( const IndexType i ) const
   {
       return Operation< typename T1::RealType >::evaluate( operand.getElement( i ) );
   }

   __cuda_callable__
   RealType operator[]( const IndexType i ) const
   {
       return Operation< typename T1::RealType >::evaluate( operand[ i ] );
   }

   __cuda_callable__
   int getSize() const
   {
       return operand.getSize();
   }

   CommunicationGroup getCommunicationGroup() const
   {
      return operand.getCommunicationGroup();
   }

   protected:
      const T1 operand; // TODO: fix
      //typename std::add_const< typename OperandType< T1, DeviceType >::type >::type operand;
};

      } //namespace Expressions
   } //namespace Containers

////
// All operations are supposed to be in namespace TNL

////
// Binary expressions addition
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
      Containers::Expressions::Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
      Containers::Expressions::Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
   Containers::Expressions::Addition >
operator + ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
      Containers::Expressions::Addition >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
      Containers::Expressions::Addition >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
   Containers::Expressions::Addition >
operator + ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
      Containers::Expressions::Addition >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
      Containers::Expressions::Addition >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
      Containers::Expressions::Addition >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1,LOperation, Communicator >& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
      Containers::Expressions::Addition >( a, b );
}

////
// Binary expression subtraction
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
   Containers::Expressions::Subtraction >
operator - ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
   Containers::Expressions::Subtraction >
operator - ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1,LOperation, Communicator >& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
      Containers::Expressions::Subtraction >( a, b );
}

////
// Binary expression multiplication
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
   Containers::Expressions::Multiplication >
operator * ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
   Containers::Expressions::Multiplication >
operator * ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1,LOperation, Communicator >& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
      Containers::Expressions::Multiplication >( a, b );
}

////
// Binary expression division
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
      Containers::Expressions::Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
      Containers::Expressions::Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
   Containers::Expressions::Division >
operator + ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
      Containers::Expressions::Division >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
      Containers::Expressions::Division >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
   Containers::Expressions::Division >
operator / ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
      Containers::Expressions::Division >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
      Containers::Expressions::Division >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
      Containers::Expressions::Division >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1,LOperation, Communicator >& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
      Containers::Expressions::Division >( a, b );
}


////
// Binary expression min
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
   Containers::Expressions::Min >
min ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
      const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
   Containers::Expressions::Min >
min( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& a,
     const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
   Containers::Expressions::Min >
min( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& a,
     const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
   Containers::Expressions::Min >
min( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& a,
     const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
   Containers::Expressions::Min >
min( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
      Containers::Expressions::Min >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
   Containers::Expressions::Min >
min( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a,
     const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
      Containers::Expressions::Min >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
   Containers::Expressions::Min >
min( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
      Containers::Expressions::Min >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
   Containers::Expressions::Min >
min( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1,LOperation, Communicator >& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
      Containers::Expressions::Min >( a, b );
}

////
// Binary expression max
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
   Containers::Expressions::Max >
max( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
     const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
   Containers::Expressions::Max >
max( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& a,
     const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
   Containers::Expressions::Max >
operator + ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType,
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
   Containers::Expressions::Max >
max( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& a,
     const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
   Containers::Expressions::Max >
max( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType,
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >,
      Containers::Expressions::Max >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
   Containers::Expressions::Max >
max( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a,
     const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >,
      Containers::Expressions::Max >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
   Containers::Expressions::Max >
max( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
      Containers::Expressions::Max >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
   Containers::Expressions::Max >
max( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1,LOperation, Communicator >& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, Communicator >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >,
      Containers::Expressions::Max >( a, b );
}


////
// Comparison operator ==
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
bool
operator == ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::EQ( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
bool
operator == ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& a,
              const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >;
   using Right = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::template EQ< Communicator >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
bool
operator == ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& a,
              const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >;
   using Right = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::EQ( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
bool
operator == ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& a,
              const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& b )
{
   using Left = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::EQ( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
bool
operator == ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& a,
              const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& b )
{
   using Left = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::EQ( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
bool
operator == ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a,
              const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::EQ( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
bool
operator == ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, Communicator >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::EQ( a, b );
}

////
// Comparison operator !=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
bool
operator != ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::NE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
bool
operator != ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& a,
              const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >;
   using Right = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::NE( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
bool
operator != ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& a,
              const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >;
   using Right = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::NE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
bool
operator != ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& a,
              const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& b )
{
   using Left = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::NE( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
bool
operator != ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& a,
              const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& b )
{
   using Left = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::NE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
bool
operator != ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a,
              const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::NE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
bool
operator != ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, Communicator >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::NE( a, b );
}

////
// Comparison operator <
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
bool
operator < ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::LT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
bool
operator < ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >;
   using Right = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::LT( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
bool
operator < ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >;
   using Right = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::LT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
bool
operator < ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& b )
{
   using Left = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::LT( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
bool
operator < ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& b )
{
   using Left = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::LT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
bool
operator < ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::LT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
bool
operator < ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, Communicator >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::LT( a, b );
}

////
// Comparison operator <=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
bool
operator <= ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
              const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::LE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
bool
operator <= ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& a,
              const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >;
   using Right = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::LE( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
bool
operator <= ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& a,
              const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >;
   using Right = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::LE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
bool
operator <= ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& a,
              const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& b )
{
   using Left = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::LE( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
bool
operator <= ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& a,
              const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& b )
{
   using Left = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::LE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
bool
operator <= ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a,
              const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::LE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
bool
operator <= ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, Communicator >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::LE( a, b );
}

////
// Comparison operator >
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
bool
operator > ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::GT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
bool
operator > ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >;
   using Right = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::GT( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
bool
operator > ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >;
   using Right = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::GT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
bool
operator > ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& b )
{
   using Left = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::GT( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
bool
operator > ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& b )
{
   using Left = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::GT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
bool
operator > ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::GT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
bool
operator > ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, Communicator >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::GT( a, b );
}

////
// Comparison operator >=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
bool
operator >= ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
              const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::GE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
bool
operator >= ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& a,
              const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >;
   using Right = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::GE( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
bool
operator >= ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& a,
              const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >;
   using Right = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::GE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
bool
operator >= ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& a,
              const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& b )
{
   using Left = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::GE( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Communicator >
bool
operator >= ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType& a,
              const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& b )
{
   using Left = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >::RealType;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::GE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
bool
operator >= ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a,
              const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::GE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
bool
operator >= ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, Communicator >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, Communicator >;
   return Containers::Expressions::DistributedComparison< Left, Right >::GE( a, b );
}

////
// Unary operations

////
// Minus
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Minus >
operator -( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Minus >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Abs >
operator -( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Minus >( a );
}

////
// Abs
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Abs >
abs( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Abs >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Abs >
abs( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Abs >( a );
}

////
// Sin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Sin >
sin( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Sin >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Sin >
sin( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Sin >( a );
}

////
// Cos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Cos >
cos( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Cos >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Cos >
cos( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Cos >( a );
}

////
// Tan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Tan >
tan( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Tan >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Tan >
tan( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Tan >( a );
}

////
// Sqrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Sqrt >
sqrt( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Sqrt >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Sqrt >
sqrt( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Sqrt >( a );
}

////
// Cbrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Cbrt >
cbrt( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Cbrt >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Cbrt >
cbrt( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Cbrt >( a );
}

////
// Pow
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator,
          typename Real >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Pow >
pow( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a, const Real& exp )
{
   auto e = Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Pow >( a );
   e.parameter.set( exp );
   return e;
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator,
          typename Real >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Pow >
pow( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a, const Real& exp )
{
   auto e = Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Pow >( a );
   e.parameter.set( exp );
   return e;
}

////
// Floor
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Sin >
floor( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Floor >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Floor >
floor( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Floor >( a );
}

////
// Ceil
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Ceil >
ceil( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Ceil >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Ceil >
sin( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Ceil >( a );
}

////
// Asin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Asin >
asin( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Asin >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Asin >
asin( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Asin >( a );
}

////
// Acos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Acos >
cos( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Acos >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Acos >
acos( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Cos >( a );
}

////
// Atan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Atan >
tan( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Atan >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Atan >
atan( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Atan >( a );
}

////
// Sinh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Sinh >
sinh( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Sinh >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Sinh >
sinh( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Sinh >( a );
}

////
// Cosh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Cosh >
cosh( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Cosh >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Cosh >
cosh( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Cosh >( a );
}

////
// Tanh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Tanh >
cosh( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Tanh >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Tanh >
tanh( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Tanh >( a );
}

////
// Log
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Log >
log( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Log >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Log >
log( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Log >( a );
}

////
// Log10
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Log10 >
log10( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Log10 >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Log10 >
log10( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Log10 >( a );
}

////
// Log2
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Log2 >
log2( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Log2 >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Log2 >
log2( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Log2 >( a );
}

////
// Exp
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
   Containers::Expressions::Exp >
exp( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >,
      Containers::Expressions::Exp >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Communicator >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
   Containers::Expressions::Exp >
exp( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >,
      Containers::Expressions::Exp >( a );
}

////
// Vertical operations - min
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::RealType
min( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   using CommunicatorType = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::CommunicatorType;
   using Real = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::RealType;
   Real result = std::numeric_limits< Real >::max();
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionMin( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MIN, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter,
          typename Communicator >
typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
min( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   using CommunicatorType = typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::CommunicatorType;
   using Real = typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType;
   Real result = std::numeric_limits< Real >::max();
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionMin( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MIN, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator,
          typename Index >
auto
argMin( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a, Index& arg ) -> decltype( ExpressionArgMin( a, arg ) )
{
   throw Exceptions::NotImplementedError( "agrMin for distributed vector view is not implemented yet." );
   return ExpressionArgMin( a, arg );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter,
          typename Communicator,
          typename Index >
auto
argMin( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >& a, Index& arg ) -> decltype( ExpressionMin( a, arg ) )
{
   throw Exceptions::NotImplementedError( "agrMin for distributed vector view is not implemented yet." );
   return ExpressionArgMin( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::RealType
max( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   using Real = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::RealType;
   using CommunicatorType = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::CommunicatorType;
   Real result = std::numeric_limits< Real >::min();
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionMax( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MAX, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter,
          typename Communicator >
typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
max( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   using Real = typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType;
   using CommunicatorType = typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::CommunicatorType;
   Real result = std::numeric_limits< Real >::min();
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionMax( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MAX, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator,
          typename Index >
auto
argMax( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a, Index& arg ) -> decltype( ExpressionArgMax( a, arg ) )
{
   throw Exceptions::NotImplementedError( "agrMax for distributed vector view is not implemented yet." );
   return ExpressionArgMax( a, arg );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter,
          typename Communicator,
          typename Index >
auto
argMax( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >& a, Index& arg ) -> decltype( ExpressionMax( a, arg ) )
{
   throw Exceptions::NotImplementedError( "agrMax for distributed vector view is not implemented yet." );
   return ExpressionArgMax( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::RealType
sum( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   using Real = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::RealType;
   using CommunicatorType = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::CommunicatorType;
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionSum( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter,
          typename Communicator >
typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
sum( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   using Real = typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType;
   using CommunicatorType = typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::CommunicatorType;
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionSum( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator,
          typename Real >
typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::RealType
lpNorm( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a, const Real& p )
{
   using CommunicatorType = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::CommunicatorType;
   Real result = ( Real ) 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::pow( Containers::Expressions::ExpressionLpNorm( a, p ), p );
      CommunicatorType::template Allreduce< Real >( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return TNL::pow( result, 1.0 / p );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter,
          typename Communicator,
          typename Real >
typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
lpNorm( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >& a, const Real& p )
{
   using CommunicatorType = typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::CommunicatorType;
   Real result = ( Real ) 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = TNL::pow( Containers::Expressions::ExpressionLpNorm( a, p ), p );
      CommunicatorType::template Allreduce< Real >( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return TNL::pow( result, 1.0 / p );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::RealType
product( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   using Real = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::RealType;
   using CommunicatorType = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::CommunicatorType;
   Real result = ( Real ) 1.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionProduct( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_PROD, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter,
          typename Communicator >
typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
product( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   using Real = typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType;
   using CommunicatorType = typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::CommunicatorType;
   Real result = ( Real ) 1.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionProduct( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_PROD, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
bool
logicalOr( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   using Real = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::RealType;
   using CommunicatorType = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::CommunicatorType;
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionLogicalOr( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LOR, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter,
          typename Communicator >
bool
logicalOr( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   using Real = typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType;
   using CommunicatorType = typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::CommunicatorType;
   bool result = false;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionLogicalOr( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LOR, a.getCommunicationGroup() );
   }
   return result;
}


template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
bool
logicalAnd( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   using CommunicatorType = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::CommunicatorType;
   bool result = false;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const bool localResult = Containers::Expressions::ExpressionLogicalAnd( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter,
          typename Communicator >
bool
logicalAnd( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   using CommunicatorType = typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::CommunicatorType;
   bool result = false;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const bool localResult = Containers::Expressions::ExpressionLogicalAnd( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::RealType
binaryOr( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   using Real = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::RealType;
   using CommunicatorType = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::CommunicatorType;
   Real result = ( Real ) 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionBinaryOr( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_BOR, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter,
          typename Communicator >
typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
binaryOr( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   using Real = typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType;
   using CommunicatorType = typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::CommunicatorType;
   Real result = ( Real ) 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionBinaryOr( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_BOR, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Communicator >
typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::RealType
binaryAnd( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a )
{
   using Real = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::RealType;
   using CommunicatorType = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >::CommunicatorType;
   Real result = std::numeric_limits< Real >::max;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionBinaryAnd( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_BAND, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter,
          typename Communicator >
typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
binaryAnd( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   using Real = typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType;
   using CommunicatorType = typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::CommunicatorType;
   Real result = std::numeric_limits< Real >::max;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionBinaryAnd( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_BAND, a.getCommunicationGroup() );
   }
   return result;
}

////
// Scalar product
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
auto
operator,( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
           const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
auto
operator,( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& a,
           const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& b )
-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
auto
operator,( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a,
           const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
auto
operator,( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
           const typename Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, Communicator >& b )
-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
auto
dot( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& a,
     const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >::RealType& b )
-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation,
          typename Communicator >
auto
dot( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Communicator >& a,
     const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation, Communicator >& b )
-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename Communicator >
auto
dot( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation, Communicator >& a,
     const typename Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, Communicator >& b )
-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}


////
// Evaluation with reduction
template< typename Vector,
   typename T1,
   typename T2,
   template< typename, typename > class Operation,
   typename Communicator,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
Result evaluateAndReduce( Vector& lhs,
   const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType { return ( lhs_data[ i ] = expression[ i ] ); };
   return Containers::Algorithms::Reduction< DeviceType >::reduce( lhs.getLocalSize(), reduction, volatileReduction, fetch, zero );
}

template< typename Vector,
   typename T1,
   template< typename > class Operation,
   typename Communicator,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
Result evaluateAndReduce( Vector& lhs,
   const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType { return ( lhs_data[ i ] = expression[ i ] ); };
   return Containers::Algorithms::Reduction< DeviceType >::reduce( lhs.getLocalSize(), reduction, volatileReduction, fetch, zero );
}

////
// Addition and reduction
template< typename Vector,
   typename T1,
   typename T2,
   template< typename, typename > class Operation,
   typename Communicator,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
Result addAndReduce( Vector& lhs,
   const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType {
      const RealType aux = expression[ i ];
      lhs_data[ i ] += aux;
      return aux;
   };
   return Containers::Algorithms::Reduction< DeviceType >::reduce( lhs.getLocalSize(), reduction, volatileReduction, fetch, zero );
}

template< typename Vector,
   typename T1,
   template< typename > class Operation,
   typename Communicator,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
Result addAndReduce( Vector& lhs,
   const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType {
      const RealType aux = expression[ i ];
      lhs_data[ i ] += aux;
      return aux;
   };
   return Containers::Algorithms::Reduction< DeviceType >::reduce( lhs.getLocalSize(), reduction, volatileReduction, fetch, zero );
}

////
// Addition and reduction
template< typename Vector,
   typename T1,
   typename T2,
   template< typename, typename > class Operation,
   typename Communicator,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
Result addAndReduceAbs( Vector& lhs,
   const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType {
      const RealType aux = expression[ i ];
      lhs_data[ i ] += aux;
      return TNL::abs( aux );
   };
   return Containers::Algorithms::Reduction< DeviceType >::reduce( lhs.getLocalSize(), reduction, volatileReduction, fetch, zero );
}

template< typename Vector,
   typename T1,
   template< typename > class Operation,
   typename Communicator,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
Result addAndReduceAbs( Vector& lhs,
   const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Communicator >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType {
      const RealType aux = expression[ i ];
      lhs_data[ i ] += aux;
      return TNL::abs( aux );
   };
   return Containers::Algorithms::Reduction< DeviceType >::reduce( lhs.getLocalSize(), reduction, volatileReduction, fetch, zero );
}

////
// Output stream
template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          typename Communicator >
std::ostream& operator << ( std::ostream& str, const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation, Communicator >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getLocalSize() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( expression.getLocalSize() - 1 ) << " ]";
   return str;
}

template< typename T,
          template< typename > class Operation,
          typename Parameter,
          typename Communicator >
std::ostream& operator << ( std::ostream& str, const Containers::Expressions::DistributedUnaryExpressionTemplate< T, Operation, Parameter >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getLocalSize() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( expression.getLocalSize() - 1 ) << " ]";
   return str;
}
} // namespace TNL
