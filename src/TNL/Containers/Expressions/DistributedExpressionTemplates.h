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
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value >
struct DistributedUnaryExpressionTemplate
{
};

////
// Distributed binary expression template
template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value,
          ExpressionVariableType T2Type = ExpressionVariableTypeGetter< T2 >::value >
struct DistributedBinaryExpressionTemplate
{
};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct DistributedBinaryExpressionTemplate< T1, T2, Operation, VectorVariable, VectorVariable >
{
   using RealType = typename T1::RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using IsExpressionTemplate = bool;
   using CommunicatorType = Communicators::MpiCommunicator;
   using CommunicationGroup = typename CommunicatorType::CommunicationGroup;

   static_assert( std::is_same< typename T1::DeviceType, typename T2::DeviceType >::value, "Attempt to mix operands allocated on different device types." );
   static_assert( IsStaticType< T1 >::value == IsStaticType< T2 >::value, "Attempt to mix static and non-static operands in binary expression templates." );
   static constexpr bool isStatic() { return false; }

   DistributedBinaryExpressionTemplate( const T1& a, const T2& b, const CommunicationGroup& group )
      : op1( a ), op2( b ), communicationGroup( group ) {}

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
      return communicationGroup;
   }

   protected:
      const T1 op1;
      const T2 op2;
      CommunicationGroup communicationGroup;
      //typename OperandType< T1, DeviceType >::type op1;
      //typename OperandType< T2, DeviceType >::type op2;
};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct DistributedBinaryExpressionTemplate< T1, T2, Operation, VectorVariable, ArithmeticVariable >
{
   using RealType = typename T1::RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using CommunicatorType = Communicators::MpiCommunicator;
   using CommunicationGroup = typename CommunicatorType::CommunicationGroup;

   using IsExpressionTemplate = bool;
   static constexpr bool isStatic() { return false; }

   DistributedBinaryExpressionTemplate( const T1& a, const T2& b, const CommunicationGroup& group )
      : op1( a ), op2( b ), communicationGroup( group ){}

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
      return communicationGroup;
   }

   protected:
      const T1 op1;
      const T2 op2;
      CommunicationGroup communicationGroup;
      //typename OperandType< T1, DeviceType >::type op1;
      //typename OperandType< T2, DeviceType >::type op2;
};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct DistributedBinaryExpressionTemplate< T1, T2, Operation, ArithmeticVariable, VectorVariable >
{
   using RealType = typename T2::RealType;
   using DeviceType = typename T2::DeviceType;
   using IndexType = typename T2::IndexType;
   using CommunicatorType = Communicators::MpiCommunicator;
   using CommunicationGroup = typename CommunicatorType::CommunicationGroup;

   using IsExpressionTemplate = bool;
   static constexpr bool isStatic() { return false; }

   DistributedBinaryExpressionTemplate( const T1& a, const T2& b, const CommunicationGroup& group )
      : op1( a ), op2( b ), communicationGroup( group ){}

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
      return communicationGroup;
   }

   protected:
      const T1 op1;
      const T2 op2;
      CommunicationGroup communicationGroup;
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
          typename Parameter >
struct DistributedUnaryExpressionTemplate< T1, Operation, Parameter, VectorVariable >
{
   using RealType = typename T1::RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using IsExpressionTemplate = bool;
   using CommunicatorType = Communicators::MpiCommunicator;
   using CommunicationGroup = typename CommunicatorType::CommunicationGroup;
   static constexpr bool isStatic() { return false; }

   DistributedUnaryExpressionTemplate( const T1& a, const Parameter& p, const CommunicationGroup& group )
   : operand( a ), parameter( p ), communicationGroup( group ) {}

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
      return communicationGroup;
   }

   protected:
      const T1 operand;
      //typename OperandType< T1, DeviceType >::type operand;
      Parameter parameter;
      CommunicationGroup communicationGroup;
};

////
// Distributed unary expression template with no parameter
template< typename T1,
          template< typename > class Operation >
struct DistributedUnaryExpressionTemplate< T1, Operation, void, VectorVariable >
{
   using RealType = typename T1::RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using IsExpressionTemplate = bool;
   using CommunicatorType = Communicators::MpiCommunicator;
   using CommunicationGroup = typename CommunicatorType::CommunicationGroup;
   static constexpr bool isStatic() { return false; }

   DistributedUnaryExpressionTemplate( const T1& a, const CommunicationGroup& group )
      : operand( a ), communicationGroup( group ){}

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
      return communicationGroup;
   }

   protected:
      const T1 operand; // TODO: fix
      //typename std::add_const< typename OperandType< T1, DeviceType >::type >::type operand;
      CommunicationGroup communicationGroup;
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
          template< typename, typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
   Containers::Expressions::Addition >
operator + ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Addition >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::Addition >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
   Containers::Expressions::Addition >
operator + ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
      Containers::Expressions::Addition >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Addition >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Addition >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1,LOperation >& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Addition >( a, b );
}

////
// Binary expression subtraction
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
   Containers::Expressions::Subtraction >
operator - ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
   Containers::Expressions::Subtraction >
operator - ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1,LOperation >& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Subtraction >( a, b );
}

////
// Binary expression multiplication
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
   Containers::Expressions::Multiplication >
operator * ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
   Containers::Expressions::Multiplication >
operator * ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1,LOperation >& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Multiplication >( a, b );
}

////
// Binary expression division
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
   Containers::Expressions::Division >
operator + ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Division >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::Division >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
   Containers::Expressions::Division >
operator / ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
      Containers::Expressions::Division >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Division >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Division >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1,LOperation >& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Division >( a, b );
}


////
// Binary expression min
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Min >
min ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
      const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::Min >
min( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
   Containers::Expressions::Min >
min( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
     const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::Min >
min( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& a,
     const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
   Containers::Expressions::Min >
min( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
      Containers::Expressions::Min >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Min >
min( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a,
     const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Min >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Min >
min( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Min >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Min >
min( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1,LOperation >& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Min >( a, b );
}

////
// Binary expression max
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Max >
max( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::Max >
max( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
   Containers::Expressions::Max >
operator + ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::Max >
max( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& a,
     const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
   Containers::Expressions::Max >
max( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >,
      Containers::Expressions::Max >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Max >
max( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a,
     const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Max >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Max >
max( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Max >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::DistributedBinaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Max >
max( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1,LOperation >& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Max >( a, b );
}


////
// Comparison operator ==
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator == ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template EQ< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator == ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::template EQ< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
bool
operator == ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >& a,
              const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >;
   using Right = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::template EQ< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator == ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
              const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template EQ< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
bool
operator == ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
              const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   using Left = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template EQ< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator == ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
              const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template EQ< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
bool
operator == ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, RParameter >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template EQ< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

////
// Comparison operator !=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator != ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
              const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template NE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator != ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::template NE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
bool
operator != ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
              const Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, RParameter >& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template NE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
bool
operator != ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >& a,
              const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >;
   using Right = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::template NE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator != ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
              const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template NE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
bool
operator != ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
              const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   using Left = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template NE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator != ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
              const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template NE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
bool
operator != ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
              const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, RParameter >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template NE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

////
// Comparison operator <
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator < ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template LT< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator < ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::template LT< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
bool
operator < ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
              const Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, RParameter >& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template LT< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
bool
operator < ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >;
   using Right = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::template LT< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator < ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template LT< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
bool
operator < ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   using Left = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template LT< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator < ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template LT< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
bool
operator < ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, RParameter >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template LT< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

////
// Comparison operator <=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator <= ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
              const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template LE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator <= ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::template LE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
bool
operator <= ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
              const Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, RParameter >& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template LE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
bool
operator <= ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >& a,
              const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >;
   using Right = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::template LE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator <= ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
              const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template LE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
bool
operator <= ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
              const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   using Left = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template LE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator <= ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
              const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template LE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
bool
operator <= ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, RParameter >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template LE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

////
// Comparison operator >
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator > ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template GT< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator > ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::template GT< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
bool
operator > ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >;
   using Right = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::template GT< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator > ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
             const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template GT< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
bool
operator > ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
             const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   using Left = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template GT< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator > ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
             const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template GT< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
bool
operator > ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, RParameter >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template GT< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

////
// Comparison operator >=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator >= ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
              const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template GE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator >= ( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::template GE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
bool
operator >= ( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >& a,
              const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >;
   using Right = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   return Containers::Expressions::DistributedComparison< Left, Right >::template GE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator >= ( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
              const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template GE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
bool
operator >= ( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
              const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   using Left = typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation, Parameter >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template GE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator >= ( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
              const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template GE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
bool
operator >= ( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation, RParameter >& b )
{
   using Left = Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return Containers::Expressions::DistributedComparison< Left, Right >::template GE< typename Left::CommunicatorType >( a, b, a.getCommunicationGroup() );
}

////
// Unary operations

////
// Minus
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Minus >
operator -( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Minus >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Abs >
operator -( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Minus >( a );
}

////
// Abs
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Abs >
abs( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Abs >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Abs >
abs( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Abs >( a );
}

////
// Sin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Sin >
sin( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Sin >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Sin >
sin( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Sin >( a );
}

////
// Cos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Cos >
cos( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Cos >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Cos >
cos( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Cos >( a );
}

////
// Tan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Tan >
tan( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Tan >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Tan >
tan( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Tan >( a );
}

////
// Sqrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Sqrt >
sqrt( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Sqrt >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Sqrt >
sqrt( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Sqrt >( a );
}

////
// Cbrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Cbrt >
cbrt( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Cbrt >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Cbrt >
cbrt( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Cbrt >( a );
}

////
// Pow
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Real >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Pow >
pow( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a, const Real& exp )
{
   auto e = Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Pow >( a );
   e.parameter.set( exp );
   return e;
}

template< typename L1,
          template< typename > class LOperation,
          typename Real >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Pow >
pow( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a, const Real& exp )
{
   auto e = Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Pow >( a );
   e.parameter.set( exp );
   return e;
}

////
// Floor
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Sin >
floor( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Floor >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Floor >
floor( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Floor >( a );
}

////
// Ceil
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Ceil >
ceil( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Ceil >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Ceil >
sin( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Ceil >( a );
}

////
// Asin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Asin >
asin( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Asin >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Asin >
asin( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Asin >( a );
}

////
// Acos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Acos >
cos( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Acos >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Acos >
acos( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Cos >( a );
}

////
// Atan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Atan >
tan( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Atan >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Atan >
atan( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Atan >( a );
}

////
// Sinh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Sinh >
sinh( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Sinh >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Sinh >
sinh( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Sinh >( a );
}

////
// Cosh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Cosh >
cosh( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Cosh >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Cosh >
cosh( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Cosh >( a );
}

////
// Tanh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Tanh >
cosh( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Tanh >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Tanh >
tanh( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Tanh >( a );
}

////
// Log
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Log >
log( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Log >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Log >
log( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Log >( a );
}

////
// Log10
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Log10 >
log10( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Log10 >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Log10 >
log10( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Log10 >( a );
}

////
// Log2
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Log2 >
log2( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Log2 >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Log2 >
log2( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Log2 >( a );
}

////
// Exp
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Exp >
exp( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Exp >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
const Containers::Expressions::DistributedUnaryExpressionTemplate<
   Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >,
   Containers::Expressions::Exp >
exp( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate<
      Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >,
      Containers::Expressions::Exp >( a );
}

////
// Vertical operations - min
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::RealType
min( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   using CommunicatorType = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::CommunicatorType;
   using Real = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::RealType;
   Real result = std::numeric_limits< Real >::max();
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionMin( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MIN, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
double 
min( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   typename Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType aux;
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
          typename Index >
auto
argMin( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a, Index& arg ) -> decltype( ExpressionArgMin( a, arg ) )
{
   throw Exceptions::NotImplementedError( "agrMin for distributed vector view is not implemented yet." );
   return ExpressionArgMin( a, arg );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter,
          typename Index >
auto
argMin( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >& a, Index& arg ) -> decltype( ExpressionMin( a, arg ) )
{
   throw Exceptions::NotImplementedError( "agrMin for distributed vector view is not implemented yet." );
   return ExpressionArgMin( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::RealType
max( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   using Real = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::RealType;
   using CommunicatorType = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::CommunicatorType;
   Real result = std::numeric_limits< Real >::min();
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionMax( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MAX, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
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
          typename Index >
auto
argMax( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a, Index& arg ) -> decltype( ExpressionArgMax( a, arg ) )
{
   throw Exceptions::NotImplementedError( "agrMax for distributed vector view is not implemented yet." );
   return ExpressionArgMax( a, arg );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter,
          typename Index >
auto
argMax( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation, Parameter >& a, Index& arg ) -> decltype( ExpressionMax( a, arg ) )
{
   throw Exceptions::NotImplementedError( "agrMax for distributed vector view is not implemented yet." );
   return ExpressionArgMax( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::RealType
sum( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   using Real = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::RealType;
   using CommunicatorType = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::CommunicatorType;
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionSum( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
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
          typename Real >
typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::RealType
lpNorm( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a, const Real& p )
{
   using CommunicatorType = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::CommunicatorType;
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
          template< typename, typename > class LOperation >
typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::RealType
product( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   using Real = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::RealType;
   using CommunicatorType = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::CommunicatorType;
   Real result = ( Real ) 1.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionProduct( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_PROD, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
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
          template< typename, typename > class LOperation >
bool
logicalOr( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   using Real = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::RealType;
   using CommunicatorType = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::CommunicatorType;
   Real result = 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionLogicalOr( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LOR, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
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
          template< typename, typename > class LOperation >
bool
logicalAnd( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   using CommunicatorType = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::CommunicatorType;
   bool result = false;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const bool localResult = Containers::Expressions::ExpressionLogicalAnd( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
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
          template< typename, typename > class LOperation >
typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::RealType
binaryOr( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   using Real = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::RealType;
   using CommunicatorType = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::CommunicatorType;
   Real result = ( Real ) 0.0;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionBinaryOr( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_BOR, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
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
          template< typename, typename > class LOperation >
typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::RealType
binaryAnd( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   using Real = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::RealType;
   using CommunicatorType = typename Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >::CommunicatorType;
   Real result = std::numeric_limits< Real >::max;
   if( a.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const Real localResult = Containers::Expressions::ExpressionBinaryAnd( a );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_BAND, a.getCommunicationGroup() );
   }
   return result;
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
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
          template< typename, typename > class ROperation >
auto
operator,( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator,( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator,( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a,
           const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator,( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const typename Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation >& b )
-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
dot( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
dot( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a,
     const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
dot( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const typename Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation >& b )
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
   typename Reduction,
   typename VolatileReduction,
   typename Result >
Result evaluateAndReduce( Vector& lhs,
   const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& expression,
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
   typename Reduction,
   typename VolatileReduction,
   typename Result >
Result evaluateAndReduce( Vector& lhs,
   const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& expression,
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
   typename Reduction,
   typename VolatileReduction,
   typename Result >
Result addAndReduce( Vector& lhs,
   const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& expression,
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
   typename Reduction,
   typename VolatileReduction,
   typename Result >
Result addAndReduce( Vector& lhs,
   const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& expression,
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
   typename Reduction,
   typename VolatileReduction,
   typename Result >
Result addAndReduceAbs( Vector& lhs,
   const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& expression,
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
   typename Reduction,
   typename VolatileReduction,
   typename Result >
Result addAndReduceAbs( Vector& lhs,
   const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& expression,
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
          template< typename, typename > class Operation >
std::ostream& operator << ( std::ostream& str, const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getLocalSize() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( expression.getLocalSize() - 1 ) << " ]";
   return str;
}

template< typename T,
          template< typename > class Operation,
          typename Parameter >
std::ostream& operator << ( std::ostream& str, const Containers::Expressions::DistributedUnaryExpressionTemplate< T, Operation, Parameter >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getLocalSize() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( expression.getLocalSize() - 1 ) << " ]";
   return str;
}
} // namespace TNL
