/***************************************************************************
                          ExpressionTemplates.h  -  description
                             -------------------
    begin                : Apr 18, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>
#include <TNL/Containers/Expressions/ExpressionTemplatesOperations.h>
#include <TNL/Containers/Expressions/ExpressionVariableType.h>
#include <TNL/Containers/Expressions/Comparison.h>
#include <TNL/Containers/Expressions/IsStatic.h>

namespace TNL {
   namespace Containers {
      namespace Expressions {

////
// Non-static unary expression template
template< typename T1,
          template< typename > class Operation,
          typename Parameter = void,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value >
struct UnaryExpressionTemplate
{
};

////
// Non-static binary expression template
template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value,
          ExpressionVariableType T2Type = ExpressionVariableTypeGetter< T2 >::value >
struct BinaryExpressionTemplate
{
};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct BinaryExpressionTemplate< T1, T2, Operation, VectorVariable, VectorVariable >
{
   using RealType = typename T1::RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using IsExpressionTemplate = bool;

   static_assert( std::is_same< typename T1::DeviceType, typename T2::DeviceType >::value, "Attempt to mix operands allocated on different device types." );
   static_assert( IsStaticType< T1 >::value == IsStaticType< T2 >::value, "Attempt to mix static and non-static operands in binary expression templates." );
   static constexpr bool isStatic() { return false; }

   BinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   static BinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return BinaryExpressionTemplate( a, b );
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

   protected:
      const T1 op1;
      const T2 op2;
      //typename OperandType< T1, DeviceType >::type op1;
      //typename OperandType< T2, DeviceType >::type op2;
};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct BinaryExpressionTemplate< T1, T2, Operation, VectorVariable, ArithmeticVariable >
{
   using RealType = typename T1::RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;

   using IsExpressionTemplate = bool;
   static constexpr bool isStatic() { return false; }

   BinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   BinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return BinaryExpressionTemplate( a, b );
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

   protected:
      const T1 op1;
      const T2 op2;
      //typename OperandType< T1, DeviceType >::type op1;
      //typename OperandType< T2, DeviceType >::type op2;
};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct BinaryExpressionTemplate< T1, T2, Operation, ArithmeticVariable, VectorVariable >
{
   using RealType = typename T2::RealType;
   using DeviceType = typename T2::DeviceType;
   using IndexType = typename T2::IndexType;

   using IsExpressionTemplate = bool;
   static constexpr bool isStatic() { return false; }

   BinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   BinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return BinaryExpressionTemplate( a, b );
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

   protected:
      const T1 op1;
      const T2 op2;
      //typename OperandType< T1, DeviceType >::type op1;
      //typename OperandType< T2, DeviceType >::type op2;
};

////
// Non-static unary expression template
//
// Parameter type serves mainly for pow( base, exp ). Here exp is parameter we need
// to pass to pow.
template< typename T1,
          template< typename > class Operation,
          typename Parameter >
struct UnaryExpressionTemplate< T1, Operation, Parameter, VectorVariable >
{
   using RealType = typename T1::RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using IsExpressionTemplate = bool;
   static constexpr bool isStatic() { return false; }

   UnaryExpressionTemplate( const T1& a, const Parameter& p )
   : operand( a ), parameter( p ) {}

   static UnaryExpressionTemplate evaluate( const T1& a )
   {
      return UnaryExpressionTemplate( a );
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

   protected:
      const T1 operand;
      //typename OperandType< T1, DeviceType >::type operand;
      Parameter parameter;
};

////
// Non-static unary expression template with no parameter
template< typename T1,
          template< typename > class Operation >
struct UnaryExpressionTemplate< T1, Operation, void, VectorVariable >
{
   using RealType = typename T1::RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using IsExpressionTemplate = bool;
   static constexpr bool isStatic() { return false; }

   UnaryExpressionTemplate( const T1& a ): operand( a ){}

   static UnaryExpressionTemplate evaluate( const T1& a )
   {
      return UnaryExpressionTemplate( a );
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
          template< typename, typename > class ROperation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   Containers::Expressions::Addition >
operator + ( const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
             const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Addition >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
   typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& a,
             const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
      typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::Addition >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
   Containers::Expressions::Addition >
operator + ( const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& a,
             const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
      Containers::Expressions::Addition >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
             const typename Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Addition >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Addition >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::UnaryExpressionTemplate< L1,LOperation >& a,
             const Containers::Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
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
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   Containers::Expressions::Subtraction >
operator - ( const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
             const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
   typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& a,
             const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
      typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
   Containers::Expressions::Subtraction >
operator - ( const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& a,
             const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
             const typename Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::UnaryExpressionTemplate< L1,LOperation >& a,
             const Containers::Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
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
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   Containers::Expressions::Multiplication >
operator * ( const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
             const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
   typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& a,
             const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
      typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
   Containers::Expressions::Multiplication >
operator * ( const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& a,
             const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
             const typename Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::UnaryExpressionTemplate< L1,LOperation >& a,
             const Containers::Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
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
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   Containers::Expressions::Division >
operator + ( const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
             const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Division >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
   typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& a,
             const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
      typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::Division >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
   Containers::Expressions::Division >
operator / ( const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& a,
             const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
      Containers::Expressions::Division >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
             const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Division >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Division >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::UnaryExpressionTemplate< L1,LOperation >& a,
             const Containers::Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
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
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Min >
min ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
      const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::Min >
min( const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   Containers::Expressions::Min >
min( const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
     const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
   typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::Min >
min( const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& a,
     const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
      typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
   Containers::Expressions::Min >
min( const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& a,
     const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
      Containers::Expressions::Min >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Min >
min( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
     const typename Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Min >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   typename Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Min >
min( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Min >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Min >
min( const Containers::Expressions::UnaryExpressionTemplate< L1,LOperation >& a,
     const Containers::Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
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
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Max >
max( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::Max >
max( const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   Containers::Expressions::Max >
operator + ( const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
             const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
   typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::Max >
max( const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& a,
     const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
      typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          template< typename > class Operation >
const Containers::Expressions::BinaryExpressionTemplate<
   typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
   Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
   Containers::Expressions::Max >
max( const typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType& a,
     const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      typename Containers::Expressions::UnaryExpressionTemplate< T1, Operation >::RealType,
      Containers::Expressions::UnaryExpressionTemplate< T1, Operation >,
      Containers::Expressions::Max >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Max >
max( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
     const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Max >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Max >
max( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
      Containers::Expressions::Max >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
const Containers::Expressions::BinaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Max >
max( const Containers::Expressions::UnaryExpressionTemplate< L1,LOperation >& a,
     const Containers::Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::BinaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >,
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
operator == ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::ComparisonEQ( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator == ( const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::ComparisonEQ( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator == ( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
              const typename Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::ComparisonEQ( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator == ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::ComparisonEQ( a, b );
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
operator != ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::ComparisonNE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator != ( const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::ComparisonNE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator != ( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
              const typename Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::ComparisonNE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator != ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
              const typename Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >& b )
{
   return Containers::Expressions::ComparisonNE( a, b );
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
operator < ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::ComparisonLT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator < ( const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::ComparisonLT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator < ( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
              const typename Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::ComparisonLT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator < ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
              const typename Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >& b )
{
   return Containers::Expressions::ComparisonLT( a, b );
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
operator <= ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::ComparisonLE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator <= ( const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::ComparisonLE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator <= ( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
              const typename Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::ComparisonLE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator <= ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
              const typename Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >& b )
{
   return Containers::Expressions::ComparisonLE( a, b );
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
operator > ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::ComparisonGT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator > ( const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::ComparisonGT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator > ( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
             const typename Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::ComparisonGT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator > ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >& b )
{
   return Containers::Expressions::ComparisonGT( a, b );
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
operator >= ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
              const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::ComparisonGE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator >= ( const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::ComparisonGE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator >= ( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
              const typename Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::ComparisonGE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator >= ( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
              const typename Containers::Expressions::UnaryExpressionTemplate< R1, ROperation >& b )
{
   return Containers::Expressions::ComparisonGE( a, b );
}

////
// Unary operations


////
// Minus
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Minus >
operator -( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Minus >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Abs >
operator -( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Minus >( a );
}

////
// Abs
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Abs >
abs( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Abs >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Abs >
abs( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Abs >( a );
}

////
// Sin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Sin >
sin( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Sin >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Sin >
sin( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Sin >( a );
}

////
// Cos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Cos >
cos( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Cos >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Cos >
cos( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Cos >( a );
}

////
// Tan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Tan >
tan( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Tan >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Tan >
tan( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Tan >( a );
}

////
// Sqrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Sqrt >
sqrt( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Sqrt >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Sqrt >
sqrt( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Sqrt >( a );
}

////
// Cbrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Cbrt >
cbrt( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Cbrt >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Cbrt >
cbrt( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Cbrt >( a );
}

////
// Pow
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Real >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Pow >
pow( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a, const Real& exp )
{
   auto e = Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Pow >( a );
   e.parameter.set( exp );
   return e;
}

template< typename L1,
          template< typename > class LOperation,
          typename Real >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Pow >
pow( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a, const Real& exp )
{
   auto e = Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Pow >( a );
   e.parameter.set( exp );
   return e;
}

////
// Floor
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Sin >
floor( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Floor >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Floor >
floor( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Floor >( a );
}

////
// Ceil
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Ceil >
ceil( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Ceil >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Ceil >
sin( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Ceil >( a );
}

////
// Asin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Asin >
asin( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Asin >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Asin >
asin( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Asin >( a );
}

////
// Acos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Acos >
cos( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Acos >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Acos >
acos( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Cos >( a );
}

////
// Atan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Atan >
tan( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Atan >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Atan >
atan( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Atan >( a );
}

////
// Sinh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Sinh >
sinh( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Sinh >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Sinh >
sinh( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Sinh >( a );
}

////
// Cosh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Cosh >
cosh( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Cosh >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Cosh >
cosh( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Cosh >( a );
}

////
// Tanh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Tanh >
cosh( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Tanh >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Tanh >
tanh( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Tanh >( a );
}

////
// Log
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Log >
log( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Log >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Log >
log( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Log >( a );
}

////
// Log10
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Log10 >
log10( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Log10 >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Log10 >
log10( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Log10 >( a );
}

////
// Log2
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Log2 >
log2( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Log2 >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Log2 >
log2( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Log2 >( a );
}

////
// Exp
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Exp >
exp( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Exp >( a );
}

template< typename L1,
          template< typename > class LOperation >
const Containers::Expressions::UnaryExpressionTemplate<
   Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Exp >
exp( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::UnaryExpressionTemplate<
      Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Containers::Expressions::Exp >( a );
}

////
// Vertical operations - min
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
typename Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >::RealType
min( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionMin( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
typename Containers::Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
min( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   return ExpressionMin( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
typename Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >::RealType
max( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionMax( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
typename Containers::Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
max( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   return ExpressionMax( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
typename Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >::RealType
sum( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionSum( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
typename Containers::Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
sum( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   return ExpressionSum( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Real >
typename Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >::RealType
lpNorm( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a, const Real& p )
{
   return ExpressionLpNorm( a, p );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter,
          typename Real >
typename Containers::Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
lpNorm( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >& a, const Real& p )
{
   return ExpressionLpNorm( a, p );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
typename Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >::RealType
product( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionProduct( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
typename Containers::Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
product( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   return ExpressionProduct( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
bool
logicalOr( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionLogicalOr( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
bool
logicalAnd( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   return ExpressionLogicalAnd( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
typename Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >::RealType
binaryOr( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionBinaryOr( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
typename Containers::Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
binaryAnd( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   return ExpressionBinaryAnd( a );
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
operator,( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
           const Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator,( const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
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
operator,( const Containers::Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
           const typename Containers::Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
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
operator,( const Containers::Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
           const typename Containers::Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

////
// Output stream
template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
std::ostream& operator << ( std::ostream& str, const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( expression.getSize() - 1 ) << " ]";
   return str;
}

template< typename T,
          template< typename > class Operation,
          typename Parameter >
std::ostream& operator << ( std::ostream& str, const Containers::Expressions::UnaryExpressionTemplate< T, Operation, Parameter >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( expression.getSize() - 1 ) << " ]";
   return str;
}
} // namespace TNL
