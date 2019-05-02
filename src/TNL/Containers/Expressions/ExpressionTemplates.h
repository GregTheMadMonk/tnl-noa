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

   RealType getElement( const int i ) const
   {
       return Operation< typename T1::RealType, typename T2::RealType >::evaluate( op1.getElement( i ), op2.getElement( i ) );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
       return Operation< typename T1::RealType, typename T2::RealType >::evaluate( op1[ i ], op2[ i ] );
   }

   __cuda_callable__
   int getSize() const
   {
       return op1.getSize();
   }

   protected:
      typename OperandType< T1, DeviceType >::type op1;
      typename OperandType< T2, DeviceType >::type op2;
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

   RealType getElement( const int i ) const
   {
       return Operation< typename T1::RealType, typename T2::RealType >::evaluate( op1.getElement( i ), op2 );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
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

   RealType getElement( const int i ) const
   {
       return Operation< typename T1::RealType, typename T2::RealType >::evaluate( op1, op2.getElement( i ) );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
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

   RealType getElement( const int i ) const
   {
       return Operation< typename T1::RealType >::evaluate( operand.getElement( i ), parameter );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
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

   RealType getElement( const int i ) const
   {
       return Operation< typename T1::RealType >::evaluate( operand.getElement( i ) );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
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

////
// Binary expressions addition
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Addition >
operator + ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Expressions::Addition >
operator + ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Expressions::Addition >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Addition >
operator + ( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
             const typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::Addition >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   typename Expressions::UnaryExpressionTemplate< R1, ROperation >,
   Expressions::Addition >
operator + ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      typename Expressions::UnaryExpressionTemplate< R1, ROperation >,
      Expressions::Addition >( a, b );
}

////
// Binary expression subtraction
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Subtraction >
operator - ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Expressions::Subtraction >
operator - ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Expressions::Subtraction >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Subtraction >
operator - ( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
             const typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::Subtraction >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   typename Expressions::UnaryExpressionTemplate< R1, ROperation >,
   Expressions::Subtraction >
operator - ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      typename Expressions::UnaryExpressionTemplate< R1, ROperation >,
      Expressions::Subtraction >( a, b );
}

////
// Binary expression multiplication
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Multiplication >
operator * ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Expressions::Multiplication >
operator * ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Expressions::Multiplication >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Multiplication >
operator * ( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
             const typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::Multiplication >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   typename Expressions::UnaryExpressionTemplate< R1, ROperation >,
   Expressions::Multiplication >
operator * ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Expressions::UnaryExpressionTemplate< R1, ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      typename Expressions::UnaryExpressionTemplate< R1, ROperation >,
      Expressions::Multiplication >( a, b );
}

////
// Binary expression division
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Division >
operator / ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Expressions::Division >
operator / ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Expressions::Division >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Division >
operator / ( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
             const typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::Division >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   typename Expressions::UnaryExpressionTemplate< R1, ROperation >,
   Expressions::Division >
operator / ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      typename Expressions::UnaryExpressionTemplate< R1, ROperation >,
      Expressions::Division >( a, b );
}

////
// Binary expression min
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Min >
min ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
      const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Expressions::Min >
min( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Expressions::Min >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Min >
min( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
     const typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::Min >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   typename Expressions::UnaryExpressionTemplate< R1, ROperation >,
   Expressions::Min >
min( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
     const typename Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      typename Expressions::UnaryExpressionTemplate< R1, ROperation >,
      Expressions::Min >( a, b );
}

////
// Binary expression max
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Max >
max( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
   typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Expressions::Max >
max( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< T1, T2, Operation >,
      typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Expressions::Max >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Max >
max( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
     const typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::Max >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
const Expressions::BinaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   typename Expressions::UnaryExpressionTemplate< R1, ROperation >,
   Expressions::Max >
max( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
     const typename Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Expressions::BinaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      typename Expressions::UnaryExpressionTemplate< R1, ROperation >,
      Expressions::Max >( a, b );
}

////
// Comparison operator ==
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator == ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::ComparisonEQ( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator == ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::ComparisonEQ( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator == ( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
              const typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::ComparisonEQ( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator == ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Expressions::UnaryExpressionTemplate< R1,ROperation >& b )
{
   return Expressions::ComparisonEQ( a, b );
}

////
// Comparison operator !=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator != ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::ComparisonNE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator != ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::ComparisonNE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator != ( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
              const typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::ComparisonNE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator != ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
              const typename Expressions::UnaryExpressionTemplate< R1, ROperation >& b )
{
   return Expressions::ComparisonNE( a, b );
}

////
// Comparison operator <
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator < ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::ComparisonLT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator < ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::ComparisonLT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator < ( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
              const typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::ComparisonLT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator < ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
              const typename Expressions::UnaryExpressionTemplate< R1, ROperation >& b )
{
   return Expressions::ComparisonLT( a, b );
}

////
// Comparison operator <=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator <= ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::ComparisonLE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator <= ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::ComparisonLE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator <= ( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
              const typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::ComparisonLE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator <= ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
              const typename Expressions::UnaryExpressionTemplate< R1, ROperation >& b )
{
   return Expressions::ComparisonLE( a, b );
}

////
// Comparison operator >
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator > ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::ComparisonGT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator > ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::ComparisonGT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator > ( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
             const typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::ComparisonGT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator > ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Expressions::UnaryExpressionTemplate< R1, ROperation >& b )
{
   return Expressions::ComparisonGT( a, b );
}

////
// Comparison operator >=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator >= ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
              const Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::ComparisonGE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator >= ( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Expressions::BinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::ComparisonGE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator >= ( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a,
              const typename Expressions::BinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::ComparisonGE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator >= ( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a,
              const typename Expressions::UnaryExpressionTemplate< R1, ROperation >& b )
{
   return Expressions::ComparisonGE( a, b );
}

////
// Unary operations


////
// Minus
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Minus >
operator -( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Minus >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Abs >
operator -( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Minus >( a );
}

////
// Abs
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Abs >
abs( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Abs >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Abs >
abs( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Abs >( a );
}

////
// Sin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Sin >
sin( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Sin >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Sin >
sin( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Sin >( a );
}

////
// Cos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Cos >
cos( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Cos >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Cos >
cos( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Cos >( a );
}

////
// Tan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Tan >
tan( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Tan >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Tan >
tan( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Tan >( a );
}

////
// Sqrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Sqrt >
sqrt( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Sqrt >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Sqrt >
sqrt( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Sqrt >( a );
}

////
// Cbrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Cbrt >
cbrt( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Cbrt >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Cbrt >
cbrt( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Cbrt >( a );
}

////
// Pow
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Pow >
pow( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a, const Real& exp )
{
   auto e = Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Pow >( a );
   e.parameter.set( exp );
   return e;
}

template< typename L1,
          template< typename > class LOperation,
          typename Real >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Pow >
pow( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a, const Real& exp )
{
   auto e = Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Pow >( a );
   e.parameter.set( exp );
   return e;
}

////
// Floor
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Sin >
floor( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Floor >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Floor >
floor( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Floor >( a );
}

////
// Ceil
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Ceil >
ceil( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Ceil >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Ceil >
sin( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Ceil >( a );
}

////
// Asin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Asin >
asin( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Asin >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Asin >
asin( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Asin >( a );
}

////
// Acos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Acos >
cos( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Acos >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Acos >
acos( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Cos >( a );
}

////
// Atan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Atan >
tan( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Atan >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Atan >
atan( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Atan >( a );
}

////
// Sinh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Sinh >
sinh( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Sinh >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Sinh >
sinh( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Sinh >( a );
}

////
// Cosh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Cosh >
cosh( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Cosh >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Cosh >
cosh( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Cosh >( a );
}

////
// Tanh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Tanh >
cosh( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Tanh >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Tanh >
tanh( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Tanh >( a );
}

////
// Log
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Log >
log( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Log >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Log >
log( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Log >( a );
}

////
// Log10
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Log10 >
log10( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Log10 >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Log10 >
log10( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Log10 >( a );
}

////
// Log2
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Log2 >
log2( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Log2 >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Log2 >
log2( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Log2 >( a );
}

////
// Exp
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Exp >
exp( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::BinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Exp >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::UnaryExpressionTemplate<
   Expressions::UnaryExpressionTemplate< L1, LOperation >,
   Expressions::Exp >
exp( const Expressions::UnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::UnaryExpressionTemplate<
      Expressions::UnaryExpressionTemplate< L1, LOperation >,
      Expressions::Exp >( a );
}

////
// Vertical operations - min
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
typename Expressions::BinaryExpressionTemplate< L1, L2, LOperation >::RealType
min( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionMin( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
__cuda_callable__
typename Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
min( const Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   return ExpressionMin( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
typename Expressions::BinaryExpressionTemplate< L1, L2, LOperation >::RealType
max( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionMax( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
__cuda_callable__
typename Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
max( const Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   return ExpressionMax( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
typename Expressions::BinaryExpressionTemplate< L1, L2, LOperation >::RealType
sum( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionSum( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
__cuda_callable__
typename Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
sum( const Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   return ExpressionSum( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
typename Expressions::BinaryExpressionTemplate< L1, L2, LOperation >::RealType
product( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionProduct( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
__cuda_callable__
typename Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
product( const Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   return ExpressionProduct( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
typename Expressions::BinaryExpressionTemplate< L1, L2, LOperation >::RealType
logicalOr( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionLogicalOr( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
__cuda_callable__
typename Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
logicalOr( const Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   return ExpressionLogicalOr( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
typename Expressions::BinaryExpressionTemplate< L1, L2, LOperation >::RealType
binaryOr( const Expressions::BinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return ExpressionBinaryOr( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter >
__cuda_callable__
typename Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
binaryOr( const Expressions::UnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   return ExpressionBinaryOr( a );
}

////
// Output stream
template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
std::ostream& operator << ( std::ostream& str, const BinaryExpressionTemplate< T1, T2, Operation >& expression )
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
std::ostream& operator << ( std::ostream& str, const UnaryExpressionTemplate< T, Operation, Parameter >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( expression.getSize() - 1 ) << " ]";
   return str;
}
      } //namespace Expressions
   } //namespace Containers
} // namespace TNL
