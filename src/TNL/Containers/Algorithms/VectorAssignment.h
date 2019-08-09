/***************************************************************************
                          VectorAssignment.h  -  description
                             -------------------
    begin                : Apr 4, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/TypeTraits.h>
#include <TNL/ParallelFor.h>
#include <TNL/Containers/Algorithms/VectorOperations.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

/**
 * \brief Vector assignment
 */
template< typename Vector,
          typename T,
          bool hasSubscriptOperator = HasSubscriptOperator< T >::value >
struct VectorAssignment;

/**
 * \brief Vector assignment with an operation: +=, -=, *=, /=
 */
template< typename Vector,
          typename T,
          bool hasSubscriptOperator = HasSubscriptOperator< T >::value,
          bool hasSetSizeMethod = HasSetSizeMethod< T >::value >
struct VectorAssignmentWithOperation;

/**
 * \brief Specialization of ASSIGNEMENT with subscript operator
 */
template< typename Vector,
          typename T >
struct VectorAssignment< Vector, T, true >
{
   static void resize( Vector& v, const T& t )
   {
      v.setSize( t.getSize() );
   }

   __cuda_callable__
   static void assignStatic( Vector& v, const T& t )
   {
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] = t[ i ];
   }

   static void assign( Vector& v, const T& t )
   {
      static_assert( std::is_same< typename Vector::DeviceType, typename T::DeviceType >::value,
                     "Cannot assign an expression to a vector allocated on a different device." );
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto assignment = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] = t[ i ];
      };
      ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), assignment );
   }
};

/**
 * \brief Specialization of ASSIGNEMENT for array-value assignment for other types. We assume
 * that T is convertible to Vector::ValueType.
 */
template< typename Vector,
          typename T >
struct VectorAssignment< Vector, T, false >
{
   static void resize( Vector& v, const T& t )
   {
   }

   __cuda_callable__
   static void assignStatic( Vector& v, const T& t )
   {
      TNL_ASSERT_GT( v.getSize(), 0, "Cannot assign value to empty vector." );
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] = t;
   }

   static void assign( Vector& v, const T& t )
   {
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto assignment = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] = t;
      };
      ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), assignment );
   }
};

/**
 * \brief Specialization for types with subscript operator and setSize
 *        method - i.e. for Vectors.
 *
 * This is necessary, because Vectors cannot be passed-by-value to CUDA kernels
 * so we have to do it with views.
 */
template< typename Vector,
          typename T >
struct VectorAssignmentWithOperation< Vector, T, true, true >
{
   static void addition( Vector& v, const T& t )
   {
      VectorAssignmentWithOperation< Vector, typename Vector::ConstViewType >::addition( v, t.getConstView() );
   }

   static void subtraction( Vector& v, const T& t )
   {
      VectorAssignmentWithOperation< Vector, typename Vector::ConstViewType >::subtraction( v, t.getConstView() );
   }

   static void multiplication( Vector& v, const T& t )
   {
      VectorAssignmentWithOperation< Vector, typename Vector::ConstViewType >::multiplication( v, t.getConstView() );
   }

   static void division( Vector& v, const T& t )
   {
      VectorAssignmentWithOperation< Vector, typename Vector::ConstViewType >::subtraction( v, t.getConstView() );
   }
};

/**
 * \brief Specialization for types with subscript operator, but without setSize
 *        method - i.e. for expressions, views and static vectors.
 */
template< typename Vector,
          typename T >
struct VectorAssignmentWithOperation< Vector, T, true, false >
{
   __cuda_callable__
   static void additionStatic( Vector& v, const T& t )
   {
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] += t[ i ];
   }

   static void addition( Vector& v, const T& t )
   {
      static_assert( std::is_same< typename Vector::DeviceType, typename T::DeviceType >::value,
                     "Cannot assign an expression to a vector allocated on a different device." );
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto add = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] += t[ i ];
      };
      ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), add );
   }

   __cuda_callable__
   static void subtractionStatic( Vector& v, const T& t )
   {
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] -= t[ i ];
   }

   static void subtraction( Vector& v, const T& t )
   {
      static_assert( std::is_same< typename Vector::DeviceType, typename T::DeviceType >::value,
                     "Cannot assign an expression to a vector allocated on a different device." );
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto subtract = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] -= t[ i ];
      };
      ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), subtract );
   }

   __cuda_callable__
   static void multiplicationStatic( Vector& v, const T& t )
   {
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] *= t[ i ];
   }

   static void multiplication( Vector& v, const T& t )
   {
      static_assert( std::is_same< typename Vector::DeviceType, typename T::DeviceType >::value,
                     "Cannot assign an expression to a vector allocated on a different device." );
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto multiply = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] *= t[ i ];
      };
      ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), multiply );
   }

   __cuda_callable__
   static void divisionStatic( Vector& v, const T& t )
   {
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] /= t[ i ];
   }

   static void division( Vector& v, const T& t )
   {
      static_assert( std::is_same< typename Vector::DeviceType, typename T::DeviceType >::value,
                     "Cannot assign an expression to a vector allocated on a different device." );
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto divide = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] /= t[ i ];
      };
      ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), divide );
   }
};

/**
 * \brief Specialization for array-value assignment for other types. We assume
 * that T is convertible to Vector::ValueType.
 */
template< typename Vector,
          typename T >
struct VectorAssignmentWithOperation< Vector, T, false, false >
{
   __cuda_callable__
   static void additionStatic( Vector& v, const T& t )
   {
      TNL_ASSERT_GT( v.getSize(), 0, "Cannot assign value to empty vector." );
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] += t;
   }

   static void addition( Vector& v, const T& t )
   {
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto add = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] += t;
      };
      ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), add );
   }

   __cuda_callable__
   static void subtractionStatic( Vector& v, const T& t )
   {
      TNL_ASSERT_GT( v.getSize(), 0, "Cannot assign value to empty vector." );
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] -= t;
   }

   static void subtraction( Vector& v, const T& t )
   {
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto subtract = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] -= t;
      };
      ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), subtract );
   }

   __cuda_callable__
   static void multiplicationStatic( Vector& v, const T& t )
   {
      TNL_ASSERT_GT( v.getSize(), 0, "Cannot assign value to empty vector." );
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] *= t;
   }

   static void multiplication( Vector& v, const T& t )
   {
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto multiply = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] *= t;
      };
      ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), multiply );
   }

   __cuda_callable__
   static void divisionStatic( Vector& v, const T& t )
   {
      TNL_ASSERT_GT( v.getSize(), 0, "Cannot assign value to empty vector." );
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] /= t;
   }

   static void division( Vector& v, const T& t )
   {
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto divide = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] /= t;
      };
      ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), divide );
   }
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
