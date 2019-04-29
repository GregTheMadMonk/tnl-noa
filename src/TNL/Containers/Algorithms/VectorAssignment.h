/***************************************************************************
                          VectorAssignment.h  -  description
                             -------------------
    begin                : Apr 4, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include<type_traits>
#include<utility>
#include <TNL/Containers/Algorithms/VectorOperations.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

namespace Details {
/**
 * SFINAE for checking if T has getSize method
 * TODO: We should better test operator[] but we need to know the indexing type.
 */
template< typename T >
class HasSubscriptOperator
{
private:
    typedef char YesType[1];
    typedef char NoType[2];

    template< typename C > static YesType& test(  decltype(std::declval< C >().getSize() ) );
    template< typename C > static NoType& test(...);

public:
    static constexpr bool value = ( sizeof( test< T >(0) ) == sizeof( YesType ) );
};
} // namespace Details

template< typename Vector,
          typename T,
          bool hasSubscriptOperator = Details::HasSubscriptOperator< T >::value >
struct VectorAssignment{};

/**
 * \brief Specialization for assignment with subscript operator
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
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i ++ )
         v[ i ] = t[ i ];
   };

   static void assign( Vector& v, const T& t )
   {
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto ass = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] = t[ i ];
      };
      ParallelFor< DeviceType >::exec( ( IndexType  ) 0, v.getSize(), ass );
   };

};

/**
 * \brief Specialization for array-value assignment for other types. We assume
 * that T is convertible to Vector::ValueType.
 */
template< typename Vector,
          typename T >
struct VectorAssignment< Vector, T, false >
{
   static void resize( Vector& v, const T& t )
   {
   };

   __cuda_callable__
   static void assignStatic( Vector& v, const T& t )
   {
      TNL_ASSERT_GT( v.getSize(), 0, "Cannot assign value to empty vector." );
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i ++ )
         v[ i ] = t;
   };

   static void assign( Vector& v, const T& t )
   {
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto ass = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] = t;
      };
      ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), ass );
   }

};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
