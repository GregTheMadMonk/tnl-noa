/***************************************************************************
                          NDArrayIndexer.h  -  description
                             -------------------
    begin                : Apr 14, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Containers/ndarray/Indexing.h>
#include <TNL/Containers/ndarray/SizesHolderHelpers.h>   // StorageSizeGetter
#include <TNL/Containers/ndarray/Subarrays.h>   // DummyStrideBase

namespace TNL {
namespace Containers {

template< typename SizesHolder,
          typename Permutation,
          typename Base,
          typename StridesHolder = __ndarray_impl::DummyStrideBase< typename SizesHolder::IndexType, SizesHolder::getDimension() >,
          typename Overlaps = __ndarray_impl::make_constant_index_sequence< SizesHolder::getDimension(), 0 > >
class NDArrayIndexer
    : public StridesHolder
{
public:
   using IndexType = typename SizesHolder::IndexType;
   using NDBaseType = Base;
   using SizesHolderType = SizesHolder;
   using StridesHolderType = StridesHolder;
   using PermutationType = Permutation;
   using OverlapsType = Overlaps;

   static_assert( StridesHolder::getDimension() == SizesHolder::getDimension(),
                  "Dimension of strides does not match the dimension of sizes." );
   static_assert( Permutation::size() == SizesHolder::getDimension(),
                  "Dimension of permutation does not match the dimension of sizes." );
   static_assert( Overlaps::size() == SizesHolder::getDimension(),
                  "Dimension of overlaps does not match the dimension of sizes." );

   __cuda_callable__
   NDArrayIndexer() = default;

   // explicit initialization by sizes and strides
   __cuda_callable__
   NDArrayIndexer( SizesHolder sizes, StridesHolder strides )
   : StridesHolder(strides), sizes(sizes) {}

   static constexpr std::size_t getDimension()
   {
      return SizesHolder::getDimension();
   }

   __cuda_callable__
   const SizesHolderType& getSizes() const
   {
      return sizes;
   }

   template< std::size_t level >
   __cuda_callable__
   IndexType getSize() const
   {
      return sizes.template getSize< level >();
   }

   // method template from base class
   using StridesHolder::getStride;

   template< std::size_t level >
   static constexpr std::size_t getOverlap()
   {
      return __ndarray_impl::get< level >( Overlaps{} );
   }

   // returns the product of the aligned sizes
   __cuda_callable__
   IndexType getStorageSize() const
   {
      using Alignment = typename Base::template Alignment< Permutation >;
      return __ndarray_impl::StorageSizeGetter< SizesHolder, Alignment, Overlaps >::get( sizes );
   }

   template< typename... IndexTypes >
   __cuda_callable__
   IndexType
   getStorageIndex( IndexTypes&&... indices ) const
   {
      static_assert( sizeof...( indices ) == SizesHolder::getDimension(), "got wrong number of indices" );
      return Base::template getStorageIndex< Permutation, Overlaps >
             ( sizes,
               static_cast< const StridesHolder& >( *this ),
               std::forward< IndexTypes >( indices )... );
   }

protected:
   // non-const reference accessor cannot be public - only subclasses like NDArrayStorage may modify the sizes
   __cuda_callable__
   SizesHolderType& getSizes()
   {
      return sizes;
   }

   SizesHolder sizes;
};

} // namespace Containers
} // namespace TNL
