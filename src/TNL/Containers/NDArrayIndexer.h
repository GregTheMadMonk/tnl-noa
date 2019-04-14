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
          typename StridesHolder = __ndarray_impl::DummyStrideBase< typename SizesHolder::IndexType, SizesHolder::getDimension() > >
class NDArrayIndexer
    : public StridesHolder
{
public:
   using IndexType = typename SizesHolder::IndexType;
   using SizesHolderType = SizesHolder;
   using PermutationType = Permutation;

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

   // returns the product of the aligned sizes
   __cuda_callable__
   IndexType getStorageSize() const
   {
      using Alignment = typename Base::template Alignment< Permutation >;
      return __ndarray_impl::StorageSizeGetter< SizesHolder, Alignment >::get( sizes );
   }

   template< typename... IndexTypes >
   __cuda_callable__
   IndexType
   getStorageIndex( IndexTypes&&... indices ) const
   {
      static_assert( sizeof...( indices ) == SizesHolder::getDimension(), "got wrong number of indices" );
      return Base::template getStorageIndex< Permutation >( sizes,
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
