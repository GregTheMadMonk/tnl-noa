/***************************************************************************
                          NDArrayView.h  -  description
                             -------------------
    begin                : Dec 24, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Containers/NDArrayIndexer.h>
#include <TNL/Containers/ndarray/SizesHolder.h>
#include <TNL/Containers/ndarray/Subarrays.h>
#include <TNL/Containers/ndarray/Executors.h>
#include <TNL/Containers/ndarray/BoundaryExecutors.h>
#include <TNL/Containers/ndarray/Operations.h>
#include <TNL/Containers/Algorithms/ArrayOperations.h>

namespace TNL {
namespace Containers {

template< typename Value,
          typename Device,
          typename SizesHolder,
          typename Permutation,
          typename Base,
          typename StridesHolder = __ndarray_impl::DummyStrideBase< typename SizesHolder::IndexType, SizesHolder::getDimension() > >
class NDArrayView
    : public NDArrayIndexer< SizesHolder, Permutation, Base, StridesHolder >
{
public:
   using ValueType = Value;
   using DeviceType = Device;
   using IndexType = typename SizesHolder::IndexType;
   using SizesHolderType = SizesHolder;
   using PermutationType = Permutation;
   using IndexerType = NDArrayIndexer< SizesHolder, Permutation, Base, StridesHolder >;
   using ViewType = NDArrayView< Value, Device, SizesHolder, Permutation, Base, StridesHolder >;
   using ConstViewType = NDArrayView< std::add_const_t< Value >, Device, SizesHolder, Permutation, Base, StridesHolder >;

   static_assert( Permutation::size() == SizesHolder::getDimension(), "invalid permutation" );

   __cuda_callable__
   NDArrayView() = default;

   // explicit initialization by raw data pointer and sizes and strides
   __cuda_callable__
   NDArrayView( Value* data, SizesHolder sizes, StridesHolder strides = StridesHolder{} )
   : IndexerType(sizes, strides), array(data) {}

   // explicit initialization by raw data pointer and indexer
   __cuda_callable__
   NDArrayView( Value* data, IndexerType indexer )
   : IndexerType(indexer), array(data) {}

   // Copy-constructor does shallow copy, so views can be passed-by-value into
   // CUDA kernels and they can be captured-by-value in __cuda_callable__
   // lambda functions.
   __cuda_callable__
   NDArrayView( const NDArrayView& ) = default;

   // default move-constructor
   __cuda_callable__
   NDArrayView( NDArrayView&& ) = default;

   // Copy-assignment does deep copy, just like regular array, but the sizes
   // must match (i.e. copy-assignment cannot resize).
   __cuda_callable__
   NDArrayView& operator=( const NDArrayView& other )
   {
      TNL_ASSERT_EQ( getSizes(), other.getSizes(), "The sizes of the array views must be equal, views are not resizable." );
      if( getStorageSize() > 0 )
         Algorithms::ArrayOperations< DeviceType >::copy( array, other.array, getStorageSize() );
      return *this;
   }

   // Templated copy-assignment
   template< typename OtherView >
   NDArrayView& operator=( const OtherView& other )
   {
      static_assert( std::is_same< PermutationType, typename OtherView::PermutationType >::value,
                     "Arrays must have the same permutation of indices." );
      static_assert( NDArrayView::isContiguous() && OtherView::isContiguous(),
                     "Non-contiguous array views cannot be assigned." );
      TNL_ASSERT_TRUE( __ndarray_impl::sizesWeakCompare( getSizes(), other.getSizes() ),
                       "The sizes of the array views must be equal, views are not resizable." );
      if( getStorageSize() > 0 ) {
         TNL_ASSERT_TRUE( array, "Attempted to assign to an empty view." );
         Algorithms::ArrayOperations< DeviceType, typename OtherView::DeviceType >::copy( array, other.getData(), getStorageSize() );
      }
      return *this;
   }

   // There is no move-assignment operator, so expressions like `a = b.getView()`
   // are resolved as copy-assignment.

   // method for rebinding (reinitialization)
   __cuda_callable__
   void bind( NDArrayView view )
   {
      IndexerType::operator=( view );
      array = view.array;
   }

   __cuda_callable__
   void reset()
   {
      IndexerType::operator=( IndexerType{} );
      array = nullptr;
   }

   __cuda_callable__
   bool operator==( const NDArrayView& other ) const
   {
      if( getSizes() != other.getSizes() )
         return false;
      // FIXME: uninitialized data due to alignment in NDArray and padding in SlicedNDArray
      return Algorithms::ArrayOperations< Device, Device >::compare( array, other.array, getStorageSize() );
   }

   __cuda_callable__
   bool operator!=( const NDArrayView& other ) const
   {
      if( getSizes() != other.getSizes() )
         return true;
      // FIXME: uninitialized data due to alignment in NDArray and padding in SlicedNDArray
      return ! Algorithms::ArrayOperations< Device, Device >::compare( array, other.array, getStorageSize() );
   }

   __cuda_callable__
   ValueType* getData()
   {
      return array;
   }

   __cuda_callable__
   std::add_const_t< ValueType >* getData() const
   {
      return array;
   }

   // methods from the base class
   using IndexerType::getDimension;
   using IndexerType::getSizes;
   using IndexerType::getSize;
   using IndexerType::getStride;
   using IndexerType::getStorageSize;
   using IndexerType::getStorageIndex;

   __cuda_callable__
   const IndexerType& getIndexer() const
   {
      return *this;
   }

   __cuda_callable__
   ViewType getView()
   {
      return ViewType( *this );
   }

   __cuda_callable__
   ConstViewType getConstView() const
   {
      return ConstViewType( array, getSizes(), static_cast< const StridesHolder& >( *this ) );
   }

   template< std::size_t... Dimensions, typename... IndexTypes >
   __cuda_callable__
   auto getSubarrayView( IndexTypes&&... indices )
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      static_assert( 0 < sizeof...(Dimensions) && sizeof...(Dimensions) <= getDimension(), "got wrong number of dimensions" );
      static_assert( __ndarray_impl::all_elements_in_range( 0, Permutation::size(), {Dimensions...} ),
                     "invalid dimensions" );
// FIXME: nvcc chokes on the variadic brace-initialization
#ifndef __NVCC__
      static_assert( __ndarray_impl::is_increasing_sequence( {Dimensions...} ),
                     "specifying permuted dimensions is not supported" );
#endif

      using Getter = __ndarray_impl::SubarrayGetter< Base, Permutation, Dimensions... >;
      using Subpermutation = typename Getter::Subpermutation;
      auto& begin = operator()( std::forward< IndexTypes >( indices )... );
      auto subarray_sizes = Getter::filterSizes( getSizes(), std::forward< IndexTypes >( indices )... );
      auto strides = Getter::getStrides( getSizes(), std::forward< IndexTypes >( indices )... );
      static_assert( Subpermutation::size() == sizeof...(Dimensions), "Bug - wrong subpermutation length." );
      static_assert( decltype(subarray_sizes)::getDimension() == sizeof...(Dimensions), "Bug - wrong dimension of the new sizes." );
      static_assert( decltype(strides)::getDimension() == sizeof...(Dimensions), "Bug - wrong dimension of the strides." );
      using SubarrayView = NDArrayView< ValueType, Device, decltype(subarray_sizes), Subpermutation, Base, decltype(strides) >;
      return SubarrayView{ &begin, subarray_sizes, strides };
   }

   template< typename... IndexTypes >
   __cuda_callable__
   ValueType&
   operator()( IndexTypes&&... indices )
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      __ndarray_impl::assertIndicesInBounds( getSizes(), std::forward< IndexTypes >( indices )... );
      return array[ getStorageIndex( std::forward< IndexTypes >( indices )... ) ];
   }

   template< typename... IndexTypes >
   __cuda_callable__
   const ValueType&
   operator()( IndexTypes&&... indices ) const
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      __ndarray_impl::assertIndicesInBounds( getSizes(), std::forward< IndexTypes >( indices )... );
      return array[ getStorageIndex( std::forward< IndexTypes >( indices )... ) ];
   }

   // bracket operator for 1D arrays
   __cuda_callable__
   ValueType&
   operator[]( IndexType&& index )
   {
      static_assert( getDimension() == 1, "the access via operator[] is provided only for 1D arrays" );
      __ndarray_impl::assertIndicesInBounds( getSizes(), std::forward< IndexType >( index ) );
      return array[ index ];
   }

   __cuda_callable__
   const ValueType&
   operator[]( IndexType index ) const
   {
      static_assert( getDimension() == 1, "the access via operator[] is provided only for 1D arrays" );
      __ndarray_impl::assertIndicesInBounds( getSizes(), std::forward< IndexType >( index ) );
      return array[ index ];
   }

   template< typename Device2 = DeviceType, typename Func >
   void forAll( Func f ) const
   {
      __ndarray_impl::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      using Begins = ConstStaticSizesHolder< IndexType, getDimension(), 0 >;
      dispatch( Begins{}, getSizes(), f );
   }

   template< typename Device2 = DeviceType, typename Func >
   void forInternal( Func f ) const
   {
      __ndarray_impl::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      using Begins = ConstStaticSizesHolder< IndexType, getDimension(), 1 >;
      // subtract static sizes
      using Ends = typename __ndarray_impl::SubtractedSizesHolder< SizesHolder, 1 >::type;
      // subtract dynamic sizes
      Ends ends;
      __ndarray_impl::SetSizesSubtractHelper< 1, Ends, SizesHolder >::subtract( ends, getSizes() );
      dispatch( Begins{}, ends, f );
   }

   template< typename Device2 = DeviceType, typename Func, typename Begins, typename Ends >
   void forInternal( Func f, const Begins& begins, const Ends& ends ) const
   {
      // TODO: assert "begins <= getSizes()", "ends <= getSizes()"
      __ndarray_impl::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( begins, ends, f );
   }

   template< typename Device2 = DeviceType, typename Func >
   void forBoundary( Func f ) const
   {
      using Begins = ConstStaticSizesHolder< IndexType, getDimension(), 0 >;
      using SkipBegins = ConstStaticSizesHolder< IndexType, getDimension(), 1 >;
      // subtract static sizes
      using SkipEnds = typename __ndarray_impl::SubtractedSizesHolder< SizesHolder, 1 >::type;
      // subtract dynamic sizes
      SkipEnds skipEnds;
      __ndarray_impl::SetSizesSubtractHelper< 1, SkipEnds, SizesHolder >::subtract( skipEnds, getSizes() );

      __ndarray_impl::BoundaryExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( Begins{}, SkipBegins{}, skipEnds, getSizes(), f );
   }

   template< typename Device2 = DeviceType, typename Func, typename SkipBegins, typename SkipEnds >
   void forBoundary( Func f, const SkipBegins& skipBegins, const SkipEnds& skipEnds ) const
   {
      // TODO: assert "skipBegins <= getSizes()", "skipEnds <= getSizes()"
      using Begins = ConstStaticSizesHolder< IndexType, getDimension(), 0 >;
      __ndarray_impl::BoundaryExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( Begins{}, skipBegins, skipEnds, getSizes(), f );
   }

protected:
   Value* array = nullptr;
   IndexerType indexer;
};

} // namespace Containers
} // namespace TNL
