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

#include <TNL/Containers/ndarray/Indexing.h>
#include <TNL/Containers/ndarray/SizesHolder.h>
#include <TNL/Containers/ndarray/Subarrays.h>
#include <TNL/Containers/ndarray/Executors.h>
#include <TNL/Containers/ndarray/BoundaryExecutors.h>
#include <TNL/Containers/ndarray/Operations.h>

namespace TNL {
namespace Containers {

template< typename Value,
          typename Device,
          typename SizesHolder,
          typename Permutation,
          typename Base,
          typename StrideBase = __ndarray_impl::DummyStrideBase< typename SizesHolder::IndexType, SizesHolder::getDimension() > >
class NDArrayView
    : public StrideBase
{
public:
   using ValueType = Value;
   using DeviceType = Device;
   using IndexType = typename SizesHolder::IndexType;
   using SizesHolderType = SizesHolder;
   using PermutationType = Permutation;
   using ViewType = NDArrayView< ValueType, Device, SizesHolder, Permutation, Base, StrideBase >;
   using ConstViewType = NDArrayView< std::add_const_t< ValueType >, Device, SizesHolder, Permutation, Base, StrideBase >;

   static_assert( Permutation::size() == SizesHolder::getDimension(), "invalid permutation" );

   __cuda_callable__
   NDArrayView() = default;

   // explicit initialization by raw data pointer and sizes
   __cuda_callable__
   NDArrayView( Value* data, SizesHolder sizes ) : array(data), sizes(sizes) {}

   // explicit initialization by raw data pointer and sizes and strides
   __cuda_callable__
   NDArrayView( Value* data, SizesHolder sizes, StrideBase strides )
   : StrideBase(strides), array(data), sizes(sizes) {}

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
      TNL_ASSERT_EQ( sizes, other.sizes, "The sizes of the array views must be equal, views are not resizable." );
      if( getStorageSize() > 0 )
         ArrayOpsHelper< Device >::copy( array, other.array, getStorageSize() );
      return *this;
   }

   // There is no move-assignment operator, so expressions like `a = b.getView()`
   // are resolved as copy-assignment.

   // method for rebinding (reinitialization)
   __cuda_callable__
   void bind( NDArrayView view )
   {
      array = view.array;
      sizes = view.sizes;
      StrideBase::operator=( view );
   }

   __cuda_callable__
   void reset()
   {
      array = nullptr;
      sizes = SizesHolder{};
      StrideBase::operator=( StrideBase{} );
   }

   __cuda_callable__
   bool operator==( const NDArrayView& other ) const
   {
      if( sizes != other.sizes )
         return false;
      // FIXME: uninitialized data due to alignment in NDArray and padding in SlicedNDArray
      return ArrayOpsHelper< Device, Device >::compare( array, other.array, getStorageSize() );
   }

   __cuda_callable__
   bool operator!=( const NDArrayView& other ) const
   {
      if( sizes != other.sizes )
         return true;
      // FIXME: uninitialized data due to alignment in NDArray and padding in SlicedNDArray
      return ! ArrayOpsHelper< Device, Device >::compare( array, other.array, getStorageSize() );
   }

   static constexpr std::size_t getDimension()
   {
      return SizesHolder::getDimension();
   }

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
   using StrideBase::getStride;

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
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      return Base::template getStorageIndex< Permutation >( sizes,
                                                            static_cast< const StrideBase& >( *this ),
                                                            std::forward< IndexTypes >( indices )... );
   }

   template< typename... IndexTypes >
   __cuda_callable__
   ValueType&
   operator()( IndexTypes&&... indices )
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      __ndarray_impl::assertIndicesInBounds( sizes, std::forward< IndexTypes >( indices )... );
      return array[ getStorageIndex( std::forward< IndexTypes >( indices )... ) ];
   }

   template< typename... IndexTypes >
   __cuda_callable__
   const ValueType&
   operator()( IndexTypes&&... indices ) const
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      __ndarray_impl::assertIndicesInBounds( sizes, std::forward< IndexTypes >( indices )... );
      return array[ getStorageIndex( std::forward< IndexTypes >( indices )... ) ];
   }

   // bracket operator for 1D arrays
   __cuda_callable__
   ValueType&
   operator[]( IndexType&& index )
   {
      static_assert( getDimension() == 1, "the access via operator[] is provided only for 1D arrays" );
      __ndarray_impl::assertIndicesInBounds( sizes, std::forward< IndexType >( index ) );
      return array[ index ];
   }

   __cuda_callable__
   const ValueType&
   operator[]( IndexType index ) const
   {
      static_assert( getDimension() == 1, "the access via operator[] is provided only for 1D arrays" );
      __ndarray_impl::assertIndicesInBounds( sizes, std::forward< IndexType >( index ) );
      return array[ index ];
   }

   __cuda_callable__
   ViewType getView()
   {
      return ViewType( *this );
   }

   __cuda_callable__
   ConstViewType getConstView() const
   {
      return ConstViewType( array, sizes );
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
      auto subarray_sizes = Getter::filterSizes( sizes, std::forward< IndexTypes >( indices )... );
      auto strides = Getter::getStrides( sizes, std::forward< IndexTypes >( indices )... );
      static_assert( Subpermutation::size() == sizeof...(Dimensions), "Bug - wrong subpermutation length." );
      static_assert( decltype(subarray_sizes)::getDimension() == sizeof...(Dimensions), "Bug - wrong dimension of the new sizes." );
      static_assert( decltype(strides)::getDimension() == sizeof...(Dimensions), "Bug - wrong dimension of the strides." );
      using SubarrayView = NDArrayView< ValueType, Device, decltype(subarray_sizes), Subpermutation, Base, decltype(strides) >;
      return SubarrayView{ &begin, subarray_sizes, strides };
   }

   template< typename Device2 = DeviceType, typename Func >
   void forAll( Func f ) const
   {
      __ndarray_impl::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      using Begins = ConstStaticSizesHolder< IndexType, getDimension(), 0 >;
      dispatch( Begins{}, sizes, f );
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
      __ndarray_impl::SetSizesSubtractHelper< 1, Ends, SizesHolder >::subtract( ends, sizes );
      dispatch( Begins{}, ends, f );
   }

   template< typename Device2 = DeviceType, typename Func, typename Begins, typename Ends >
   void forInternal( Func f, const Begins& begins, const Ends& ends ) const
   {
      // TODO: assert "begins <= sizes", "ends <= sizes"
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
      __ndarray_impl::SetSizesSubtractHelper< 1, SkipEnds, SizesHolder >::subtract( skipEnds, sizes );

      __ndarray_impl::BoundaryExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( Begins{}, SkipBegins{}, skipEnds, sizes, f );
   }

   template< typename Device2 = DeviceType, typename Func, typename SkipBegins, typename SkipEnds >
   void forBoundary( Func f, const SkipBegins& skipBegins, const SkipEnds& skipEnds ) const
   {
      // TODO: assert "skipBegins <= sizes", "skipEnds <= sizes"
      using Begins = ConstStaticSizesHolder< IndexType, getDimension(), 0 >;
      __ndarray_impl::BoundaryExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( Begins{}, skipBegins, skipEnds, sizes, f );
   }

protected:
   Value* array = nullptr;
   SizesHolder sizes;

   // TODO: establish the concept of a "void device" for static computations in the whole TNL

   template< typename DestinationDevice, typename SourceDevice = DestinationDevice, typename _unused = void >
   struct ArrayOpsHelper
   {
      template< typename DestinationValue,
                typename SourceValue,
                typename Index >
      static void copy( DestinationValue* destination,
                        const SourceValue* source,
                        const Index size )
      {
         Algorithms::ArrayOperations< DestinationDevice, SourceDevice >::copy( destination, source, size );
      }

      template< typename Value1,
                typename Value2,
                typename Index >
      static bool compare( const Value1* destination,
                           const Value2* source,
                           const Index size )
      {
         return Algorithms::ArrayOperations< DestinationDevice, SourceDevice >::compare( destination, source, size );
      }
   };

   template< typename _unused >
   struct ArrayOpsHelper< void, void, _unused >
   {
      template< typename DestinationValue,
                typename SourceValue,
                typename Index >
      __cuda_callable__
      static void copy( DestinationValue* destination,
                        const SourceValue* source,
                        const Index size )
      {
         for( Index i = 0; i < size; i ++ )
            destination[ i ] = source[ i ];
      }

      template< typename Value1,
                typename Value2,
                typename Index >
      __cuda_callable__
      static bool compare( const Value1* destination,
                           const Value2* source,
                           const Index size )
      {
         for( Index i = 0; i < size; i++ )
            if( ! ( destination[ i ] == source[ i ] ) )
               return false;
         return true;
      }
   };
};

} // namespace Containers
} // namespace TNL
