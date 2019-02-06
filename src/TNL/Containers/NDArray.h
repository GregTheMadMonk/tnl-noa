/***************************************************************************
                          NDArray.h  -  description
                             -------------------
    begin                : Dec 24, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Containers/Array.h>
#include <TNL/Containers/StaticArray.h>

#include <TNL/Containers/NDArrayView.h>

namespace TNL {
namespace Containers {

template< std::size_t slicedDimension = 0,
          std::size_t sliceSize = 0 >
struct SliceInfo
{
   // sliceSize == 0 means no slicing
   static constexpr std::size_t getSliceSize( std::size_t dimension )
   {
      return (dimension == slicedDimension) ? sliceSize : 0;
   }
};




template< typename Array,
          typename SizesHolder,
          typename Permutation,
          typename Base,
          typename Device = typename Array::DeviceType >
class NDArrayStorage
{
public:
   using StorageArray = Array;
   using ValueType = typename Array::ValueType;
   using DeviceType = Device;
   using IndexType = typename Array::IndexType;
   using SizesHolderType = SizesHolder;
   using PermutationType = Permutation;
   using ViewType = NDArrayView< ValueType, DeviceType, SizesHolder, Permutation, Base >;
   using ConstViewType = NDArrayView< std::add_const_t< ValueType >, DeviceType, SizesHolder, Permutation, Base >;

   static_assert( Permutation::size() == SizesHolder::getDimension(), "invalid permutation" );

   // all methods from NDArrayView

   NDArrayStorage() = default;

   // The copy-constructor of TNL::Containers::Array makes shallow copy so our
   // copy-constructor cannot be default. Actually, we most likely don't need
   // it anyway, so let's just delete it.
   NDArrayStorage( const NDArrayStorage& ) = delete;

   // Standard copy-semantics with deep copy, just like regular 1D array.
   // Mismatched sizes cause reallocations.
   NDArrayStorage& operator=( const NDArrayStorage& other ) = default;

   // default move-semantics
   NDArrayStorage( NDArrayStorage&& ) = default;
   NDArrayStorage& operator=( NDArrayStorage&& ) = default;

   bool operator==( const NDArrayStorage& other ) const
   {
      // FIXME: uninitialized data due to alignment in NDArray and padding in SlicedNDArray
      return sizes == other.sizes && array == other.array;
   }

   bool operator!=( const NDArrayStorage& other ) const
   {
      // FIXME: uninitialized data due to alignment in NDArray and padding in SlicedNDArray
      return sizes != other.sizes || array != other.array;
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
                                                            StrideBase{},
                                                            std::forward< IndexTypes >( indices )... );
   }

   template< typename... IndexTypes >
   __cuda_callable__
   ValueType&
   operator()( IndexTypes&&... indices )
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      __ndarray_impl::assertIndicesInBounds( sizes, std::forward< IndexTypes >( indices )... );
      TNL_ASSERT_LT( getStorageIndex( std::forward< IndexTypes >( indices )... ), getStorageSize(),
                     "storage index out of bounds - either input error or a bug in the indexer" );
      return array[ getStorageIndex( std::forward< IndexTypes >( indices )... ) ];
   }

   template< typename... IndexTypes >
   __cuda_callable__
   const ValueType&
   operator()( IndexTypes&&... indices ) const
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      __ndarray_impl::assertIndicesInBounds( sizes, std::forward< IndexTypes >( indices )... );
      TNL_ASSERT_LT( getStorageIndex( std::forward< IndexTypes >( indices )... ), getStorageSize(),
                     "storage index out of bounds - either input error or a bug in the indexer" );
      return array[ getStorageIndex( std::forward< IndexTypes >( indices )... ) ];
   }

   // bracket operator for 1D arrays
   __cuda_callable__
   ValueType&
   operator[]( IndexType index )
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
      return ViewType( array.getData(), sizes );
   }

   __cuda_callable__
   ConstViewType getConstView() const
   {
      return ConstViewType( array.getData(), sizes );
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


   // extra methods

   // TODO: rename to setSizes and make sure that overloading with the following method works
   void setSize( const SizesHolderType& sizes )
   {
      this->sizes = sizes;
      array.setSize( getStorageSize() );
   }

   template< typename... IndexTypes >
   void setSizes( IndexTypes&&... sizes )
   {
      static_assert( sizeof...( sizes ) == getDimension(), "got wrong number of sizes" );
      __ndarray_impl::setSizesHelper( this->sizes, std::forward< IndexTypes >( sizes )... );
      array.setSize( getStorageSize() );
   }

   void setLike( const NDArrayStorage& other )
   {
      this->sizes = other.getSizes();
      array.setSize( getStorageSize() );
   }

   void reset()
   {
      this->sizes = SizesHolder{};
      TNL_ASSERT_EQ( getStorageSize(), 0, "Failed to reset the sizes." );
      array.reset();
   }

   // "safe" accessor - will do slow copy from device
   template< typename... IndexTypes >
   ValueType
   getElement( IndexTypes&&... indices ) const
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      __ndarray_impl::assertIndicesInBounds( sizes, std::forward< IndexTypes >( indices )... );
      TNL_ASSERT_LT( getStorageIndex( std::forward< IndexTypes >( indices )... ), getStorageSize(),
                     "storage index out of bounds - either input error or a bug in the indexer" );
      return array.getElement( getStorageIndex( std::forward< IndexTypes >( indices )... ) );
   }

   const StorageArray& getStorageArray() const
   {
      return array;
   }

   StorageArray& getStorageArray()
   {
      return array;
   }

   void setValue( ValueType value )
   {
      array.setValue( value );
   }

protected:
   StorageArray array;
   SizesHolder sizes;

   using StrideBase = __ndarray_impl::DummyStrideBase< typename SizesHolder::IndexType, SizesHolder::getDimension() >;
};

template< typename Value,
          typename SizesHolder,
          typename PermutationHost = std::make_index_sequence< SizesHolder::getDimension() >,  // identity by default
          typename PermutationCuda = std::make_index_sequence< SizesHolder::getDimension() >,  // identity by default
          typename Device = Devices::Host,
          typename Index = typename SizesHolder::IndexType >
class NDArray
: public NDArrayStorage< Array< Value, Device, Index >,
                         SizesHolder,
                         typename std::conditional< std::is_same< Device, Devices::Host >::value,
                                                    PermutationHost,
                                                    PermutationCuda >::type,
                         __ndarray_impl::NDArrayBase< SliceInfo< 0, 0 > > >
{};

template< typename Value,
          typename SizesHolder,
          typename Permutation = std::make_index_sequence< SizesHolder::getDimension() >,  // identity by default
          typename Index = typename SizesHolder::IndexType >
class StaticNDArray
: public NDArrayStorage< StaticArray< __ndarray_impl::StaticStorageSizeGetter< SizesHolder >::get(), Value >,
                         SizesHolder,
                         Permutation,
                         __ndarray_impl::NDArrayBase< SliceInfo< 0, 0 > >,
                         void >
{
   static_assert( __ndarray_impl::StaticStorageSizeGetter< SizesHolder >::get() > 0,
                  "All dimensions of a static array must to be positive." );
};

template< typename Value,
          std::size_t Rows,
          std::size_t Columns,
          typename Permutation = std::index_sequence< 0, 1 > >  // identity by default
class StaticMatrix
: public StaticNDArray< Value,
                        SizesHolder< std::size_t, Rows, Columns >,
                        Permutation >
{
public:
   static constexpr std::size_t getRows()
   {
      return Rows;
   }

   __cuda_callable__
   static constexpr std::size_t getColumns()
   {
      return Columns;
   }
};

template< typename Value,
          typename SizesHolder,
          typename PermutationHost = std::make_index_sequence< SizesHolder::getDimension() >,  // identity by default
          typename SliceInfoHost = SliceInfo<>,  // no slicing by default
          typename PermutationCuda = std::make_index_sequence< SizesHolder::getDimension() >,  // identity by default
          typename SliceInfoCuda = SliceInfo<>,  // no slicing by default
          typename Device = Devices::Host,
          typename Index = typename SizesHolder::IndexType >
class SlicedNDArray
: public NDArrayStorage< Array< Value, Device, Index >,
                         SizesHolder,
                         typename std::conditional< std::is_same< Device, Devices::Host >::value,
                                                    PermutationHost,
                                                    PermutationCuda >::type,
                         __ndarray_impl::SlicedNDArrayBase<
                            typename std::conditional< std::is_same< Device, Devices::Host >::value,
                                                       SliceInfoHost,
                                                       SliceInfoCuda >::type >
                        >
{};

} // namespace Containers
} // namespace TNL
