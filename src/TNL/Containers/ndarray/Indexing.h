/***************************************************************************
                          Indexing.h  -  description
                             -------------------
    begin                : Dec 24, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <array>

#include <TNL/Assert.h>
#include <TNL/Devices/CudaCallable.h>
#include <TNL/StaticFor.h>

#include <TNL/Containers/ndarray/Meta.h>

namespace TNL {
namespace Containers {
namespace __ndarray_impl {

// Dynamic storage size with alignment
template< typename SizesHolder,
          typename Alignment,
          typename LevelTag = IndexTag< SizesHolder::getDimension() - 1 > >
struct StorageSizeGetter
{
   static typename SizesHolder::IndexType
   __cuda_callable__
   get( const SizesHolder& sizes )
   {
      const auto size = Alignment::template getAlignedSize< LevelTag::value >( sizes );
      return size * StorageSizeGetter< SizesHolder, Alignment, IndexTag< LevelTag::value - 1 > >::get( sizes );
   }

   template< typename Permutation >
   __cuda_callable__
   static typename SizesHolder::IndexType
   getPermuted( const SizesHolder& sizes, Permutation )
   {
      constexpr std::size_t idx = __ndarray_impl::get< LevelTag::value >( Permutation{} );
      const auto size = Alignment::template getAlignedSize< idx >( sizes );
      return size * StorageSizeGetter< SizesHolder, Alignment, IndexTag< LevelTag::value - 1 > >::get( sizes );
   }
};

template< typename SizesHolder, typename Alignment >
struct StorageSizeGetter< SizesHolder, Alignment, IndexTag< 0 > >
{
   static typename SizesHolder::IndexType
   __cuda_callable__
   get( const SizesHolder& sizes )
   {
      return Alignment::template getAlignedSize< 0 >( sizes );
   }

   template< typename Permutation >
   __cuda_callable__
   static typename SizesHolder::IndexType
   getPermuted( const SizesHolder& sizes, Permutation )
   {
      constexpr std::size_t idx = __ndarray_impl::get< 0 >( Permutation{} );
      return Alignment::template getAlignedSize< idx >( sizes );
   }
};


// Static storage size without alignment, used in StaticNDArray
template< typename SizesHolder,
          typename LevelTag = IndexTag< SizesHolder::getDimension() - 1 > >
struct StaticStorageSizeGetter
{
   constexpr static std::size_t get()
   {
      return SizesHolder::template getStaticSize< LevelTag::value >() *
             StaticStorageSizeGetter< SizesHolder, IndexTag< LevelTag::value - 1 > >::get();
   }
};

template< typename SizesHolder >
struct StaticStorageSizeGetter< SizesHolder, IndexTag< 0 > >
{
   constexpr static std::size_t get()
   {
      return SizesHolder::template getStaticSize< 0 >();
   }
};


template< std::size_t level = 0,
          typename SizesHolder,
          typename Index,
          typename... IndexTypes >
void setSizesHelper( SizesHolder& holder,
                     Index&& size,
                     IndexTypes&&... otherSizes )
{
   holder.template setSize< level >( std::forward< Index >( size ) );
   setSizesHelper< level + 1 >( holder, std::forward< IndexTypes >( otherSizes )... );
}

template< std::size_t level = 0,
          typename SizesHolder,
          typename Index >
void setSizesHelper( SizesHolder& holder,
                     Index&& size )
{
   holder.template setSize< level >( std::forward< Index >( size ) );
}


// helper for the forInternal method
template< std::size_t ConstValue,
          typename TargetHolder,
          typename SourceHolder,
          std::size_t level = TargetHolder::getDimension() - 1 >
struct SetSizesSubtractHelper
{
   static void subtract( TargetHolder& target,
                         const SourceHolder& source )
   {
      if( source.template getStaticSize< level >() == 0 )
         target.template setSize< level >( source.template getSize< level >() - ConstValue );
      SetSizesSubtractHelper< ConstValue, TargetHolder, SourceHolder, level - 1 >::subtract( target, source );
   }
};

template< std::size_t ConstValue,
          typename TargetHolder,
          typename SourceHolder >
struct SetSizesSubtractHelper< ConstValue, TargetHolder, SourceHolder, 0 >
{
   static void subtract( TargetHolder& target,
                         const SourceHolder& source )
   {
      if( source.template getStaticSize< 0 >() == 0 )
         target.template setSize< 0 >( source.template getSize< 0 >() - ConstValue );
   }
};


// A variadic bounds-checker for indices
template< typename SizesHolder >
__cuda_callable__
void assertIndicesInBounds( const SizesHolder& )
{}

template< typename SizesHolder,
          typename Index,
          typename... IndexTypes >
__cuda_callable__
void assertIndicesInBounds( const SizesHolder& sizes, Index&& i, IndexTypes&&... indices )
{
#ifndef NDEBUG
   // sizes.template getSize<...>() cannot be inside the assert macro, but the variables
   // shouldn't be declared when compiling without assertions
   constexpr std::size_t level = SizesHolder::getDimension() - sizeof...(indices) - 1;
   const auto size = sizes.template getSize< level >();
   TNL_ASSERT_LT( i, size, "Input error - some index is out of bounds." );
#endif
   assertIndicesInBounds( sizes, std::forward< IndexTypes >( indices )... );
}


// A variadic bounds-checker for distributed indices with overlaps
template< typename SizesHolder1, typename SizesHolder2, typename Overlaps >
__cuda_callable__
void assertIndicesInRange( const SizesHolder1&, const SizesHolder2&, const Overlaps& )
{}

template< typename SizesHolder1,
          typename SizesHolder2,
          typename Overlaps,
          typename Index,
          typename... IndexTypes >
__cuda_callable__
void assertIndicesInRange( const SizesHolder1& begins, const SizesHolder2& ends, const Overlaps& overlaps, Index&& i, IndexTypes&&... indices )
{
   static_assert( SizesHolder1::getDimension() == SizesHolder2::getDimension(),
                  "Inconsistent begins and ends." );
#ifndef NDEBUG
   // sizes.template getSize<...>() cannot be inside the assert macro, but the variables
   // shouldn't be declared when compiling without assertions
   constexpr std::size_t level = SizesHolder1::getDimension() - sizeof...(indices) - 1;
   const auto begin = begins.template getSize< level >();
   const auto end = ends.template getSize< level >();
   TNL_ASSERT_LE( begin - get<level>( overlaps ), i, "Input error - some index is below the lower bound." );
   TNL_ASSERT_LT( i, end + get<level>( overlaps ), "Input error - some index is above the upper bound." );
#endif
   assertIndicesInRange( begins, ends, overlaps, std::forward< IndexTypes >( indices )... );
}


template< typename SizesHolder,
          typename Overlaps,
          typename Sequence >
struct IndexUnshiftHelper
{};

template< typename SizesHolder,
          typename Overlaps,
          std::size_t... N >
struct IndexUnshiftHelper< SizesHolder, Overlaps, std::index_sequence< N... > >
{
   template< typename Func,
             typename... Indices >
   __cuda_callable__
   static auto apply( const SizesHolder& begins, Func&& f, Indices&&... indices ) -> decltype(auto)
   {
      return f( ( get<N>( Overlaps{} ) + std::forward< Indices >( indices ) - begins.template getSize< N >() )... );
   }

   template< typename Func,
             typename... Indices >
   static auto apply_host( const SizesHolder& begins, Func&& f, Indices&&... indices ) -> decltype(auto)
   {
      return f( ( get<N>( Overlaps{} ) + std::forward< Indices >( indices ) - begins.template getSize< N >() )... );
   }
};

template< typename SizesHolder,
          typename Overlaps = make_constant_index_sequence< SizesHolder::getDimension(), 0 >,
          typename Func,
          typename... Indices >
__cuda_callable__
auto call_with_unshifted_indices( const SizesHolder& begins, Func&& f, Indices&&... indices ) -> decltype(auto)
{
   return IndexUnshiftHelper< SizesHolder, Overlaps, std::make_index_sequence< sizeof...( Indices ) > >
          ::apply( begins, std::forward< Func >( f ), std::forward< Indices >( indices )... );
}

template< typename SizesHolder,
          typename Overlaps = make_constant_index_sequence< SizesHolder::getDimension(), 0 >,
          typename Func,
          typename... Indices >
auto host_call_with_unshifted_indices( const SizesHolder& begins, Func&& f, Indices&&... indices ) -> decltype(auto)
{
   return IndexUnshiftHelper< SizesHolder, Overlaps, std::make_index_sequence< sizeof...( Indices ) > >
          ::apply_host( begins, std::forward< Func >( f ), std::forward< Indices >( indices )... );
}


template< typename Permutation,
          typename Alignment,
          typename SliceInfo,
          std::size_t level = Permutation::size() - 1,
          bool _sliced_level = ( SliceInfo::getSliceSize( get< level >( Permutation{} ) ) > 0 ) >
struct SlicedIndexer
{};

template< typename Permutation,
          typename Alignment,
          typename SliceInfo,
          std::size_t level >
struct SlicedIndexer< Permutation, Alignment, SliceInfo, level, false >
{
   template< typename SizesHolder, typename StridesHolder, typename... Indices >
   __cuda_callable__
   static typename SizesHolder::IndexType
   getIndex( const SizesHolder& sizes,
             const StridesHolder& strides,
             Indices&&... indices )
   {
      static constexpr std::size_t idx = get< level >( Permutation{} );
      const auto alpha = get_from_pack< idx >( std::forward< Indices >( indices )... );
      const auto previous = SlicedIndexer< Permutation, Alignment, SliceInfo, level - 1 >::getIndex( sizes, strides, std::forward< Indices >( indices )... );
      return strides.template getStride< idx >( alpha ) * ( alpha + Alignment::template getAlignedSize< idx >( sizes ) * previous );
   }
};

template< typename Permutation,
          typename Alignment,
          typename SliceInfo,
          std::size_t level >
struct SlicedIndexer< Permutation, Alignment, SliceInfo, level, true >
{
   template< typename SizesHolder, typename StridesHolder, typename... Indices >
   __cuda_callable__
   static typename SizesHolder::IndexType
   getIndex( const SizesHolder& sizes,
             const StridesHolder& strides,
             Indices&&... indices )
   {
      static_assert( SizesHolder::template getStaticSize< get< level >( Permutation{} ) >() == 0,
                     "Invalid SliceInfo: static dimension cannot be sliced." );

      static constexpr std::size_t idx = get< level >( Permutation{} );
      const auto alpha = get_from_pack< idx >( std::forward< Indices >( indices )... );
      static constexpr std::size_t S = SliceInfo::getSliceSize( idx );
      // TODO: check the calculation with strides
      return strides.template getStride< idx >( alpha ) *
                  ( S * (alpha / S) * StorageSizeGetter< SizesHolder, Alignment, IndexTag< level - 1 > >::getPermuted( sizes, Permutation{} ) +
                    alpha % S ) +
             S * SlicedIndexer< Permutation, Alignment, SliceInfo, level - 1 >::getIndex( sizes, strides, std::forward< Indices >( indices )... );
   }
};

template< typename Permutation,
          typename Alignment,
          typename SliceInfo >
struct SlicedIndexer< Permutation, Alignment, SliceInfo, 0, false >
{
   template< typename SizesHolder, typename StridesHolder, typename... Indices >
   __cuda_callable__
   static typename SizesHolder::IndexType
   getIndex( const SizesHolder& sizes,
             const StridesHolder& strides,
             Indices&&... indices )
   {
      static constexpr std::size_t idx = get< 0 >( Permutation{} );
      const auto alpha = get_from_pack< idx >( std::forward< Indices >( indices )... );
      return strides.template getStride< idx >( alpha ) * alpha;
   }
};

template< typename Permutation,
          typename Alignment,
          typename SliceInfo >
struct SlicedIndexer< Permutation, Alignment, SliceInfo, 0, true >
{
   template< typename SizesHolder, typename StridesHolder, typename... Indices >
   __cuda_callable__
   static typename SizesHolder::IndexType
   getIndex( const SizesHolder& sizes,
             const StridesHolder& strides,
             Indices&&... indices )
   {
      static constexpr std::size_t idx = get< 0 >( Permutation{} );
      const auto alpha = get_from_pack< idx >( std::forward< Indices >( indices )... );
      return strides.template getStride< idx >( alpha ) * alpha;
   }
};


// SliceInfo should be always empty (i.e. sliceSize == 0)
template< typename SliceInfo >
struct NDArrayBase
{
   template< typename Permutation >
   struct Alignment
   {
      template< std::size_t dimension, typename SizesHolder >
      __cuda_callable__
      static typename SizesHolder::IndexType
      getAlignedSize( const SizesHolder& sizes )
      {
         const auto size = sizes.template getSize< dimension >();
         // round up the last dynamic dimension to improve performance
         // TODO: aligning is good for GPU, but bad for CPU
//         static constexpr decltype(size) mult = 32;
//         if( dimension == get< Permutation::size() - 1 >( Permutation{} )
//                 && SizesHolder::template getStaticSize< dimension >() == 0 )
//             return mult * ( size / mult + ( size % mult != 0 ) );
         return size;
      }
   };

   template< typename Permutation, typename SizesHolder, typename StridesHolder, typename... Indices >
   __cuda_callable__
   typename SizesHolder::IndexType
   static getStorageIndex( const SizesHolder& sizes, const StridesHolder& strides, Indices&&... indices )
   {
      static_assert( check_slice_size( SizesHolder::getDimension(), 0 ), "BUG - invalid SliceInfo type passed to NDArrayBase" );
      using Alignment = Alignment< Permutation >;
      return SlicedIndexer< Permutation, Alignment, SliceInfo >::getIndex( sizes, strides, std::forward< Indices >( indices )... );
   }

private:
   static constexpr bool check_slice_size( std::size_t dim, std::size_t sliceSize )
   {
      for( std::size_t i = 0; i < dim; i++ )
         if( SliceInfo::getSliceSize( i ) != sliceSize )
            return false;
      return true;
   }
};


template< typename SliceInfo >
struct SlicedNDArrayBase
{
   template< typename Permutation >
   struct Alignment
   {
      template< std::size_t dimension, typename SizesHolder >
      __cuda_callable__
      static typename SizesHolder::IndexType
      getAlignedSize( const SizesHolder& sizes )
      {
         const auto size = sizes.template getSize< dimension >();
         if( SliceInfo::getSliceSize(dimension) > 0 )
            // round to multiple of SliceSize
            return SliceInfo::getSliceSize(dimension) * (
                        size / SliceInfo::getSliceSize(dimension) +
                        ( size % SliceInfo::getSliceSize(dimension) != 0 )
                     );
         // unmodified
         return size;
      }
   };

   template< typename Permutation, typename SizesHolder, typename StridesHolder, typename... Indices >
   __cuda_callable__
   static typename SizesHolder::IndexType
   getStorageIndex( const SizesHolder& sizes, const StridesHolder& strides, Indices&&... indices )
   {
      using Alignment = Alignment< Permutation >;
      return SlicedIndexer< Permutation, Alignment, SliceInfo >::getIndex( sizes, strides, std::forward< Indices >( indices )... );
   }
};

} // namespace __ndarray_impl
} // namespace Containers
} // namespace TNL
