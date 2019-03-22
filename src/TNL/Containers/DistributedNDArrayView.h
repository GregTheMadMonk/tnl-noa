/***************************************************************************
                          DistributedNDArrayView.h  -  description
                             -------------------
    begin                : Dec 27, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Containers/NDArrayView.h>
#include <TNL/Containers/Subrange.h>

namespace TNL {
namespace Containers {

template< typename NDArrayView,
          typename Communicator = Communicators::MpiCommunicator,
          typename Overlaps = __ndarray_impl::make_constant_index_sequence< NDArrayView::getDimension(), 0 > >
class DistributedNDArrayView
{
   using CommunicationGroup = typename Communicator::CommunicationGroup;
public:
   using ValueType = typename NDArrayView::ValueType;
   using DeviceType = typename NDArrayView::DeviceType;
   using IndexType = typename NDArrayView::IndexType;
   using SizesHolderType = typename NDArrayView::SizesHolderType;
   using PermutationType = typename NDArrayView::PermutationType;
   using CommunicatorType = Communicator;
   using LocalBeginsType = __ndarray_impl::LocalBeginsHolder< typename NDArrayView::SizesHolderType >;
   using LocalRangeType = Subrange< IndexType >;
   using OverlapsType = Overlaps;

   using ViewType = DistributedNDArrayView< NDArrayView, Communicator, Overlaps >;
   using ConstViewType = DistributedNDArrayView< typename NDArrayView::ConstViewType, Communicator, Overlaps >;

   static_assert( Overlaps::size() == NDArrayView::getDimension(), "invalid overlaps" );

   __cuda_callable__
   DistributedNDArrayView() = default;

   // explicit initialization by local array view, global sizes and local begins and ends
   __cuda_callable__
   DistributedNDArrayView( NDArrayView localView, SizesHolderType globalSizes, LocalBeginsType localBegins, SizesHolderType localEnds, CommunicationGroup group )
   : localView(localView), group(group), globalSizes(globalSizes), localBegins(localBegins), localEnds(localEnds) {}

   // Copy-constructor does shallow copy, so views can be passed-by-value into
   // CUDA kernels and they can be captured-by-value in __cuda_callable__
   // lambda functions.
   __cuda_callable__
   DistributedNDArrayView( const DistributedNDArrayView& ) = default;

   // default move-constructor
   __cuda_callable__
   DistributedNDArrayView( DistributedNDArrayView&& ) = default;

   // Copy-assignment does deep copy, just like regular array, but the sizes
   // must match (i.e. copy-assignment cannot resize).
   __cuda_callable__
   DistributedNDArrayView& operator=( const DistributedNDArrayView& other ) = default;

   // There is no move-assignment operator, so expressions like `a = b.getView()`
   // are resolved as copy-assignment.

   // method for rebinding (reinitialization)
   __cuda_callable__
   void bind( DistributedNDArrayView view )
   {
      localView.bind( view.localView );
      group = view.group;
      globalSizes = view.globalSizes;
      localBegins = view.localBegins;
      localEnds = view.localEnds;
   }

   __cuda_callable__
   void reset()
   {
      localView.reset();
      group = CommunicatorType::NullGroup;
      globalSizes = SizesHolderType{};
      localBegins = LocalBeginsType{};
      localEnds = SizesHolderType{};
   }

   static constexpr std::size_t getDimension()
   {
      return NDArrayView::getDimension();
   }

   __cuda_callable__
   CommunicationGroup getCommunicationGroup() const
   {
      return group;
   }

   // Returns the *global* sizes
   __cuda_callable__
   const SizesHolderType& getSizes() const
   {
      return globalSizes;
   }

   // Returns the *global* size
   template< std::size_t level >
   __cuda_callable__
   IndexType getSize() const
   {
      return globalSizes.template getSize< level >();
   }

   __cuda_callable__
   LocalBeginsType getLocalBegins() const
   {
      return localBegins;
   }

   __cuda_callable__
   SizesHolderType getLocalEnds() const
   {
      return localEnds;
   }

   template< std::size_t level >
   __cuda_callable__
   LocalRangeType getLocalRange() const
   {
      return LocalRangeType( localBegins.template getSize< level >(), localEnds.template getSize< level >() );
   }

   // returns the local storage size
   __cuda_callable__
   IndexType getLocalStorageSize() const
   {
      return localView.getStorageSize();
   }

   template< typename... IndexTypes >
   __cuda_callable__
   ValueType&
   operator()( IndexTypes&&... indices )
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      __ndarray_impl::assertIndicesInRange( localBegins, localEnds, Overlaps{}, std::forward< IndexTypes >( indices )... );
      return __ndarray_impl::call_with_unshifted_indices< LocalBeginsType, Overlaps >( localBegins, localView, std::forward< IndexTypes >( indices )... );
   }

   template< typename... IndexTypes >
   __cuda_callable__
   const ValueType&
   operator()( IndexTypes&&... indices ) const
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      __ndarray_impl::assertIndicesInRange( localBegins, localEnds, Overlaps{}, std::forward< IndexTypes >( indices )... );
      return __ndarray_impl::call_with_unshifted_indices< LocalBeginsType, Overlaps >( localBegins, localView, std::forward< IndexTypes >( indices )... );
   }

   // bracket operator for 1D arrays
   __cuda_callable__
   ValueType&
   operator[]( IndexType index )
   {
      static_assert( getDimension() == 1, "the access via operator[] is provided only for 1D arrays" );
      __ndarray_impl::assertIndicesInRange( localBegins, localEnds, Overlaps{}, std::forward< IndexType >( index ) );
      return localView[ __ndarray_impl::get<0>( Overlaps{} ) + index - localBegins.template getSize< 0 >() ];
   }

   __cuda_callable__
   const ValueType&
   operator[]( IndexType index ) const
   {
      static_assert( getDimension() == 1, "the access via operator[] is provided only for 1D arrays" );
      __ndarray_impl::assertIndicesInRange( localBegins, localEnds, Overlaps{}, std::forward< IndexType >( index ) );
      return localView[ __ndarray_impl::get<0>( Overlaps{} ) + index - localBegins.template getSize< 0 >() ];
   }

   __cuda_callable__
   ViewType getView()
   {
      return ViewType( *this );
   }

   __cuda_callable__
   ConstViewType getConstView() const
   {
      return ConstViewType( localView, globalSizes, localBegins, localEnds, group );
   }

   // TODO: overlaps should be skipped, otherwise it works only after synchronization
   bool operator==( const DistributedNDArrayView& other ) const
   {
      // we can't run allreduce if the communication groups are different
      if( group != other.getCommunicationGroup() )
         return false;
      const bool localResult =
            globalSizes == other.globalSizes &&
            localBegins == other.localBegins &&
            localEnds == other.localEnds &&
            localView == other.localView;
      bool result = true;
      if( group != CommunicatorType::NullGroup )
         CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LAND, group );
      return result;
   }

   bool operator!=( const DistributedNDArrayView& other ) const
   {
      return ! (*this == other);
   }

   // iterate over all local elements
   template< typename Device2 = DeviceType, typename Func >
   void forAll( Func f ) const
   {
      __ndarray_impl::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( localBegins, localEnds, f );
   }

   // iterate over local elements which are not neighbours of *global* boundaries
   template< typename Device2 = DeviceType, typename Func >
   void forInternal( Func f ) const
   {
      // add static sizes
      using Begins = __ndarray_impl::LocalBeginsHolder< SizesHolderType, 1 >;
      // add dynamic sizes
      Begins begins;
      __ndarray_impl::SetSizesAddHelper< 1, Begins, SizesHolderType, Overlaps >::add( begins, SizesHolderType{} );
      __ndarray_impl::SetSizesMaxHelper< Begins, LocalBeginsType >::max( begins, localBegins );

      // subtract static sizes
      using Ends = typename __ndarray_impl::SubtractedSizesHolder< SizesHolderType, 1 >::type;
      // subtract dynamic sizes
      Ends ends;
      __ndarray_impl::SetSizesSubtractHelper< 1, Ends, SizesHolderType, Overlaps >::subtract( ends, globalSizes );
      __ndarray_impl::SetSizesMinHelper< Ends, SizesHolderType >::min( ends, localEnds );

      __ndarray_impl::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( begins, ends, f );
   }

   // iterate over local elements inside the given [begins, ends) range specified by global indices
   template< typename Device2 = DeviceType, typename Func, typename Begins, typename Ends >
   void forInternal( Func f, const Begins& begins, const Ends& ends ) const
   {
      // TODO: assert "localBegins <= begins <= localEnds", "localBegins <= ends <= localEnds"
      __ndarray_impl::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( begins, ends, f );
   }

   // iterate over local elements which are not neighbours of overlaps (if all overlaps are 0, it is equivalent to forAll)
   template< typename Device2 = DeviceType, typename Func >
   void forLocalInternal( Func f ) const
   {
      // add dynamic sizes
      LocalBeginsType begins;
      __ndarray_impl::SetSizesAddHelper< 1, LocalBeginsType, SizesHolderType, Overlaps >::add( begins, localBegins, false );

      // subtract dynamic sizes
      SizesHolderType ends;
      __ndarray_impl::SetSizesSubtractHelper< 1, SizesHolderType, SizesHolderType, Overlaps >::subtract( ends, localEnds, false );

      __ndarray_impl::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( begins, ends, f );
   }

protected:
   NDArrayView localView;
   CommunicationGroup group = Communicator::NullGroup;
   SizesHolderType globalSizes;
   // static sizes should have different type: localBegin is always 0, localEnd is always the full size
   LocalBeginsType localBegins;
   SizesHolderType localEnds;
};

} // namespace Containers
} // namespace TNL
