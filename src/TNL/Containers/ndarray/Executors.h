/***************************************************************************
                          Executors.h  -  description
                             -------------------
    begin                : Dec 24, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/ParallelFor.h>

#include <TNL/Containers/ndarray/Meta.h>
#include <TNL/Containers/ndarray/SizesHolder.h>

namespace TNL {
namespace Containers {
namespace __ndarray_impl {

template< typename Permutation,
          typename LevelTag = IndexTag< 0 > >
struct SequentialExecutor
{
   template< typename Begins,
             typename Ends,
             typename Func,
             typename... Indices >
   __cuda_callable__
   void operator()( const Begins& begins, const Ends& ends, Func f, Indices&&... indices )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(),
                     "wrong begins or ends" );

      SequentialExecutor< Permutation, IndexTag< LevelTag::value + 1 > > exec;
      const auto begin = begins.template getSize< get< LevelTag::value >( Permutation{} ) >();
      const auto end = ends.template getSize< get< LevelTag::value >( Permutation{} ) >();
      for( auto i = begin; i < end; i++ )
         exec( begins, ends, f, std::forward< Indices >( indices )..., i );
   }
};

template< typename Permutation >
struct SequentialExecutor< Permutation, IndexTag< Permutation::size() - 1 > >
{
   template< typename Begins,
             typename Ends,
             typename Func,
             typename... Indices >
   __cuda_callable__
   void operator()( const Begins& begins, const Ends& ends, Func f, Indices&&... indices )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(),
                     "wrong begins or ends" );
      static_assert( sizeof...(indices) == Begins::getDimension() - 1,
                     "invalid number of indices in the final step of the SequentialExecutor" );

      using LevelTag = IndexTag< Permutation::size() - 1 >;

      const auto begin = begins.template getSize< get< LevelTag::value >( Permutation{} ) >();
      const auto end = ends.template getSize< get< LevelTag::value >( Permutation{} ) >();
      for( auto i = begin; i < end; i++ )
         call_with_unpermuted_arguments< Permutation >( f, std::forward< Indices >( indices )..., i );
   }
};


template< typename Permutation,
          typename LevelTag = IndexTag< Permutation::size() - 1 > >
struct SequentialExecutorRTL
{
   template< typename Begins,
             typename Ends,
             typename Func,
             typename... Indices >
   __cuda_callable__
   void operator()( const Begins& begins, const Ends& ends, Func f, Indices&&... indices )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(),
                     "wrong begins or ends" );

      SequentialExecutorRTL< Permutation, IndexTag< LevelTag::value - 1 > > exec;
      const auto begin = begins.template getSize< get< LevelTag::value >( Permutation{} ) >();
      const auto end = ends.template getSize< get< LevelTag::value >( Permutation{} ) >();
      for( auto i = begin; i < end; i++ )
         exec( begins, ends, f, i, std::forward< Indices >( indices )... );
   }
};

template< typename Permutation >
struct SequentialExecutorRTL< Permutation, IndexTag< 0 > >
{
   template< typename Begins,
             typename Ends,
             typename Func,
             typename... Indices >
   __cuda_callable__
   void operator()( const Begins& begins, const Ends& ends, Func f, Indices&&... indices )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(),
                     "wrong begins or ends" );
      static_assert( sizeof...(indices) == Begins::getDimension() - 1,
                     "invalid number of indices in the final step of the SequentialExecutorRTL" );

      const auto begin = begins.template getSize< get< 0 >( Permutation{} ) >();
      const auto end = ends.template getSize< get< 0 >( Permutation{} ) >();
      for( auto i = begin; i < end; i++ )
         call_with_unpermuted_arguments< Permutation >( f, i, std::forward< Indices >( indices )... );
   }
};


template< typename Permutation,
          typename Device >
struct ParallelExecutorDeviceDispatch
{
   template< typename Begins,
             typename Ends,
             typename Func >
   void operator()( const Begins& begins, const Ends& ends, Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(),
                     "wrong begins or ends" );

      using Index = typename Ends::IndexType;

      auto kernel = [=] ( Index i2, Index i1, Index i0 )
      {
         SequentialExecutor< Permutation, IndexTag< 3 > > exec;
         exec( begins, ends, f, i0, i1, i2 );
      };

      const Index begin0 = begins.template getSize< get< 0 >( Permutation{} ) >();
      const Index begin1 = begins.template getSize< get< 1 >( Permutation{} ) >();
      const Index begin2 = begins.template getSize< get< 2 >( Permutation{} ) >();
      const Index end0 = ends.template getSize< get< 0 >( Permutation{} ) >();
      const Index end1 = ends.template getSize< get< 1 >( Permutation{} ) >();
      const Index end2 = ends.template getSize< get< 2 >( Permutation{} ) >();
      ParallelFor3D< Device >::exec( begin2, begin1, begin0, end2, end1, end0, kernel );
   }
};

template< typename Permutation >
struct ParallelExecutorDeviceDispatch< Permutation, Devices::Cuda >
{
   template< typename Begins,
             typename Ends,
             typename Func >
   void operator()( const Begins& begins, const Ends& ends, Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(),
                     "wrong begins or ends" );

      using Index = typename Ends::IndexType;

      auto kernel = [=] __cuda_callable__ ( Index i2, Index i1, Index i0 )
      {
         SequentialExecutorRTL< Permutation, IndexTag< Begins::getDimension() - 4 > > exec;
         exec( begins, ends, f, i0, i1, i2 );
      };

      const Index begin0 = begins.template getSize< get< Begins::getDimension() - 3 >( Permutation{} ) >();
      const Index begin1 = begins.template getSize< get< Begins::getDimension() - 2 >( Permutation{} ) >();
      const Index begin2 = begins.template getSize< get< Begins::getDimension() - 1 >( Permutation{} ) >();
      const Index end0 = ends.template getSize< get< Ends::getDimension() - 3 >( Permutation{} ) >();
      const Index end1 = ends.template getSize< get< Ends::getDimension() - 2 >( Permutation{} ) >();
      const Index end2 = ends.template getSize< get< Ends::getDimension() - 1 >( Permutation{} ) >();
      ParallelFor3D< Devices::Cuda >::exec( begin2, begin1, begin0, end2, end1, end0, kernel );
   }
};

template< typename Permutation,
          typename Device,
          typename DimTag = IndexTag< Permutation::size() > >
struct ParallelExecutor
{
   template< typename Begins,
             typename Ends,
             typename Func >
   void operator()( const Begins& begins, const Ends& ends, Func f )
   {
      ParallelExecutorDeviceDispatch< Permutation, Device > dispatch;
      dispatch( begins, ends, f );
   }
};

template< typename Permutation,
          typename Device >
struct ParallelExecutor< Permutation, Device, IndexTag< 3 > >
{
   template< typename Begins,
             typename Ends,
             typename Func >
   void operator()( const Begins& begins, const Ends& ends, Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(),
                     "wrong begins or ends" );

      using Index = typename Ends::IndexType;

      auto kernel = [=] __cuda_callable__ ( Index i2, Index i1, Index i0 )
      {
         call_with_unpermuted_arguments< Permutation >( f, i0, i1, i2 );
      };

      const Index begin0 = begins.template getSize< get< 0 >( Permutation{} ) >();
      const Index begin1 = begins.template getSize< get< 1 >( Permutation{} ) >();
      const Index begin2 = begins.template getSize< get< 2 >( Permutation{} ) >();
      const Index end0 = ends.template getSize< get< 0 >( Permutation{} ) >();
      const Index end1 = ends.template getSize< get< 1 >( Permutation{} ) >();
      const Index end2 = ends.template getSize< get< 2 >( Permutation{} ) >();
      ParallelFor3D< Device >::exec( begin2, begin1, begin0, end2, end1, end0, kernel );
   }
};

template< typename Permutation,
          typename Device >
struct ParallelExecutor< Permutation, Device, IndexTag< 2 > >
{
   template< typename Begins,
             typename Ends,
             typename Func >
   void operator()( const Begins& begins, const Ends& ends, Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(),
                     "wrong begins or ends" );

      using Index = typename Ends::IndexType;

      auto kernel = [=] __cuda_callable__ ( Index i1, Index i0 )
      {
         call_with_unpermuted_arguments< Permutation >( f, i0, i1 );
      };

      const Index begin0 = begins.template getSize< get< 0 >( Permutation{} ) >();
      const Index begin1 = begins.template getSize< get< 1 >( Permutation{} ) >();
      const Index end0 = ends.template getSize< get< 0 >( Permutation{} ) >();
      const Index end1 = ends.template getSize< get< 1 >( Permutation{} ) >();
      ParallelFor2D< Device >::exec( begin1, begin0, end1, end0, kernel );
   }
};

template< typename Permutation,
          typename Device >
struct ParallelExecutor< Permutation, Device, IndexTag< 1 > >
{
   template< typename Begins,
             typename Ends,
             typename Func >
   void operator()( const Begins& begins, const Ends& ends, Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(),
                     "wrong begins or ends" );

      using Index = typename Ends::IndexType;

//      auto kernel = [=] __cuda_callable__ ( Index i )
//      {
//         call_with_unpermuted_arguments< Permutation >( f, i );
//      };

      const Index begin = begins.template getSize< get< 0 >( Permutation{} ) >();
      const Index end = ends.template getSize< get< 0 >( Permutation{} ) >();
//      ParallelFor< Device >::exec( begin, end, kernel );
      ParallelFor< Device >::exec( begin, end, f );
   }
};


// Device may be void which stands for StaticNDArray
template< typename Permutation,
          typename Device >
struct ExecutorDispatcher
{
   template< typename Begins, typename Ends, typename Func >
   void operator()( const Begins& begins, const Ends& ends, Func f )
   {
      SequentialExecutor< Permutation >()( begins, ends, f );
   }
};

template< typename Permutation >
struct ExecutorDispatcher< Permutation, Devices::Host >
{
   template< typename Begins, typename Ends, typename Func >
   void operator()( const Begins& begins, const Ends& ends, Func f )
   {
      if( Devices::Host::isOMPEnabled() && Devices::Host::getMaxThreadsCount() > 1 )
         ParallelExecutor< Permutation, Devices::Host >()( begins, ends, f );
      else
         SequentialExecutor< Permutation >()( begins, ends, f );
   }
};

template< typename Permutation >
struct ExecutorDispatcher< Permutation, Devices::Cuda >
{
   template< typename Begins, typename Ends, typename Func >
   void operator()( const Begins& begins, const Ends& ends, Func f )
   {
      ParallelExecutor< Permutation, Devices::Cuda >()( begins, ends, f );
   }
};

} // namespace __ndarray_impl
} // namespace Containers
} // namespace TNL
