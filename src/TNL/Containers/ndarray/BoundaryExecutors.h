/***************************************************************************
                          BoundaryExecutors.h  -  description
                             -------------------
    begin                : Feb 09, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
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
struct SequentialBoundaryExecutor_inner
{
   template< typename Begins,
             typename SkipBegins,
             typename SkipEnds,
             typename Ends,
             typename Func,
             typename... Indices >
   __cuda_callable__
   void operator()( const Begins& begins,
                    const SkipBegins& skipBegins,
                    const SkipEnds& skipEnds,
                    const Ends& ends,
                    std::size_t level,
                    Func f,
                    Indices&&... indices )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(),
                     "wrong begins or ends" );

      SequentialBoundaryExecutor_inner< Permutation, IndexTag< LevelTag::value + 1 > > exec;
      const auto begin = begins.template getSize< get< LevelTag::value >( Permutation{} ) >();
      const auto skipBegin = skipBegins.template getSize< get< LevelTag::value >( Permutation{} ) >();
      const auto skipEnd = skipEnds.template getSize< get< LevelTag::value >( Permutation{} ) >();
      const auto end = ends.template getSize< get< LevelTag::value >( Permutation{} ) >();
      if( level == LevelTag::value ) {
         for( auto i = begin; i < skipBegin; i++ )
            exec( begins, skipBegins, skipEnds, ends, level, f, std::forward< Indices >( indices )..., i );
         for( auto i = skipEnd; i < end; i++ )
            exec( begins, skipBegins, skipEnds, ends, level, f, std::forward< Indices >( indices )..., i );
      }
      else if( level > LevelTag::value ) {
         for( auto i = skipBegin; i < skipEnd; i++ )
            exec( begins, skipBegins, skipEnds, ends, level, f, std::forward< Indices >( indices )..., i );
      }
      else {
         for( auto i = begin; i < end; i++ )
            exec( begins, skipBegins, skipEnds, ends, level, f, std::forward< Indices >( indices )..., i );
      }
   }
};

template< typename Permutation >
struct SequentialBoundaryExecutor_inner< Permutation, IndexTag< Permutation::size() - 1 > >
{
   template< typename Begins,
             typename SkipBegins,
             typename SkipEnds,
             typename Ends,
             typename Func,
             typename... Indices >
   __cuda_callable__
   void operator()( const Begins& begins,
                    const SkipBegins& skipBegins,
                    const SkipEnds& skipEnds,
                    const Ends& ends,
                    std::size_t level,
                    Func f,
                    Indices&&... indices )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(),
                     "wrong begins or ends" );
      static_assert( sizeof...(indices) == Begins::getDimension() - 1,
                     "invalid number of indices in the final step of the SequentialBoundaryExecutor" );

      using LevelTag = IndexTag< Permutation::size() - 1 >;

      const auto begin = begins.template getSize< get< LevelTag::value >( Permutation{} ) >();
      const auto skipBegin = skipBegins.template getSize< get< LevelTag::value >( Permutation{} ) >();
      const auto skipEnd = skipEnds.template getSize< get< LevelTag::value >( Permutation{} ) >();
      const auto end = ends.template getSize< get< LevelTag::value >( Permutation{} ) >();
      if( level == LevelTag::value ) {
         for( auto i = begin; i < skipBegin; i++ )
            call_with_unpermuted_arguments< Permutation >( f, std::forward< Indices >( indices )..., i );
         for( auto i = skipEnd; i < end; i++ )
            call_with_unpermuted_arguments< Permutation >( f, std::forward< Indices >( indices )..., i );
      }
      else if( level > LevelTag::value ) {
         for( auto i = skipBegin; i < skipEnd; i++ )
            call_with_unpermuted_arguments< Permutation >( f, std::forward< Indices >( indices )..., i );
      }
      else {
         for( auto i = begin; i < end; i++ )
            call_with_unpermuted_arguments< Permutation >( f, std::forward< Indices >( indices )..., i );
      }
   }
};

template< typename Permutation,
          std::size_t dim = Permutation::size() >
struct SequentialBoundaryExecutor
{
   template< typename Begins,
             typename SkipBegins,
             typename SkipEnds,
             typename Ends,
             typename Func >
   __cuda_callable__
   void operator()( const Begins& begins,
                    const SkipBegins& skipBegins,
                    const SkipEnds& skipEnds,
                    const Ends& ends,
                    Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(),
                     "wrong begins or ends" );

      SequentialBoundaryExecutor_inner< Permutation > exec;
      for( std::size_t level = 0; level < Permutation::size(); level++ )
         exec( begins, skipBegins, skipEnds, ends, level, f );
   }
};

template< typename Permutation >
struct SequentialBoundaryExecutor< Permutation, 0 >
{
   template< typename Begins,
             typename SkipBegins,
             typename SkipEnds,
             typename Ends,
             typename Func >
   __cuda_callable__
   void operator()( const Begins& begins,
                    const SkipBegins& skipBegins,
                    const SkipEnds& skipEnds,
                    const Ends& ends,
                    Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(),
                     "wrong begins or ends" );

      const auto begin = begins.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipBegin = skipBegins.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipEnd = skipEnds.template getSize< get< 0 >( Permutation{} ) >();
      const auto end = ends.template getSize< get< 0 >( Permutation{} ) >();
      for( auto i = begin; i < skipBegin; i++ )
         f( i );
      for( auto i = skipEnd; i < end; i++ )
         f( i );
   }
};


template< typename Permutation,
          typename Device,
          typename DimTag = IndexTag< Permutation::size() > >
struct ParallelBoundaryExecutor
{
   template< typename Begins,
             typename SkipBegins,
             typename SkipEnds,
             typename Ends,
             typename Func >
   void operator()( const Begins& begins,
                    const SkipBegins& skipBegins,
                    const SkipEnds& skipEnds,
                    const Ends& ends,
                    Func f )
   {
      static_assert( Permutation::size() <= 3, "ParallelBoundaryExecutor is implemented only for 1D, 2D, and 3D." );
   }
};

template< typename Permutation,
          typename Device >
struct ParallelBoundaryExecutor< Permutation, Device, IndexTag< 3 > >
{
   template< typename Begins,
             typename SkipBegins,
             typename SkipEnds,
             typename Ends,
             typename Func >
   void operator()( const Begins& begins,
                    const SkipBegins& skipBegins,
                    const SkipEnds& skipEnds,
                    const Ends& ends,
                    Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(),
                     "wrong begins or ends" );

      using Index = typename Ends::IndexType;

      auto kernel = [=] __cuda_callable__ ( Index i2, Index i1, Index i0 )
      {
         call_with_unpermuted_arguments< Permutation >( f, i0, i1, i2 );
      };

      const auto begin0 = begins.template getSize< get< 0 >( Permutation{} ) >();
      const auto begin1 = begins.template getSize< get< 1 >( Permutation{} ) >();
      const auto begin2 = begins.template getSize< get< 2 >( Permutation{} ) >();
      const auto skipBegin0 = skipBegins.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipBegin1 = skipBegins.template getSize< get< 1 >( Permutation{} ) >();
      const auto skipBegin2 = skipBegins.template getSize< get< 2 >( Permutation{} ) >();
      const auto skipEnd0 = skipEnds.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipEnd1 = skipEnds.template getSize< get< 1 >( Permutation{} ) >();
      const auto skipEnd2 = skipEnds.template getSize< get< 2 >( Permutation{} ) >();
      const auto end0 = ends.template getSize< get< 0 >( Permutation{} ) >();
      const auto end1 = ends.template getSize< get< 1 >( Permutation{} ) >();
      const auto end2 = ends.template getSize< get< 2 >( Permutation{} ) >();

      ParallelFor3D< Device >::exec( begin2,     begin1,     begin0,   skipBegin2, end1,       end0,       kernel );
      ParallelFor3D< Device >::exec( skipEnd2,   begin1,     begin0,   end2,       end1,       end0,       kernel );
      ParallelFor3D< Device >::exec( skipBegin2, begin1,     begin0,   skipEnd2,   skipBegin1, end0,       kernel );
      ParallelFor3D< Device >::exec( skipBegin2, skipEnd1,   begin0,   skipEnd2,   end1,       end0,       kernel );
      ParallelFor3D< Device >::exec( skipBegin2, skipBegin1, begin0,   skipEnd2,   skipEnd1,   skipBegin0, kernel );
      ParallelFor3D< Device >::exec( skipBegin2, skipBegin1, skipEnd0, skipEnd2,   skipEnd1,   end0,       kernel );
   }
};

template< typename Permutation,
          typename Device >
struct ParallelBoundaryExecutor< Permutation, Device, IndexTag< 2 > >
{
   template< typename Begins,
             typename SkipBegins,
             typename SkipEnds,
             typename Ends,
             typename Func >
   void operator()( const Begins& begins,
                    const SkipBegins& skipBegins,
                    const SkipEnds& skipEnds,
                    const Ends& ends,
                    Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(),
                     "wrong begins or ends" );

      using Index = typename Ends::IndexType;

      auto kernel = [=] __cuda_callable__ ( Index i1, Index i0 )
      {
         call_with_unpermuted_arguments< Permutation >( f, i0, i1 );
      };

      const auto begin0 = begins.template getSize< get< 0 >( Permutation{} ) >();
      const auto begin1 = begins.template getSize< get< 1 >( Permutation{} ) >();
      const auto skipBegin0 = skipBegins.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipBegin1 = skipBegins.template getSize< get< 1 >( Permutation{} ) >();
      const auto skipEnd0 = skipEnds.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipEnd1 = skipEnds.template getSize< get< 1 >( Permutation{} ) >();
      const auto end0 = ends.template getSize< get< 0 >( Permutation{} ) >();
      const auto end1 = ends.template getSize< get< 1 >( Permutation{} ) >();

      ParallelFor2D< Device >::exec( begin1,     begin0,   skipBegin1, end0,       kernel );
      ParallelFor2D< Device >::exec( skipEnd1,   begin0,   end1,       end0,       kernel );
      ParallelFor2D< Device >::exec( skipBegin1, begin0,   skipEnd1,   skipBegin0, kernel );
      ParallelFor2D< Device >::exec( skipBegin1, skipEnd0, skipEnd1,   end0,       kernel );
   }
};

template< typename Permutation,
          typename Device >
struct ParallelBoundaryExecutor< Permutation, Device, IndexTag< 1 > >
{
   template< typename Begins,
             typename SkipBegins,
             typename SkipEnds,
             typename Ends,
             typename Func >
   void operator()( const Begins& begins,
                    const SkipBegins& skipBegins,
                    const SkipEnds& skipEnds,
                    const Ends& ends,
                    Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(),
                     "wrong begins or ends" );

      const auto begin = begins.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipBegin = skipBegins.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipEnd = skipEnds.template getSize< get< 0 >( Permutation{} ) >();
      const auto end = ends.template getSize< get< 0 >( Permutation{} ) >();

      ParallelFor< Device >::exec( begin, skipBegin, f );
      ParallelFor< Device >::exec( skipEnd, end, f );
   }
};


// Device may be void which stands for StaticNDArray
template< typename Permutation,
          typename Device >
struct BoundaryExecutorDispatcher
{
   template< typename Begins,
             typename SkipBegins,
             typename SkipEnds,
             typename Ends,
             typename Func >
   void operator()( const Begins& begins,
                    const SkipBegins& skipBegins,
                    const SkipEnds& skipEnds,
                    const Ends& ends,
                    Func f )
   {
      SequentialBoundaryExecutor< Permutation >()( begins, skipBegins, skipEnds, ends, f );
   }
};

template< typename Permutation >
struct BoundaryExecutorDispatcher< Permutation, Devices::Host >
{
   template< typename Begins,
             typename SkipBegins,
             typename SkipEnds,
             typename Ends,
             typename Func >
   void operator()( const Begins& begins,
                    const SkipBegins& skipBegins,
                    const SkipEnds& skipEnds,
                    const Ends& ends,
                    Func f )
   {
      if( Devices::Host::isOMPEnabled() && Devices::Host::getMaxThreadsCount() > 1 )
         ParallelBoundaryExecutor< Permutation, Devices::Host >()( begins, skipBegins, skipEnds, ends, f );
      else
         SequentialBoundaryExecutor< Permutation >()( begins, skipBegins, skipEnds, ends, f );
   }
};

template< typename Permutation >
struct BoundaryExecutorDispatcher< Permutation, Devices::Cuda >
{
   template< typename Begins,
             typename SkipBegins,
             typename SkipEnds,
             typename Ends,
             typename Func >
   void operator()( const Begins& begins,
                    const SkipBegins& skipBegins,
                    const SkipEnds& skipEnds,
                    const Ends& ends,
                    Func f )
   {
      ParallelBoundaryExecutor< Permutation, Devices::Cuda >()( begins, skipBegins, skipEnds, ends, f );
   }
};

} // namespace __ndarray_impl
} // namespace Containers
} // namespace TNL
