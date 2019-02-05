/***************************************************************************
                          Operations.h  -  description
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

namespace TNL {
namespace Containers {

namespace __ndarray_impl {

template< typename Permutation,
          typename LevelTag = IndexTag< 0 > >
struct SequentialExecutor
{
   template< typename SizesHolder,
             typename Func,
             typename... Indices >
   __cuda_callable__
   void operator()( const SizesHolder& sizes, Func f, Indices&&... indices )
   {
      SequentialExecutor< Permutation, IndexTag< LevelTag::value + 1 > > exec;
      const auto size = sizes.template getSize< get< LevelTag::value >( Permutation{} ) >();
      for( typename SizesHolder::IndexType i = 0; i < size; i++ )
         exec( sizes, f, std::forward< Indices >( indices )..., i );
   }
};

template< typename Permutation >
struct SequentialExecutor< Permutation, IndexTag< Permutation::size() - 1 > >
{
   template< typename SizesHolder,
             typename Func,
             typename... Indices >
   __cuda_callable__
   void operator()( const SizesHolder& sizes, Func f, Indices&&... indices )
   {
      static_assert( sizeof...(indices) == SizesHolder::getDimension() - 1,
                     "invalid number of indices in the final step of the SequentialExecutor" );

      const auto size = sizes.template getSize< get< SizesHolder::getDimension() - 1 >( Permutation{} ) >();
      for( typename SizesHolder::IndexType i = 0; i < size; i++ )
         call_with_permuted_arguments< Permutation >( f, std::forward< Indices >( indices )..., i );
   }
};


template< typename Permutation,
          typename LevelTag = IndexTag< Permutation::size() - 1 > >
struct SequentialExecutorRTL
{
   template< typename SizesHolder,
             typename Func,
             typename... Indices >
   __cuda_callable__
   void operator()( const SizesHolder& sizes, Func f, Indices&&... indices )
   {
      SequentialExecutorRTL< Permutation, IndexTag< LevelTag::value - 1 > > exec;
      const auto size = sizes.template getSize< get< LevelTag::value >( Permutation{} ) >();
      for( typename SizesHolder::IndexType i = 0; i < size; i++ )
         exec( sizes, f, i, std::forward< Indices >( indices )... );
   }
};

template< typename Permutation >
struct SequentialExecutorRTL< Permutation, IndexTag< 0 > >
{
   template< typename SizesHolder,
             typename Func,
             typename... Indices >
   __cuda_callable__
   void operator()( const SizesHolder& sizes, Func f, Indices&&... indices )
   {
      static_assert( sizeof...(indices) == SizesHolder::getDimension() - 1,
                     "invalid number of indices in the final step of the SequentialExecutor" );

      const auto size = sizes.template getSize< get< 0 >( Permutation{} ) >();
      for( typename SizesHolder::IndexType i = 0; i < size; i++ )
         call_with_permuted_arguments< Permutation >( f, i, std::forward< Indices >( indices )... );
   }
};


template< typename Permutation,
          typename Device >
struct ParallelExecutorDeviceDispatch
{
   template< typename SizesHolder, typename Func >
   void operator()( const SizesHolder& sizes, Func f )
   {
      using Index = typename SizesHolder::IndexType;

      auto kernel = [=] ( Index i2, Index i1, Index i0 )
      {
         SequentialExecutor< Permutation, IndexTag< 3 > > exec;
         exec( sizes, f, i0, i1, i2 );
      };

      const Index size0 = sizes.template getSize< get< 0 >( Permutation{} ) >();
      const Index size1 = sizes.template getSize< get< 1 >( Permutation{} ) >();
      const Index size2 = sizes.template getSize< get< 2 >( Permutation{} ) >();
      ParallelFor3D< Device >::exec( (Index) 0, (Index) 0, (Index) 0, size2, size1, size0, kernel );
   }
};

template< typename Permutation >
struct ParallelExecutorDeviceDispatch< Permutation, Devices::Cuda >
{
   template< typename SizesHolder, typename Func >
   void operator()( const SizesHolder& sizes, Func f )
   {
      using Index = typename SizesHolder::IndexType;

      auto kernel = [=] __cuda_callable__ ( Index i2, Index i1, Index i0 )
      {
         SequentialExecutorRTL< Permutation, IndexTag< SizesHolder::getDimension() - 4 > > exec;
         exec( sizes, f, i0, i1, i2 );
      };

      const Index size0 = sizes.template getSize< get< SizesHolder::getDimension() - 3 >( Permutation{} ) >();
      const Index size1 = sizes.template getSize< get< SizesHolder::getDimension() - 2 >( Permutation{} ) >();
      const Index size2 = sizes.template getSize< get< SizesHolder::getDimension() - 1 >( Permutation{} ) >();
      ParallelFor3D< Devices::Cuda >::exec( (Index) 0, (Index) 0, (Index) 0, size2, size1, size0, kernel );
   }
};

template< typename Permutation,
          typename Device,
          typename DimTag = IndexTag< Permutation::size() > >
struct ParallelExecutor
{
   template< typename SizesHolder, typename Func >
   void operator()( const SizesHolder& sizes, Func f )
   {
      ParallelExecutorDeviceDispatch< Permutation, Device > dispatch;
      dispatch( sizes, f );
   }
};

template< typename Permutation,
          typename Device >
struct ParallelExecutor< Permutation, Device, IndexTag< 3 > >
{
   template< typename SizesHolder, typename Func >
   void operator()( const SizesHolder& sizes, Func f )
   {
      using Index = typename SizesHolder::IndexType;

      auto kernel = [=] __cuda_callable__ ( Index i2, Index i1, Index i0 )
      {
         call_with_permuted_arguments< Permutation >( f, i0, i1, i2 );
      };

      const Index size0 = sizes.template getSize< get< 0 >( Permutation{} ) >();
      const Index size1 = sizes.template getSize< get< 1 >( Permutation{} ) >();
      const Index size2 = sizes.template getSize< get< 2 >( Permutation{} ) >();
      ParallelFor3D< Device >::exec( (Index) 0, (Index) 0, (Index) 0, size2, size1, size0, kernel );
   }
};

template< typename Permutation,
          typename Device >
struct ParallelExecutor< Permutation, Device, IndexTag< 2 > >
{
   template< typename SizesHolder, typename Func >
   void operator()( const SizesHolder& sizes, Func f )
   {
      using Index = typename SizesHolder::IndexType;

      auto kernel = [=] __cuda_callable__ ( Index i1, Index i0 )
      {
         call_with_permuted_arguments< Permutation >( f, i0, i1 );
      };

      const Index size0 = sizes.template getSize< get< 0 >( Permutation{} ) >();
      const Index size1 = sizes.template getSize< get< 1 >( Permutation{} ) >();
      ParallelFor2D< Device >::exec( (Index) 0, (Index) 0, size1, size0, kernel );
   }
};

template< typename Permutation,
          typename Device >
struct ParallelExecutor< Permutation, Device, IndexTag< 1 > >
{
   template< typename SizesHolder, typename Func >
   void operator()( const SizesHolder& sizes, Func f )
   {
      using Index = typename SizesHolder::IndexType;

      auto kernel = [=] __cuda_callable__ ( Index i )
      {
         call_with_permuted_arguments< Permutation >( f, i );
      };

      const Index size = sizes.template getSize< get< 0 >( Permutation{} ) >();
      ParallelFor< Device >::exec( (Index) 0, size, kernel );
   }
};


// Device may be void which stands for StaticNDArray
template< typename Permutation,
          typename Device >
struct ExecutorDispatcher
{
   template< typename SizesHolder, typename Func >
   void operator()( const SizesHolder& sizes, Func f )
   {
      SequentialExecutor< Permutation >()( sizes, f );
   }
};

template< typename Permutation >
struct ExecutorDispatcher< Permutation, Devices::Host >
{
   template< typename SizesHolder, typename Func >
   void operator()( const SizesHolder& sizes, Func f )
   {
      if( Devices::Host::isOMPEnabled() && Devices::Host::getMaxThreadsCount() > 1 )
         ParallelExecutor< Permutation, Devices::Host >()( sizes, f );
      else
         SequentialExecutor< Permutation >()( sizes, f );
   }
};

template< typename Permutation >
struct ExecutorDispatcher< Permutation, Devices::Cuda >
{
   template< typename SizesHolder, typename Func >
   void operator()( const SizesHolder& sizes, Func f )
   {
      ParallelExecutor< Permutation, Devices::Cuda >()( sizes, f );
   }
};

#ifndef __NVCC__
template< typename Output,
          typename Func,
          typename... Input >
void nd_map_view( Output output, Func f, const Input... input )
{
   static_assert( all_elements_equal_to_value( Output::getDimension(), {Input::getDimension()...} ),
                  "all arrays must be of the same dimension" );

   // without mutable, the operator() would be const so output would be const as well
   // https://stackoverflow.com/a/2835645/4180822
   auto wrapper = [=] __cuda_callable__ ( auto... indices ) mutable {
      static_assert( sizeof...( indices ) == Output::getDimension(),
                     "wrong number of indices passed to the wrapper lambda function" );
      output( indices... ) = f( input( indices... )... );
   };

   ExecutorDispatcher< typename Output::PermutationType, typename Output::DeviceType > dispatch;
   dispatch( output.getSizes(), wrapper );
}

#else

   template< typename Output,
             typename Func >
   struct nvcc_map_helper_0
   {
      Output output;
      Func f;

      nvcc_map_helper_0( Output o, Func f ) : output(o), f(f) {}

      template< typename... Ts >
      __cuda_callable__
      void operator()( Ts... indices )
      {
         static_assert( sizeof...( indices ) == Output::getDimension(),
                        "wrong number of indices passed to the wrapper operator() function" );
         output( indices... ) = f();
      }
   };

   template< typename Output,
             typename Func,
             typename Input1 >
   struct nvcc_map_helper_1
   {
      Output output;
      Func f;
      Input1 input1;

      nvcc_map_helper_1( Output o, Func f, Input1 i1 ) : output(o), f(f), input1(i1) {}

      template< typename... Ts >
      __cuda_callable__
      void operator()( Ts... indices )
      {
         static_assert( sizeof...( indices ) == Output::getDimension(),
                        "wrong number of indices passed to the wrapper operator() function" );
         output( indices... ) = f( input1( indices... ) );
      }
   };

   template< typename Output,
             typename Func,
             typename Input1,
             typename Input2 >
   struct nvcc_map_helper_2
   {
      Output output;
      Func f;
      Input1 input1;
      Input2 input2;

      nvcc_map_helper_2( Output o, Func f, Input1 i1, Input2 i2 ) : output(o), f(f), input1(i1), input2(i2) {}

      template< typename... Ts >
      __cuda_callable__
      void operator()( Ts... indices )
      {
         static_assert( sizeof...( indices ) == Output::getDimension(),
                        "wrong number of indices passed to the wrapper operator() function" );
         output( indices... ) = f( input1( indices... ), input2( indices... ) );
      }
   };

   template< typename Output,
             typename Func,
             typename Input1,
             typename Input2,
             typename Input3 >
   struct nvcc_map_helper_3
   {
      Output output;
      Func f;
      Input1 input1;
      Input2 input2;
      Input3 input3;

      nvcc_map_helper_3( Output o, Func f, Input1 i1, Input2 i2, Input3 i3 ) : output(o), f(f), input1(i1), input2(i2), input3(i3) {}

      template< typename... Ts >
      __cuda_callable__
      void operator()( Ts... indices )
      {
         static_assert( sizeof...( indices ) == Output::getDimension(),
                        "wrong number of indices passed to the wrapper operator() function" );
         output( indices... ) = f( input1( indices... ), input2( indices... ), input3( indices... ) );
      }
   };

template< typename Output,
          typename Func >
void nd_map_view( Output output, Func f )
{
   nvcc_map_helper_0< Output, Func > wrapper( output, f );
   ExecutorDispatcher< typename Output::PermutationType, typename Output::DeviceType > dispatch;
   dispatch( output.getSizes(), wrapper );
}

template< typename Output,
          typename Func,
          typename Input1 >
void nd_map_view( Output output, Func f, const Input1 input1 )
{
   static_assert( all_elements_equal_to_value( Output::getDimension(), {Input1::getDimension()} ),
                  "all arrays must be of the same dimension" );

   nvcc_map_helper_1< Output, Func, Input1 > wrapper( output, f, input1 );
   ExecutorDispatcher< typename Output::PermutationType, typename Output::DeviceType > dispatch;
   dispatch( output.getSizes(), wrapper );
}

template< typename Output,
          typename Func,
          typename Input1,
          typename Input2 >
void nd_map_view( Output output, Func f, const Input1 input1, const Input2 input2 )
{
   static_assert( all_elements_equal_to_value( Output::getDimension(), {Input1::getDimension(), Input2::getDimension()} ),
                  "all arrays must be of the same dimension" );

   nvcc_map_helper_2< Output, Func, Input1, Input2 > wrapper( output, f, input1, input2 );
   ExecutorDispatcher< typename Output::PermutationType, typename Output::DeviceType > dispatch;
   dispatch( output.getSizes(), wrapper );
}

template< typename Output,
          typename Func,
          typename Input1,
          typename Input2,
          typename Input3 >
void nd_map_view( Output output, Func f, const Input1 input1, const Input2 input2, const Input3 input3 )
{
   static_assert( all_elements_equal_to_value( Output::getDimension(), {Input1::getDimension(), Input2::getDimension(), Input3::getDimension()} ),
                  "all arrays must be of the same dimension" );

   nvcc_map_helper_3< Output, Func, Input1, Input2, Input3 > wrapper( output, f, input1, input2, input3 );
   ExecutorDispatcher< typename Output::PermutationType, typename Output::DeviceType > dispatch;
   dispatch( output.getSizes(), wrapper );
}

#endif

} // namespace __ndarray_impl


// f must be an N-ary function, where N is the dimension of the output and input arrays:
//      output( i1, ..., iN ) = f( input1( i1, ..., iN ), ... inputM( i1, ..., iN ) )
template< typename Output,
          typename Func,
          typename... Input >
void nd_map( Output& output, Func f, const Input&... input )
{
   __ndarray_impl::nd_map_view( output.getView(), f, input.getConstView()... );
}

template< typename Output,
          typename Input >
void nd_assign( Output& output, const Input& input )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v ){ return v; }, input );
#else
   using value_type = typename Input::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type v ){ return v; }, input );
#endif
}

// Some mathematical functions, inspired by NumPy:
// https://docs.scipy.org/doc/numpy/reference/ufuncs.html#math-operations

template< typename Output,
          typename Input1,
          typename Input2 >
void nd_add( Output& output, const Input1& input1, const Input2& input2 )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v1, auto v2 ){ return v1 + v2; }, input1, input2 );
#else
   using value_type_1 = typename Input1::ValueType;
   using value_type_2 = typename Input2::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type_1 v1, value_type_2 v2 ){ return v1 + v2; }, input1, input2 );
#endif
}

template< typename Output,
          typename Input1,
          typename Input2 >
void nd_subtract( Output& output, const Input1& input1, const Input2& input2 )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v1, auto v2 ){ return v1 - v2; }, input1, input2 );
#else
   using value_type_1 = typename Input1::ValueType;
   using value_type_2 = typename Input2::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type_1 v1, value_type_2 v2 ){ return v1 - v2; }, input1, input2 );
#endif
}

template< typename Output,
          typename Input1,
          typename Input2 >
void nd_multiply( Output& output, const Input1& input1, const Input2& input2 )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v1, auto v2 ){ return v1 * v2; }, input1, input2 );
#else
   using value_type_1 = typename Input1::ValueType;
   using value_type_2 = typename Input2::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type_1 v1, value_type_2 v2 ){ return v1 * v2; }, input1, input2 );
#endif
}

template< typename Output,
          typename Input1,
          typename Input2 >
void nd_divide( Output& output, const Input1& input1, const Input2& input2 )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v1, auto v2 ){ return v1 / v2; }, input1, input2 );
#else
   using value_type_1 = typename Input1::ValueType;
   using value_type_2 = typename Input2::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type_1 v1, value_type_2 v2 ){ return v1 / v2; }, input1, input2 );
#endif
}

template< typename Output,
          typename Input1,
          typename Input2 >
void nd_maximum( Output& output, const Input1& input1, const Input2& input2 )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v1, auto v2 ){ return TNL::max( v1, v2 ); }, input1, input2 );
#else
   using value_type_1 = typename Input1::ValueType;
   using value_type_2 = typename Input2::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type_1 v1, value_type_2 v2 ){ return TNL::max( v1, v2 ); }, input1, input2 );
#endif
}

template< typename Output,
          typename Input1,
          typename Input2 >
void nd_minimum( Output& output, const Input1& input1, const Input2& input2 )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v1, auto v2 ){ return TNL::min( v1, v2 ); }, input1, input2 );
#else
   using value_type_1 = typename Input1::ValueType;
   using value_type_2 = typename Input2::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type_1 v1, value_type_2 v2 ){ return TNL::min( v1, v2 ); }, input1, input2 );
#endif
}

template< typename Output,
          typename Input >
void nd_absolute( Output& output, const Input& input )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v ){ return TNL::abs( v ); }, input );
#else
   using value_type = typename Input::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type v ){ return TNL::abs( v ); }, input );
#endif
}

template< typename Output,
          typename Input >
void nd_sign( Output& output, const Input& input )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v ){ return TNL::sign( v ); }, input );
#else
   using value_type = typename Input::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type v ){ return TNL::sign( v ); }, input );
#endif
}

template< typename Output,
          typename Input1,
          typename Input2 >
void nd_pow( Output& output, const Input1& input1, const Input2& input2 )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v1, auto v2 ){ return TNL::pow( v1, v2 ); }, input1, input2 );
#else
   using value_type_1 = typename Input1::ValueType;
   using value_type_2 = typename Input2::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type_1 v1, value_type_2 v2 ){ return TNL::pow( v1, v2 ); }, input1, input2 );
#endif
}

template< typename Output,
          typename Input >
void nd_sqrt( Output& output, const Input& input )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v ){ return TNL::sqrt( v ); }, input );
#else
   using value_type = typename Input::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type v ){ return TNL::sqrt( v ); }, input );
#endif
}

template< typename Output,
          typename Input >
void nd_square( Output& output, const Input& input )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v ){ return v*v; }, input );
#else
   using value_type = typename Input::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type v ){ return v*v; }, input );
#endif
}

} // namespace Containers
} // namespace TNL
