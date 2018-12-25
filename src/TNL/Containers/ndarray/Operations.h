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

template< typename Array,
          typename LevelTag = IndexTag< 0 > >
struct SequentialExecutor
{
   template< typename Func,
             typename... Indices >
   __cuda_callable__
   void operator()( const Array& array, Func f, Indices&&... indices )
   {
      SequentialExecutor< Array, IndexTag< LevelTag::value + 1 > > exec;
      const auto size = array.template getSize< get< LevelTag::value >( typename Array::PermutationType{} ) >();
      for( typename Array::IndexType i = 0; i < size; i++ )
         exec( array, f, std::forward< Indices >( indices )..., i );
   }
};

template< typename Array >
struct SequentialExecutor< Array, IndexTag< Array::getDimension() - 1 > >
{
   template< typename Func,
             typename... Indices >
   __cuda_callable__
   void operator()( const Array& array, Func f, Indices&&... indices )
   {
      static_assert( sizeof...(indices) == Array::getDimension() - 1,
                     "invalid number of indices in the final step of the SequentialExecutor" );

      const auto size = array.template getSize< get< Array::getDimension() - 1 >( typename Array::PermutationType{} ) >();
      for( typename Array::IndexType i = 0; i < size; i++ )
         call_with_permuted_arguments< typename Array::PermutationType >( f, std::forward< Indices >( indices )..., i );
   }
};


template< typename Array,
          typename LevelTag = IndexTag< Array::getDimension() - 1 > >
struct SequentialExecutorRTL
{
   template< typename Func,
             typename... Indices >
   __cuda_callable__
   void operator()( const Array& array, Func f, Indices&&... indices )
   {
      SequentialExecutorRTL< Array, IndexTag< LevelTag::value - 1 > > exec;
      const auto size = array.template getSize< get< LevelTag::value >( typename Array::PermutationType{} ) >();
      for( typename Array::IndexType i = 0; i < size; i++ )
         exec( array, f, i, std::forward< Indices >( indices )... );
   }
};

template< typename Array >
struct SequentialExecutorRTL< Array, IndexTag< 0 > >
{
   template< typename Func,
             typename... Indices >
   __cuda_callable__
   void operator()( const Array& array, Func f, Indices&&... indices )
   {
      static_assert( sizeof...(indices) == Array::getDimension() - 1,
                     "invalid number of indices in the final step of the SequentialExecutor" );

      const auto size = array.template getSize< get< 0 >( typename Array::PermutationType{} ) >();
      for( typename Array::IndexType i = 0; i < size; i++ )
         call_with_permuted_arguments< typename Array::PermutationType >( f, i, std::forward< Indices >( indices )... );
   }
};


template< typename Array,
          typename DimTag = IndexTag< Array::getDimension() > >
struct OpenMPExecutor
{
   template< typename Func >
   void operator()( const Array& array, Func f )
   {
      SequentialExecutor< Array, IndexTag< 3 > > exec;

      const auto size0 = array.template getSize< get< 0 >( typename Array::PermutationType{} ) >();
      const auto size1 = array.template getSize< get< 1 >( typename Array::PermutationType{} ) >();
      const auto size2 = array.template getSize< get< 2 >( typename Array::PermutationType{} ) >();

      #ifdef HAVE_OPENMP
      #pragma omp parallel for collapse(3)
      #endif
      for( typename Array::IndexType i0 = 0; i0 < size0; i0++ )
      for( typename Array::IndexType i1 = 0; i1 < size1; i1++ )
      for( typename Array::IndexType i2 = 0; i2 < size2; i2++ )
         exec( array, f, i0, i1, i2 );
   }
};

template< typename Array >
struct OpenMPExecutor< Array, IndexTag< 3 > >
{
   template< typename Func >
   void operator()( const Array& array, Func f )
   {
      const auto size0 = array.template getSize< get< 0 >( typename Array::PermutationType{} ) >();
      const auto size1 = array.template getSize< get< 1 >( typename Array::PermutationType{} ) >();
      const auto size2 = array.template getSize< get< 2 >( typename Array::PermutationType{} ) >();

      #ifdef HAVE_OPENMP
      #pragma omp parallel for collapse(2)
      #endif
      for( typename Array::IndexType i0 = 0; i0 < size0; i0++ )
      for( typename Array::IndexType i1 = 0; i1 < size1; i1++ )
      for( typename Array::IndexType i2 = 0; i2 < size2; i2++ )
         call_with_permuted_arguments< typename Array::PermutationType >( f, i0, i1, i2 );
   }
};

template< typename Array >
struct OpenMPExecutor< Array, IndexTag< 2 > >
{
   template< typename Func >
   void operator()( const Array& array, Func f )
   {
      const auto size0 = array.template getSize< get< 0 >( typename Array::PermutationType{} ) >();
      const auto size1 = array.template getSize< get< 1 >( typename Array::PermutationType{} ) >();

      #ifdef HAVE_OPENMP
      #pragma omp parallel for
      #endif
      for( typename Array::IndexType i0 = 0; i0 < size0; i0++ )
      for( typename Array::IndexType i1 = 0; i1 < size1; i1++ )
         call_with_permuted_arguments< typename Array::PermutationType >( f, i0, i1 );
   }
};

template< typename Array >
struct OpenMPExecutor< Array, IndexTag< 1 > >
{
   template< typename Func >
   void operator()( const Array& array, Func f )
   {
      const auto size0 = array.template getSize< get< 0 >( typename Array::PermutationType{} ) >();

      #ifdef HAVE_OPENMP
      #pragma omp parallel for
      #endif
      for( typename Array::IndexType i0 = 0; i0 < size0; i0++ )
         call_with_permuted_arguments< typename Array::PermutationType >( f, i0 );
   }
};


template< typename Array,
          typename DimTag = IndexTag< Array::getDimension() > >
struct CudaExecutor
{
   template< typename Func >
   void operator()( const Array& array, Func f )
   {
      using Index = typename Array::IndexType;

      auto kernel = [=] __cuda_callable__ ( Index i2, Index i1, Index i0 )
      {
         SequentialExecutorRTL< Array, IndexTag< Array::getDimension() - 4 > > exec;
         exec( array, f, i0, i1, i2 );
      };

      const Index size0 = array.template getSize< get< Array::getDimension() - 3 >( typename Array::PermutationType{} ) >();
      const Index size1 = array.template getSize< get< Array::getDimension() - 2 >( typename Array::PermutationType{} ) >();
      const Index size2 = array.template getSize< get< Array::getDimension() - 1 >( typename Array::PermutationType{} ) >();
      ParallelFor3D< Devices::Cuda >::exec( (Index) 0, (Index) 0, (Index) 0, size2, size1, size0, kernel );
   }
};

template< typename Array >
struct CudaExecutor< Array, IndexTag< 3 > >
{
   template< typename Func >
   void operator()( const Array& array, Func f )
   {
      using Index = typename Array::IndexType;

      auto kernel = [=] __cuda_callable__ ( Index i2, Index i1, Index i0 )
      {
         call_with_permuted_arguments< typename Array::PermutationType >( f, i0, i1, i2 );
      };

      const Index size0 = array.template getSize< get< 0 >( typename Array::PermutationType{} ) >();
      const Index size1 = array.template getSize< get< 1 >( typename Array::PermutationType{} ) >();
      const Index size2 = array.template getSize< get< 2 >( typename Array::PermutationType{} ) >();
      ParallelFor3D< Devices::Cuda >::exec( (Index) 0, (Index) 0, (Index) 0, size2, size1, size0, kernel );
   }
};

template< typename Array >
struct CudaExecutor< Array, IndexTag< 2 > >
{
   template< typename Func >
   void operator()( const Array& array, Func f )
   {
      using Index = typename Array::IndexType;

      auto kernel = [=] __cuda_callable__ ( Index i1, Index i0 )
      {
         call_with_permuted_arguments< typename Array::PermutationType >( f, i0, i1 );
      };

      const Index size0 = array.template getSize< get< 0 >( typename Array::PermutationType{} ) >();
      const Index size1 = array.template getSize< get< 1 >( typename Array::PermutationType{} ) >();
      ParallelFor2D< Devices::Cuda >::exec( (Index) 0, (Index) 0, size1, size0, kernel );
   }
};

template< typename Array >
struct CudaExecutor< Array, IndexTag< 1 > >
{
   template< typename Func >
   void operator()( const Array& array, Func f )
   {
      using Index = typename Array::IndexType;

      auto kernel = [=] __cuda_callable__ ( Index i )
      {
         call_with_permuted_arguments< typename Array::PermutationType >( f, i );
      };

      const Index size = array.template getSize< get< 0 >( typename Array::PermutationType{} ) >();
      ParallelFor< Devices::Cuda >::exec( (Index) 0, size, kernel );
   }
};


// Device may be void which stands for StaticNDArray
template< typename Array, typename Device = typename Array::DeviceType >
struct ExecutorDispatcher
{
   template< typename Func >
   void operator()( const Array& array, Func f )
   {
      SequentialExecutor< Array >()( array, f );
   }
};

template< typename Array >
struct ExecutorDispatcher< Array, Devices::Host >
{
   template< typename Func >
   void operator()( const Array& array, Func f )
   {
      if( Devices::Host::isOMPEnabled() && Devices::Host::getMaxThreadsCount() > 1 )
         OpenMPExecutor< Array >()( array, f );
      else
         SequentialExecutor< Array >()( array, f );
   }
};

template< typename Array >
struct ExecutorDispatcher< Array, Devices::Cuda >
{
   template< typename Func >
   void operator()( const Array& array, Func f )
   {
      CudaExecutor< Array >()( array, f );
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

   // From here on, the output array is used only for getting the sizes,
   // the writing of the result is done inside the wrapper.
   ExecutorDispatcher< Output >()( output, wrapper );
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

   // From here on, the output array is used only for getting the sizes,
   // the writing of the result is done inside the wrapper.
   ExecutorDispatcher< Output >()( output, wrapper );
}

template< typename Output,
          typename Func,
          typename Input1 >
void nd_map_view( Output output, Func f, const Input1 input1 )
{
   static_assert( all_elements_equal_to_value( Output::getDimension(), {Input1::getDimension()} ),
                  "all arrays must be of the same dimension" );

   nvcc_map_helper_1< Output, Func, Input1 > wrapper( output, f, input1 );

   // From here on, the output array is used only for getting the sizes,
   // the writing of the result is done inside the wrapper.
   ExecutorDispatcher< Output >()( output, wrapper );
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

   // From here on, the output array is used only for getting the sizes,
   // the writing of the result is done inside the wrapper.
   ExecutorDispatcher< Output >()( output, wrapper );
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

   // From here on, the output array is used only for getting the sizes,
   // the writing of the result is done inside the wrapper.
   ExecutorDispatcher< Output >()( output, wrapper );
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
