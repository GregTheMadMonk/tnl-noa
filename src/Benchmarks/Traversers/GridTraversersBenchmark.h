/***************************************************************************
                          GridTraversersBenchmark.h  -  description
                             -------------------
    begin                : Dec 19, 2018
    copyright            : (C) 2018 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber

#pragma once

#include <TNL/ParallelFor.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridEntityConfig.h>
#include <TNL/Meshes/Traverser.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Pointers/SharedPointer.h>
#include "cuda-kernels.h"

namespace TNL {
   namespace Benchmarks {
      namespace Traversers {

template< typename TraverserUserData >
class WriteOneEntitiesProcessor
{
   public:
      
      using MeshType = typename TraverserUserData::MeshType;
      using DeviceType = typename MeshType::DeviceType;

      template< typename GridEntity >
      __cuda_callable__
      static inline void processEntity( const MeshType& mesh,
                                        TraverserUserData& userData,
                                        const GridEntity& entity )
      {
         auto& u = userData.u.template modifyData< DeviceType >();
         u( entity ) = 1.0;
      }
};

template< typename MeshFunctionPointer >
class WriteOneUserData
{
   public:
      
      using MeshType = typename MeshFunctionPointer::ObjectType::MeshType;
      
      MeshFunctionPointer u;
};


template< int Dimension,
          typename Device,
          typename Real,
          typename Index >
class GridTraversersBenchmark{};

template< typename Device,
          typename Real,
          typename Index >
class GridTraversersBenchmark< 1, Device, Real, Index >
{
   public:
      
      using Vector = Containers::Vector< Real, Device, Index >;
      using Grid = Meshes::Grid< 1, Real, Device, Index >;
      using GridPointer = Pointers::SharedPointer< Grid >;
      using Coordinates = typename Grid::CoordinatesType;
      using MeshFunction = Functions::MeshFunction< Grid >;
      using MeshFunctionPointer = Pointers::SharedPointer< MeshFunction >;
      using Cell = typename Grid::template EntityType< 1, Meshes::GridEntityNoStencilStorage >;
      using Traverser = Meshes::Traverser< Grid, Cell >;
      using WriteOneTraverserUserDataType = WriteOneUserData< MeshFunctionPointer >;
      using WriteOneEntitiesProcessorType = WriteOneEntitiesProcessor< WriteOneTraverserUserDataType >;
      
      GridTraversersBenchmark( Index size )
      :v( size ), size( size ), grid( size ), u( grid )
      {
         userData.u = this->u;
         v_data = v.getData();
      }

      void reset()
      {
         v.setValue( 0.0 );
         u->getData().setValue( 0.0 );
      };

      void writeOneUsingPureC()
      {
         if( std::is_same< Device, Devices::Host >::value )
         {
            for( int i = 0; i < size; i++ )
               v_data[ i ] += 1.0;
         }
         else // Device == Devices::Cuda
         {
#ifdef HAVE_CUDA
            dim3 blockSize( 256 ), blocksCount, gridsCount;
            Devices::Cuda::setupThreads(
               blockSize,
               blocksCount,
               gridsCount,
               size );
            dim3 gridIdx;
            for( gridIdx.x = 0; gridIdx.x < gridsCount.x; gridIdx.x++ )
            {
               dim3 gridSize;
               Devices::Cuda::setupGrid(
                  blocksCount,
                  gridsCount,
                  gridIdx,
                  gridSize );
               fullGridTraverseKernel1D<<< gridSize, blockSize >>>( size, gridIdx, v_data );
            }
#endif
         }
      }

      void writeOneUsingParallelFor()
      {
         auto f = [] __cuda_callable__ ( Index i, Real* data )
         {
            data[ i ] = +1.0;
         };
         ParallelFor< Device >::exec( ( Index ) 0, size, f, v.getData() );
      }

      void writeOneUsingTraverser()
      {
         traverser.template processAllEntities< WriteOneTraverserUserDataType, WriteOneEntitiesProcessorType >
            ( grid, userData );
      }

      void traverseUsingPureC()
      {
         if( std::is_same< Device, Devices::Host >::value )
         {
            v_data[ 0 ] = 2;
            for( int i = 1; i < size - 1; i++ )
               v_data[ i ] = 1.0;
            v_data[ size - 1 ] =  2;
         }
         else // Device == Devices::Cuda
         {
#ifdef HAVE_CUDA
            dim3 blockSize( 256 ), blocksCount, gridsCount;
            Devices::Cuda::setupThreads(
               blockSize,
               blocksCount,
               gridsCount,
               size );
            dim3 gridIdx;
            for( gridIdx.x = 0; gridIdx.x < gridsCount.x; gridIdx.x++ )
            {
               dim3 gridSize;
               Devices::Cuda::setupGrid(
                  blocksCount,
                  gridsCount,
                  gridIdx,
                  gridSize );
               boundariesTraverseKernel1D<<< gridSize, blockSize >>>( size, gridIdx, v_data );
            }
            for( gridIdx.x = 0; gridIdx.x < gridsCount.x; gridIdx.x++ )
            {
               dim3 gridSize;
               Devices::Cuda::setupGrid(
                  blocksCount,
                  gridsCount,
                  gridIdx,
                  gridSize );
               interiorTraverseKernel1D<<< gridSize, blockSize >>>( size, gridIdx, v_data );
            }
#endif
         }
      }

      void traverseUsingTraverser()
      {
         // TODO !!!!!!!!!!!!!!!!!!!!!!
         traverser.template processAllEntities< WriteOneTraverserUserDataType, WriteOneEntitiesProcessorType >
            ( grid, userData );
      }

      protected:

         Index size;
         Vector v;
         Real* v_data;
         GridPointer grid;
         MeshFunctionPointer u;
         Traverser traverser;
         WriteOneTraverserUserDataType userData;
};


template< typename Device,
          typename Real,
          typename Index >
class GridTraversersBenchmark< 2, Device, Real, Index >
{
   public:
      
      using Vector = Containers::Vector< Real, Device, Index >;
      using Grid = Meshes::Grid< 2, Real, Device, Index >;
      using GridPointer = Pointers::SharedPointer< Grid >;
      using Coordinates = typename Grid::CoordinatesType;
      using MeshFunction = Functions::MeshFunction< Grid >;
      using MeshFunctionPointer = Pointers::SharedPointer< MeshFunction >;
      using Cell = typename Grid::template EntityType< 2, Meshes::GridEntityNoStencilStorage >;
      using Traverser = Meshes::Traverser< Grid, Cell >;
      using TraverserUserData = WriteOneUserData< MeshFunctionPointer >;
      using WriteOneTraverserUserDataType = WriteOneUserData< MeshFunctionPointer >;
      using WriteOneEntitiesProcessorType = WriteOneEntitiesProcessor< WriteOneTraverserUserDataType >;

      GridTraversersBenchmark( Index size )
      :size( size ), v( size * size ), grid( size, size ), u( grid )
      {
         userData.u = this->u;
         v_data = v.getData();
      }

      void reset()
      {
         v.setValue( 0.0 );
         u->getData().setValue( 0.0 );
      };

      void writeOneUsingPureC()
      {
         if( std::is_same< Device, Devices::Host >::value )
         {
            for( int i = 0; i < size; i++ )
               for( int j = 0; j < size; j++ )
                  v_data[ i * size + j ] = 1.0;
         }
         else // Device == Devices::Cuda
         {
#ifdef HAVE_CUDA
            dim3 blockSize( 256 ), blocksCount, gridsCount;
            Devices::Cuda::setupThreads(
               blockSize,
               blocksCount,
               gridsCount,
               size,
               size );
            dim3 gridIdx;
            for( gridIdx.y = 0; gridIdx.y < gridsCount.y; gridIdx.y++ )
               for( gridIdx.x = 0; gridIdx.x < gridsCount.x; gridIdx.x++ )
               {
                  dim3 gridSize;
                  Devices::Cuda::setupGrid(
                     blocksCount,
                     gridsCount,
                     gridIdx,
                     gridSize );
                  fullGridTraverseKernel2D<<< gridSize, blockSize >>>( size, gridIdx, v_data );
               }
#endif
         }
      }
      
      void writeOneUsingParallelFor()
      {
         Index _size = this->size;
         auto f = [=] __cuda_callable__ ( Index i, Index j,  Real* data )
         {
            data[ i * _size + j ] = 1.0;
         };
         
         ParallelFor2D< Device >::exec( ( Index ) 0,
                                        ( Index ) 0,
                                        this->size,
                                        this->size,
                                        f, v.getData() );
      }

      void writeOneUsingTraverser()
      {
         traverser.template processAllEntities< WriteOneTraverserUserDataType, WriteOneEntitiesProcessorType >
            ( grid, userData );
      }

      void traverseUsingPureC()
      {
         if( std::is_same< Device, Devices::Host >::value )
         {
            for( int i = 0; i < size; i++ )
            {
               v_data[ i * size ] = 2.0;
               v_data[ i * size + size - 1 ] = 2.0;
            }
            for( int j = 1; j < size - 1; j++ )
            {
               v_data[ j ] = 2.0;
               v_data[ ( size - 1 ) * size + j ] = 2.0;
            }

            for( int i = 1; i < size - 1; i++ )
               for( int j = 1; j < size - 1; j++ )
                  v_data[ i * size + j ] = 1.0;
         }
         else // Device == Devices::Cuda
         {
#ifdef HAVE_CUDA
            dim3 blockSize( 256 ), blocksCount, gridsCount;
            Devices::Cuda::setupThreads(
               blockSize,
               blocksCount,
               gridsCount,
               size,
               size );
            dim3 gridIdx;
            for( gridIdx.y = 0; gridIdx.y < gridsCount.y; gridIdx.y++ )
               for( gridIdx.x = 0; gridIdx.x < gridsCount.x; gridIdx.x++ )
               {
                  dim3 gridSize;
                  Devices::Cuda::setupGrid(
                     blocksCount,
                     gridsCount,
                     gridIdx,
                     gridSize );
                  boundariesTraverseKernel2D<<< gridSize, blockSize >>>( size, gridIdx, v_data );
               }
            for( gridIdx.y = 0; gridIdx.y < gridsCount.y; gridIdx.y++ )
               for( gridIdx.x = 0; gridIdx.x < gridsCount.x; gridIdx.x++ )
               {
                  dim3 gridSize;
                  Devices::Cuda::setupGrid(
                     blocksCount,
                     gridsCount,
                     gridIdx,
                     gridSize );
                  interiorTraverseKernel2D<<< gridSize, blockSize >>>( size, gridIdx, v_data );
               }
#endif
         }
      }

      void traversingUsingTraverser()
      {
         // TODO !!!!!!!!!!!!!!!!!!!!!!
         traverser.template processAllEntities< WriteOneTraverserUserDataType, WriteOneEntitiesProcessorType >
            ( grid, userData );
      }

   protected:
        
      Index size;
      Vector v;
      Real* v_data;
      GridPointer grid;
      MeshFunctionPointer u;
      Traverser traverser;
      WriteOneTraverserUserDataType userData;
};

template< typename Device,
          typename Real,
          typename Index >
class GridTraversersBenchmark< 3, Device, Real, Index >
{
   public:

      using Vector = Containers::Vector< Real, Device, Index >;
      using Grid = Meshes::Grid< 3, Real, Device, Index >;
      using GridPointer = Pointers::SharedPointer< Grid >;
      using Coordinates = typename Grid::CoordinatesType;
      using MeshFunction = Functions::MeshFunction< Grid >;
      using MeshFunctionPointer = Pointers::SharedPointer< MeshFunction >;
      using Cell = typename Grid::template EntityType< 3, Meshes::GridEntityNoStencilStorage >;
      using Traverser = Meshes::Traverser< Grid, Cell >;
      using TraverserUserData = WriteOneUserData< MeshFunctionPointer >;
      using WriteOneTraverserUserDataType = WriteOneUserData< MeshFunctionPointer >;
      using WriteOneEntitiesProcessorType = WriteOneEntitiesProcessor< WriteOneTraverserUserDataType >;
      
      GridTraversersBenchmark( Index size )
      : size( size ),
        v( size * size * size ),
        grid( size, size, size ),
        u( grid )
      {
         userData.u = this->u;
         v_data = v.getData();
      }

      void reset()
      {
         v.setValue( 0.0 );
         u->getData().setValue( 0.0 );
      };

      void writeOneUsingPureC()
      {
         if( std::is_same< Device, Devices::Host >::value )
         {
            for( int i = 0; i < size; i++ )
               for( int j = 0; j < size; j++ )
                  for( int k = 0; k < size; k++ )
                     v_data[ ( i * size + j ) * size + k ] = 1.0;
         }
         else // Device == Devices::Cuda
         {
#ifdef HAVE_CUDA
            dim3 blockSize( 256 ), blocksCount, gridsCount;
            Devices::Cuda::setupThreads(
               blockSize,
               blocksCount,
               gridsCount,
               size,
               size,
               size );
            dim3 gridIdx;
            for( gridIdx.z = 0; gridIdx.z < gridsCount.z; gridIdx.z++ )
               for( gridIdx.y = 0; gridIdx.y < gridsCount.y; gridIdx.y++ )
                  for( gridIdx.x = 0; gridIdx.x < gridsCount.x; gridIdx.x++ )
                  {
                     dim3 gridSize;
                     Devices::Cuda::setupGrid(
                        blocksCount,
                        gridsCount,
                        gridIdx,
                        gridSize );
                     fullGridTraverseKernel3D<<< gridSize, blockSize >>>( size, gridIdx, v_data );
                  }
#endif
         }
      }

      void writeOneUsingParallelFor()
      {
         Index _size = this->size;
         auto f = [=] __cuda_callable__ ( Index i, Index j, Index k, Real* data )
         {
            data[ ( i * _size + j ) * _size + k ] = 1.0;
         };
         
         ParallelFor3D< Device >::exec( ( Index ) 0,
                                        ( Index ) 0,
                                        ( Index ) 0,
                                        this->size,
                                        this->size,
                                        this->size,
                                        f, v.getData() );
      }

      void writeOneUsingTraverser()
      {
         traverser.template processAllEntities< WriteOneTraverserUserDataType, WriteOneEntitiesProcessorType >
            ( grid, userData );
      }

      void traverseUsingPureC()
      {
         if( std::is_same< Device, Devices::Host >::value )
         {
            for( int i = 0; i < size; i++ )
               for( int j = 0; j < size; j++ )
               {
                  v_data[ ( i * size + j ) * size ] = 2.0;
                  v_data[ ( i * size + j ) * size + size - 1 ] = 2.0;
               }
            for( int j = 0; j < size; j++ )
               for( int k = 1; k < size - 1; k++ )
               {
                  v_data[ j * size + k ] = 1.0;
                  v_data[ ( ( size - 1) * size + j ) * size + k ] = 1.0;
               }

            for( int i = 1; i < size -1; i++ )
               for( int k = 1; k < size - 1; k++ )
               {
                  v_data[ ( i * size ) * size + k ] = 2.0;
                  v_data[ ( i * size + size - 1 ) * size + k ] = 2.0;
               }

            for( int i = 1; i < size -1; i++ )
               for( int j = 1; j < size -1; j++ )
                  for( int k = 1; k < size - 1; k++ )
                     v_data[ ( i * size + j ) * size + k ] = 1.0;
         }
         else // Device == Devices::Cuda
         {
#ifdef HAVE_CUDA
            dim3 blockSize( 256 ), blocksCount, gridsCount;
            Devices::Cuda::setupThreads(
               blockSize,
               blocksCount,
               gridsCount,
               size,
               size,
               size );
            dim3 gridIdx;
            for( gridIdx.z = 0; gridIdx.z < gridsCount.z; gridIdx.z++ )
               for( gridIdx.y = 0; gridIdx.y < gridsCount.y; gridIdx.y++ )
                  for( gridIdx.x = 0; gridIdx.x < gridsCount.x; gridIdx.x++ )
                  {
                     dim3 gridSize;
                     Devices::Cuda::setupGrid(
                        blocksCount,
                        gridsCount,
                        gridIdx,
                        gridSize );
                     boundariesTraverseKernel3D<<< gridSize, blockSize >>>( size, gridIdx, v_data );
                  }
            for( gridIdx.z = 0; gridIdx.z < gridsCount.z; gridIdx.z++ )
               for( gridIdx.y = 0; gridIdx.y < gridsCount.y; gridIdx.y++ )
                  for( gridIdx.x = 0; gridIdx.x < gridsCount.x; gridIdx.x++ )
                  {
                     dim3 gridSize;
                     Devices::Cuda::setupGrid(
                        blocksCount,
                        gridsCount,
                        gridIdx,
                        gridSize );
                     interiorTraverseKernel3D<<< gridSize, blockSize >>>( size, gridIdx, v_data );
                  }
#endif
         }
      }

      void traverseUsingTraverser()
      {
         // TODO !!!!!!!!!!!!!!!!!!!!!!
         traverser.template processAllEntities< WriteOneTraverserUserDataType, WriteOneEntitiesProcessorType >
            ( grid, userData );
      }

   protected:
      
      Index size;
      Vector v;
      Real* v_data;
      GridPointer grid;
      MeshFunctionPointer u;
      Traverser traverser;
      WriteOneTraverserUserDataType userData;      
};

      } // namespace Traversers
   } // namespace Benchmarks
} // namespace TNL