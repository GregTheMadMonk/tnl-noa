/***************************************************************************
                          GridTraversersBenchmark_1D.h  -  description
                             -------------------
    begin                : Jan 3, 2019
    copyright            : (C) 2019 by oberhuber
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

      void writeOneUsingParallelForAndGridEntity()
      {
         const Grid* currentGrid = &grid.template getData< Device >();
         auto f = [=] __cuda_callable__ ( Index i, Real* data )
         {
            Cell entity( *currentGrid );
            entity.getCoordinates().x() = i;
            entity.refresh();
            data[ entity.getIndex() ] = +1.0;
         };
         ParallelFor< Device >::exec( ( Index ) 0, size, f, v.getData() );
      }

      void writeOneUsingParallelForAndMeshFunction()
      {
         const Grid* currentGrid = &grid.template getData< Device >();
         MeshFunction* _u = &u.template modifyData< Device >();
         auto f = [=] __cuda_callable__ ( Index i, Real* data )
         {
            Cell entity( *currentGrid );
            entity.getCoordinates().x() = i;
            entity.refresh();
            ( *_u )( entity ) = +1.0;
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

      } // namespace Traversers
   } // namespace Benchmarks
} // namespace TNL