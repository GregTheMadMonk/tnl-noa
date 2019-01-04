/***************************************************************************
                          GridTraversersBenchmark_2D.h  -  description
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
                  v_data[ i * size + j ] += (Real) 1.0;
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
            data[ j * _size + i ] += (Real) 1.0;
         };
         
         ParallelFor2D< Device >::exec( ( Index ) 0,
                                        ( Index ) 0,
                                        this->size,
                                        this->size,
                                        f, v.getData() );
      }

      void writeOneUsingParallelForAndGridEntity()
      {
         const Grid* currentGrid = &grid.template getData< Device >();
         auto f = [=] __cuda_callable__ ( Index i, Index j,  Real* data )
         {
            Cell entity( *currentGrid );
            entity.getCoordinates().x() = i;
            entity.getCoordinates().y() = j;
            entity.refresh();
            data[ entity.getIndex() ] += (Real) 1.0;
         };
         
         ParallelFor2D< Device >::exec( ( Index ) 0,
                                        ( Index ) 0,
                                        this->size,
                                        this->size,
                                        f, v.getData() );
      }

      void writeOneUsingParallelForAndMeshFunction()
      {
         const Grid* currentGrid = &grid.template getData< Device >();
         MeshFunction* _u = &u.template modifyData< Device >();
         auto f = [=] __cuda_callable__ ( Index i, Index j,  Real* data )
         {
            Cell entity( *currentGrid );
            entity.getCoordinates().x() = i;
            entity.getCoordinates().y() = j;
            entity.refresh();
            ( *_u )( entity ) += (Real) 1.0;
         };
         
         ParallelFor2D< Device >::exec( ( Index ) 0,
                                        ( Index ) 0,
                                        this->size,
                                        this->size,
                                        f, v.getData() );
      }


      void writeOneUsingTraverser()
      {
         using CoordinatesType = typename Grid::CoordinatesType;
         traverser.template processAllEntities< WriteOneTraverserUserDataType, WriteOneEntitiesProcessorType >
            ( grid, userData );
         
         /*Meshes::GridTraverser< Grid >::template processEntities< Cell, WriteOneEntitiesProcessorType, WriteOneTraverserUserDataType, false >(
           grid,
           CoordinatesType( 0 ),
           grid->getDimensions() - CoordinatesType( 1 ),
           userData );*/
         /*const CoordinatesType begin( 0 );
         const CoordinatesType end = CoordinatesType( size ) - CoordinatesType( 1 );
         MeshFunction* _u = &u.template modifyData< Device >();
         Cell entity( *grid );
         for( Index y = begin.y(); y <= end.y(); y ++ )
            for( Index x = begin.x(); x <= end.x(); x ++ )
            {
               entity.getCoordinates().x() = x;
               entity.getCoordinates().y() = y;
               entity.refresh();
               WriteOneEntitiesProcessorType::processEntity( entity.getMesh(), userData, entity );
            }*/
      }

      void traverseUsingPureC()
      {
         if( std::is_same< Device, Devices::Host >::value )
         {
            for( int i = 0; i < size; i++ )
            {
               v_data[ i * size ] += (Real) 2.0;
               v_data[ i * size + size - 1 ] += (Real) 2.0;
            }
            for( int j = 1; j < size - 1; j++ )
            {
               v_data[ j ] += (Real) 2.0;
               v_data[ ( size - 1 ) * size + j ] += (Real) 2.0;
            }

            for( int i = 1; i < size - 1; i++ )
               for( int j = 1; j < size - 1; j++ )
                  v_data[ i * size + j ] += (Real) 1.0;
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

      } // namespace Traversers
   } // namespace Benchmarks
} // namespace TNL
