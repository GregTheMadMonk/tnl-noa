/***************************************************************************
                          GridTraversersBenchmarkHelper.h  -  description
                             -------------------
    begin                : Jan 5, 2019
    copyright            : (C) 2019 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber

#pragma once

#include "AddOneEntitiesProcessor.h"
#include "BenchmarkTraverserUserData.h"

namespace TNL {
   namespace Benchmarks {
      namespace Traversers {

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename GridEntity,
          typename UserData,
          typename EntitiesProcessor >
__global__ void
_GridTraverser1D(
   const Meshes::Grid< 1, Real, Devices::Cuda, Index >* grid,
   UserData userData,
   const typename GridEntity::CoordinatesType begin,
   const typename GridEntity::CoordinatesType end,
   const Index gridIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef Meshes::Grid< 1, Real, Devices::Cuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;
 
   coordinates.x() = begin.x() + ( gridIdx * Devices::Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( coordinates <= end )
   {   
      GridEntity entity( *grid, coordinates );
      entity.refresh();
      ( userData.u->getData() )[ coordinates.x() ] += ( RealType ) 1.0;
      //( *userData.u )( entity) += 1.0;
      //EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
   }
}
#endif

template< typename Grid,
          typename Device = typename Grid::DeviceType >
class GridTraverserBenchmarkHelper{};

template< typename Grid >
class GridTraverserBenchmarkHelper< Grid, Devices::Host >
{
   public:

      using GridType = Grid;
      using GridPointer = Pointers::SharedPointer< Grid >;
      using RealType = typename GridType::RealType;
      using IndexType = typename GridType::IndexType;
      using CoordinatesType = typename Grid::CoordinatesType;
      using MeshFunction = Functions::MeshFunction< Grid >;
      using MeshFunctionPointer = Pointers::SharedPointer< MeshFunction >;
      using Cell = typename Grid::template EntityType< 1, Meshes::GridEntityNoStencilStorage >;
      using Traverser = Meshes::Traverser< Grid, Cell >;
      using UserDataType = BenchmarkTraverserUserData< MeshFunction >;
      using AddOneEntitiesProcessorType = AddOneEntitiesProcessor< UserDataType >;

      static void noBCTraverserTest( const GridPointer& grid,
                                     UserDataType& userData,
                                     std::size_t size )
      {
         /*Meshes::GridTraverser< Grid >::template processEntities< Cell, WriteOneEntitiesProcessorType, WriteOneTraverserUserDataType, false >(
           grid,
           CoordinatesType( 0 ),
           grid->getDimensions() - CoordinatesType( 1 ),
           userData );*/

         const CoordinatesType begin( 0 );
         const CoordinatesType end = CoordinatesType( size ) - CoordinatesType( 1 );
         //MeshFunction* _u = &u.template modifyData< Device >();
         Cell entity( *grid );
         for( IndexType x = begin.x(); x <= end.x(); x ++ )
         {
            entity.getCoordinates().x() = x;
            entity.refresh();
            AddOneEntitiesProcessorType::processEntity( entity.getMesh(), userData, entity );
         }

      }
};

template< typename Grid >
class GridTraverserBenchmarkHelper< Grid, Devices::Cuda >
{
   public:

      using GridType = Grid;
      using GridPointer = Pointers::SharedPointer< Grid >;
      using RealType = typename GridType::RealType;
      using IndexType = typename GridType::IndexType;
      using CoordinatesType = typename Grid::CoordinatesType;
      using MeshFunction = Functions::MeshFunction< Grid >;
      using MeshFunctionPointer = Pointers::SharedPointer< MeshFunction >;
      using Cell = typename Grid::template EntityType< 1, Meshes::GridEntityNoStencilStorage >;
      using Traverser = Meshes::Traverser< Grid, Cell >;
      using UserDataType = BenchmarkTraverserUserData< MeshFunction >;
      using AddOneEntitiesProcessorType = AddOneEntitiesProcessor< UserDataType >;

      static void noBCTraverserTest( const GridPointer& grid,
                                     UserDataType& userData,
                                     std::size_t size )
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
               _GridTraverser1D< RealType, IndexType, Cell, UserDataType, AddOneEntitiesProcessorType >
               <<< blocksCount, blockSize >>>
               ( &grid.template getData< Devices::Cuda >(),
                 userData,
                 CoordinatesType( 0 ),
                 CoordinatesType( size ) - CoordinatesType( 1 ),
                 gridIdx.x );

            }
#endif
      }
};

      } // namespace Traversers
   } // namespace Benchmarks
} // namespace TNL


