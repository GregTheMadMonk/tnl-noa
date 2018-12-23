/***************************************************************************
                          WriteOne.h  -  description
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

namespace TNL {
   namespace Benchmarks {
      

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
      using Cell = typename Grid::EntityType< 1, Meshes::GridEntityNoStencilStorage >;
      using Traverser = Meshes::Traverser< Grid, Cell >;
      using WriteOneTraverserUserDataType = WriteOneUserData< MeshFunctionPointer >;
      using WriteOneEntitiesProcessorType = WriteOneEntitiesProcessor< WriteOneTraverserUserDataType >;
      
      GridTraversersBenchmark( Index size )
      :v( size ), size( size ), grid( size ), u( grid )
      {
         userData.u = this->u;
      }
      
      void writeOneUsingParallelFor()
      {
         
         auto f = [] __cuda_callable__ ( Index i, Real* data )
         {
            data[ i ] = 1.0;
         };
         
         ParallelFor< Device >::exec( ( Index ) 0, size, f, v.getData() );
      }
      
      void writeOneUsingTraverser()
      {
         traverser.template processAllEntities< WriteOneTraverserUserDataType, WriteOneEntitiesProcessorType >
            ( grid, userData );
      }

      protected:

         Index size;
         Vector v;
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
      using Cell = typename Grid::EntityType< 2, Meshes::GridEntityNoStencilStorage >;
      using Traverser = Meshes::Traverser< Grid, Cell >;
      using TraverserUserData = WriteOneUserData< MeshFunctionPointer >;
      using WriteOneTraverserUserDataType = WriteOneUserData< MeshFunctionPointer >;
      using WriteOneEntitiesProcessorType = WriteOneEntitiesProcessor< WriteOneTraverserUserDataType >;
      
      GridTraversersBenchmark( Index size )
      :size( size ), v( size * size ), grid( size, size ), u( grid )
      {
         userData.u = this->u;
      }
      
      void writeOneUsingParallelFor()
      {
         Index _size = this->size;
         auto f = [=] __cuda_callable__ ( Index j, Index i,  Real* data )
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


   protected:
        
      Index size;
      Vector v;
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
      using Cell = typename Grid::EntityType< 3, Meshes::GridEntityNoStencilStorage >;
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
      }
      
      void writeOneUsingParallelFor()
      {
         Index _size = this->size;
         auto f = [=] __cuda_callable__ ( Index k, Index j, Index i, Real* data )
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

   protected:
      
      Index size;
      Vector v;
      GridPointer grid;
      MeshFunctionPointer u;
      Traverser traverser;
      WriteOneTraverserUserDataType userData;      
};


   } // namespace Benchmarks
} // namespace TNL



