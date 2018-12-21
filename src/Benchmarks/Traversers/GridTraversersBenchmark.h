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

namespace TNL {
   namespace Benchmarks {
      

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
      using Coordinates = typename Grid::CoordinatesType;
      using MeshFunction = Functions::MeshFunction< Grid >;
      using Cell = typename Grid::EntityType< 1, Meshes::GridEntityNoStencilStorage >;
      using Traverser = Meshes::Traverser< Grid, Cell >;
      
      GridTraversersBenchmark( Index size )
      :v( size ), size( size ), grid( size )
      {
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
         class EntitiesProcessor
         {
            
         };
         
         class UserData
         {
            
         };
         
         Traverser traverser;
         /*traverser.template processAllEntities< UserData, EntitiesProcessor >
                                           ( meshPointer,
                                             userData );*/
         
      }
      
      protected:
         
         Index size;
         Vector v;
         Grid grid;
};


template< typename Device,
          typename Real,
          typename Index >
class GridTraversersBenchmark< 2, Device, Real, Index >
{
   public:
      
      using Vector = Containers::Vector< Real, Device, Index >;
      using Grid = Meshes::Grid< 2, Real, Device, Index >;
      using Coordinates = typename Grid::CoordinatesType;
      
      GridTraversersBenchmark( Index size )
      :size( size ), v( size * size ), grid( size, size )
      {
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

   protected:
        
      Index size;
      Vector v;
      Grid grid;
      
};

template< typename Device,
          typename Real,
          typename Index >
class GridTraversersBenchmark< 3, Device, Real, Index >
{
   public:
      
      using Vector = Containers::Vector< Real, Device, Index >;
      using Grid = Meshes::Grid< 3, Real, Device, Index >;
      using Coordinates = typename Grid::CoordinatesType;
      
      GridTraversersBenchmark( Index size )
      : size( size ),
        v( size * size * size ),
        grid( size, size, size )
      {
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

   protected:
      
      Index size;
      Vector v;
      Grid grid;
      
};


   } // namespace Benchmarks
} // namespace TNL



