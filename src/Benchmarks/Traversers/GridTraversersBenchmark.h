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
      
      GridTraversersBenchmark( Index size )
      :v( size ), size( size )
      {}
      
      void writeOne()
      {
         
         auto f = [] __cuda_callable__ ( Index i, Real* data )
         {
            data[ i ] = i;
         };
         
         ParallelFor< Device >::exec( ( Index ) 0, size, f, v.getData() );
      }
      
      protected:
         
         Index size;
         Vector v;
};


template< typename Device,
          typename Real,
          typename Index >
class GridTraversersBenchmark< 2, Device, Real, Index >
{
   public:
      
      using Vector = Containers::Vector< Real, Device, Index >;
      
      GridTraversersBenchmark( Index size )
      :size( size ), v( size * size )  { }
      
      void writeOne()
      {
         Index _size = this->size;
         auto f = [=] __cuda_callable__ ( Index i, Index j,  Real* data )
         {
            data[ i * _size + j ] = i + j;
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
      
};

template< typename Device,
          typename Real,
          typename Index >
class GridTraversersBenchmark< 3, Device, Real, Index >
{
   public:
      
      using Vector = Containers::Vector< Real, Device, Index >;
      
      GridTraversersBenchmark( Index size )
      : size( size ), v( size * size * size ) {}
      
      void writeOne()
      {
         Index _size = this->size;
         auto f = [=] __cuda_callable__ ( Index i, Index j, Index k, Real* data )
         {
            data[ ( i * _size + j ) * _size + k ] = i + j + k;
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
      
};


   } // namespace Benchmarks
} // namespace TNL



