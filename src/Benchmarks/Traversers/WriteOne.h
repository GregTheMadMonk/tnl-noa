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
      

template< int Dimenions,
          typename Device,
          typename Real,
          typename Index >
class WriteOne{};

template< typename Device,
          typename Real,
          typename Index >
class WriteOne< 1, Device, Real, Index >
{
   public:
      
      using Vector = Containers::Vector< Real, Device, Index >;
      
      static void run( std::size_t size )
      {
         Vector v( size );
         auto writeOne = [] __cuda_callable__ ( Index i, Real* data )
         {
            data[ i ] = 1.0;
         };
         
         ParallelFor< Device >::exec( ( std::size_t ) 0, size, writeOne, v.getData() );
      }
};


template< typename Device,
          typename Real,
          typename Index >
class WriteOne< 2, Device, Real, Index >
{
   public:
      
      using Vector = Containers::Vector< Real, Device, Index >;
      
      static void run( std::size_t size )
      {
         Vector v( size * size );
         auto writeOne = [=] __cuda_callable__ ( Index i, Index j,  Real* data )
         {
            data[ i * size + j ] = 1.0;
         };
         
         ParallelFor2D< Device >::exec( ( std::size_t ) 0,
                                        ( std::size_t ) 0,
                                        size,
                                        size,
                                        writeOne, v.getData() );         
      }
};

template< typename Device,
          typename Real,
          typename Index >
class WriteOne< 3, Device, Real, Index >
{
   public:
      
      using Vector = Containers::Vector< Real, Device, Index >;
      
      static void run( std::size_t size )
      {
         Vector v( size * size * size );
         auto writeOne = [=] __cuda_callable__ ( Index i, Index j, Index k, Real* data )
         {
            data[ ( i * size + j ) * size + k ] = 1.0;
         };
         
         ParallelFor3D< Device >::exec( ( std::size_t ) 0, 
                                        ( std::size_t ) 0, 
                                        ( std::size_t ) 0, 
                                        size,
                                        size,
                                        size,
                                        writeOne, v.getData() );         
      }
};


   } // namespace Benchmarks
} // namespace TNL



