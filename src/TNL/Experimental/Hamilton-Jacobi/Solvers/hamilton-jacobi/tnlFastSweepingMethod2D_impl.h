/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   tnlFastSweepingMethod2D_impl.h
 * Author: oberhuber
 *
 * Created on July 14, 2016, 10:32 AM
 */

#pragma once

#include "tnlFastSweepingMethod.h"
#include <TNL/TypeInfo.h>
#include <TNL/Devices/Cuda.h>


template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
FastSweepingMethod< Meshes::Grid< 2, Real, Device, Index >, Anisotropy >::
FastSweepingMethod()
: maxIterations( 1 )
{
   
}

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
const Index&
FastSweepingMethod< Meshes::Grid< 2, Real, Device, Index >, Anisotropy >::
getMaxIterations() const
{
   
}

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
void
FastSweepingMethod< Meshes::Grid< 2, Real, Device, Index >, Anisotropy >::
setMaxIterations( const IndexType& maxIterations )
{
   
}

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
void
FastSweepingMethod< Meshes::Grid< 2, Real, Device, Index >, Anisotropy >::
solve( const MeshPointer& mesh,
       const AnisotropyPointer& anisotropy,
       MeshFunctionPointer& u )
{
   MeshFunctionPointer auxPtr;
   InterfaceMapPointer interfaceMapPtr;
   auxPtr->setMesh( mesh );
   interfaceMapPtr->setMesh( mesh );
   std::cout << "Initiating the interface cells ..." << std::endl;
   BaseType::initInterface( u, auxPtr, interfaceMapPtr );
#ifdef HAVE_CUDA
   cudaDeviceSynchronize();
#endif
        
   auxPtr->save( "aux-ini.tnl" );

   typename MeshType::Cell cell( *mesh );
   
   IndexType iteration( 0 );
   InterfaceMapType interfaceMap = *interfaceMapPtr;
   MeshFunctionType aux = *auxPtr;
   while( iteration < this->maxIterations )
   {
      if( std::is_same< DeviceType, Devices::Host >::value )
      {
         for( cell.getCoordinates().y() = 0;
              cell.getCoordinates().y() < mesh->getDimensions().y();
              cell.getCoordinates().y()++ )
         {
            for( cell.getCoordinates().x() = 0;
                 cell.getCoordinates().x() < mesh->getDimensions().x();
                 cell.getCoordinates().x()++ )
               {
                  cell.refresh();
                  if( ! interfaceMap( cell ) )
                     this->updateCell( aux, cell );
               }
         }

         //aux.save( "aux-1.tnl" );

         for( cell.getCoordinates().y() = 0;
              cell.getCoordinates().y() < mesh->getDimensions().y();
              cell.getCoordinates().y()++ )
         {
            for( cell.getCoordinates().x() = mesh->getDimensions().x() - 1;
                 cell.getCoordinates().x() >= 0 ;
                 cell.getCoordinates().x()-- )		
               {
                  //std::cerr << "2 -> ";
                  cell.refresh();
                  if( ! interfaceMap( cell ) )            
                     this->updateCell( aux, cell );
               }
         }

         //aux.save( "aux-2.tnl" );

         for( cell.getCoordinates().y() = mesh->getDimensions().y() - 1;
              cell.getCoordinates().y() >= 0 ;
              cell.getCoordinates().y()-- )
            {
            for( cell.getCoordinates().x() = 0;
                 cell.getCoordinates().x() < mesh->getDimensions().x();
                 cell.getCoordinates().x()++ )
               {
                  //std::cerr << "3 -> ";
                  cell.refresh();
                  if( ! interfaceMap( cell ) )            
                     this->updateCell( aux, cell );
               }
            }

         //aux.save( "aux-3.tnl" );

         for( cell.getCoordinates().y() = mesh->getDimensions().y() - 1;
              cell.getCoordinates().y() >= 0;
              cell.getCoordinates().y()-- )
            {
            for( cell.getCoordinates().x() = mesh->getDimensions().x() - 1;
                 cell.getCoordinates().x() >= 0 ;
                 cell.getCoordinates().x()-- )		
               {
                  //std::cerr << "4 -> ";
                  cell.refresh();
                  if( ! interfaceMap( cell ) )            
                     this->updateCell( aux, cell );
               }
            }

         //aux.save( "aux-4.tnl" );

         /*for( cell.getCoordinates().x() = 0;
              cell.getCoordinates().x() < mesh->getDimensions().y();
              cell.getCoordinates().x()++ )
         {
            for( cell.getCoordinates().y() = 0;
                 cell.getCoordinates().y() < mesh->getDimensions().x();
                 cell.getCoordinates().y()++ )
               {
                  cell.refresh();
                  if( ! interfaceMap( cell ) )
                     this->updateCell( aux, cell );
               }
         }     


         aux.save( "aux-5.tnl" );

         for( cell.getCoordinates().x() = 0;
              cell.getCoordinates().x() < mesh->getDimensions().y();
              cell.getCoordinates().x()++ )
         {
            for( cell.getCoordinates().y() = mesh->getDimensions().x() - 1;
                 cell.getCoordinates().y() >= 0 ;
                 cell.getCoordinates().y()-- )		
               {
                  //std::cerr << "2 -> ";
                  cell.refresh();
                  if( ! interfaceMap( cell ) )            
                     this->updateCell( aux, cell );
               }
         }
         aux.save( "aux-6.tnl" );

         for( cell.getCoordinates().x() = mesh->getDimensions().y() - 1;
              cell.getCoordinates().x() >= 0 ;
              cell.getCoordinates().x()-- )
            {
            for( cell.getCoordinates().y() = 0;
                 cell.getCoordinates().y() < mesh->getDimensions().x();
                 cell.getCoordinates().y()++ )
               {
                  //std::cerr << "3 -> ";
                  cell.refresh();
                  if( ! interfaceMap( cell ) )            
                     this->updateCell( aux, cell );
               }
            }
         aux.save( "aux-7.tnl" );

         for( cell.getCoordinates().x() = mesh->getDimensions().y() - 1;
              cell.getCoordinates().x() >= 0;
              cell.getCoordinates().x()-- )
            {
            for( cell.getCoordinates().y() = mesh->getDimensions().x() - 1;
                 cell.getCoordinates().y() >= 0 ;
                 cell.getCoordinates().y()-- )		
               {
                  //std::cerr << "4 -> ";
                  cell.refresh();
                  if( ! interfaceMap( cell ) )            
                     this->updateCell( aux, cell );
               }
            }*/
      }
      if( std::is_same< DeviceType, Devices::Cuda >::value )
      {
         // TODO: CUDA code
#ifdef HAVE_CUDA
          const int cudaBlockSize( 16 );
          int numBlocksX = Devices::Cuda::getNumberOfBlocks( mesh->getDimensions().x(), cudaBlockSize );
          int numBlocksY = Devices::Cuda::getNumberOfBlocks( mesh->getDimensions().y(), cudaBlockSize );
          dim3 blockSize( cudaBlockSize, cudaBlockSize );
          dim3 gridSize( numBlocksX, numBlocksY );
          Devices::Cuda::synchronizeDevice();
          
          tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > > ptr;
          int nBlockIter = numBlocksX > numBlocksY ? numBlocksX : numBlocksY;
          nBlockIter = numBlocksX == numBlocksY ? nBlockIter + 1 : nBlockIter;
          for( int k = 0; k < nBlockIter; k++)
            CudaUpdateCellCaller<<< gridSize, blockSize, 18 * 18 * sizeof( Real ) >>>( ptr,
                                                                                    interfaceMapPtr.template getData< Device >(),
                                                                                    auxPtr.template modifyData< Device>() );
          cudaDeviceSynchronize();
          TNL_CHECK_CUDA_DEVICE;
          aux = *auxPtr;
          interfaceMap = *interfaceMapPtr;
#endif
      }
      iteration++;
   }
   aux.save("aux-final.tnl");
}

#ifdef HAVE_CUDA
template < typename Real, typename Device, typename Index >
__global__ void CudaUpdateCellCaller( tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > > ptr,
                                      const Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index >, 2, bool >& interfaceMap,
                                      Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& aux )
{
    int thri = threadIdx.x; int thrj = threadIdx.y; // nelze ke stejnym pristupovat znovu pres threadIdx (vzdy da jine hodnoty)
    int blIdx = blockIdx.x; int blIdy = blockIdx.y;
    int i = thri + blockDim.x*blIdx;
    int j = blockDim.y*blIdy + thrj;
    const Meshes::Grid< 2, Real, Device, Index >& mesh = interfaceMap.template getMesh< Devices::Cuda >();
    const Real hx = mesh.getSpaceSteps().x();
    const Real hy = mesh.getSpaceSteps().y();
    
    __shared__ Real sArray[ 18 ][ 18 ];
    for( int k = 0; k < 18; k++ )
        for( int l = 0; l < 18; l++ )
            sArray[ k ][ l ] = TypeInfo< Real >::getMaxValue();
    __syncthreads();
    /*//filling shared array
    ptr.setsArray( aux, sArray, mesh.getDimensions().x(), mesh.getDimensions().y(), blIdx, blIdy );
    __syncthreads();*/
    
    if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() )
    {
        sArray[thrj+1][thri+1] = aux[ j*mesh.getDimensions().x() + i ];
        ptr.setsArray( aux, sArray, mesh.getDimensions().x(), mesh.getDimensions().y(), blIdx, blIdy ); //fill edges of sArray
        __syncthreads();
            
        
        if( ! interfaceMap[ j*mesh.getDimensions().x() + i ] ) 
        {
            for( int k = 0; k < 17; k++ )
            {
                ptr.updateCell( sArray, thri+1, thrj+1, hx, hy );
                __syncthreads();
            }
        }
        /*for( int k = 0; k < mesh.getDimensions().x(); k++)
            for( int l = 0; l < mesh.getDimensions().y(); l++)
                aux[ k*mesh.getDimensions().x() + l ] = TypeInfo< Real >::getMaxValue();*/
        aux[j*mesh.getDimensions().x() + i] = sArray[thrj+1][thri+1];
        __syncthreads();
    }
    //ptr.getsArray( aux, sArray, mesh.getDimensions().x(), mesh.getDimensions().y(), blIdx, blIdy );
}
#endif
