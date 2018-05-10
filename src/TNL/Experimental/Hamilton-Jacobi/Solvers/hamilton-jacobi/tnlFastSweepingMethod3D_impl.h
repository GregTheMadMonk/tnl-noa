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

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
FastSweepingMethod< Meshes::Grid< 3, Real, Device, Index >, Anisotropy >::
FastSweepingMethod()
: maxIterations( 1 )
{
   
}

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
const Index&
FastSweepingMethod< Meshes::Grid< 3, Real, Device, Index >, Anisotropy >::
getMaxIterations() const
{
   
}

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
void
FastSweepingMethod< Meshes::Grid< 3, Real, Device, Index >, Anisotropy >::
setMaxIterations( const IndexType& maxIterations )
{
   
}

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
void
FastSweepingMethod< Meshes::Grid< 3, Real, Device, Index >, Anisotropy >::
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
   MeshFunctionType aux = *auxPtr;
   InterfaceMapType interfaceMap = * interfaceMapPtr;
    while( iteration < this->maxIterations )
    {
        if( std::is_same< DeviceType, Devices::Host >::value )
        {
           for( cell.getCoordinates().z() = 0;
                cell.getCoordinates().z() < mesh->getDimensions().z();
                cell.getCoordinates().z()++ )
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
           }
           //aux.save( "aux-1.tnl" );

           for( cell.getCoordinates().z() = 0;
                cell.getCoordinates().z() < mesh->getDimensions().z();
                cell.getCoordinates().z()++ )
           {
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
           }
           //aux.save( "aux-2.tnl" );
           for( cell.getCoordinates().z() = 0;
                cell.getCoordinates().z() < mesh->getDimensions().z();
                cell.getCoordinates().z()++ )
           {
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
           }
           //aux.save( "aux-3.tnl" );

           for( cell.getCoordinates().z() = 0;
                cell.getCoordinates().z() < mesh->getDimensions().z();
                cell.getCoordinates().z()++ )
           {
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
           }     
           //aux.save( "aux-4.tnl" );

           for( cell.getCoordinates().z() = mesh->getDimensions().z() - 1;
                cell.getCoordinates().z() >= 0;
                cell.getCoordinates().z()-- )
           {
              for( cell.getCoordinates().y() = 0;
                   cell.getCoordinates().y() < mesh->getDimensions().y();
                   cell.getCoordinates().y()++ )
              {
                 for( cell.getCoordinates().x() = 0;
                      cell.getCoordinates().x() < mesh->getDimensions().x();
                      cell.getCoordinates().x()++ )
                 {
                    //std::cerr << "5 -> ";
                    cell.refresh();
                    if( ! interfaceMap( cell ) )
                       this->updateCell( aux, cell );
                 }
              }
           }
           //aux.save( "aux-5.tnl" );

           for( cell.getCoordinates().z() = mesh->getDimensions().z() - 1;
                cell.getCoordinates().z() >= 0;
                cell.getCoordinates().z()-- )
           {
              for( cell.getCoordinates().y() = 0;
                   cell.getCoordinates().y() < mesh->getDimensions().y();
                   cell.getCoordinates().y()++ )
              {
                 for( cell.getCoordinates().x() = mesh->getDimensions().x() - 1;
                      cell.getCoordinates().x() >= 0 ;
                      cell.getCoordinates().x()-- )		
                 {
                    //std::cerr << "6 -> ";
                    cell.refresh();
                    if( ! interfaceMap( cell ) )            
                       this->updateCell( aux, cell );
                 }
              }
           }
           //aux.save( "aux-6.tnl" );

           for( cell.getCoordinates().z() = mesh->getDimensions().z() - 1;
                cell.getCoordinates().z() >= 0;
                cell.getCoordinates().z()-- )
           {
              for( cell.getCoordinates().y() = mesh->getDimensions().y() - 1;
                   cell.getCoordinates().y() >= 0 ;
                   cell.getCoordinates().y()-- )
              {
                 for( cell.getCoordinates().x() = 0;
                      cell.getCoordinates().x() < mesh->getDimensions().x();
                      cell.getCoordinates().x()++ )
                 {
                    //std::cerr << "7 -> ";
                    cell.refresh();
                    if( ! interfaceMap( cell ) )            
                       this->updateCell( aux, cell );
                 }
              }
           }
           //aux.save( "aux-7.tnl" );

           for( cell.getCoordinates().z() = mesh->getDimensions().z() - 1;
                cell.getCoordinates().z() >= 0;
                cell.getCoordinates().z()-- )
           {
              for( cell.getCoordinates().y() = mesh->getDimensions().y() - 1;
                   cell.getCoordinates().y() >= 0;
                   cell.getCoordinates().y()-- )
              {
                 for( cell.getCoordinates().x() = mesh->getDimensions().x() - 1;
                      cell.getCoordinates().x() >= 0 ;
                      cell.getCoordinates().x()-- )		
                 {
                    //std::cerr << "8 -> ";
                    cell.refresh();
                    if( ! interfaceMap( cell ) )            
                       this->updateCell( aux, cell );
                 }
              }
           }
      }
      if( std::is_same< DeviceType, Devices::Cuda >::value )
      {
         // TODO: CUDA code
#ifdef HAVE_CUDA
          const int cudaBlockSize( 8 );
          int numBlocksX = Devices::Cuda::getNumberOfBlocks( mesh->getDimensions().x(), cudaBlockSize );
          int numBlocksY = Devices::Cuda::getNumberOfBlocks( mesh->getDimensions().y(), cudaBlockSize );
          int numBlocksZ = Devices::Cuda::getNumberOfBlocks( mesh->getDimensions().z(), cudaBlockSize ); 
          if( cudaBlockSize * cudaBlockSize * cudaBlockSize > 1024 || numBlocksX > 1024 || numBlocksY > 1024 || numBlocksZ > 64 )
              std::cout << "Invalid kernel call. Dimensions of grid are max: [1024,1024,64], and maximum threads per block are 1024!" << std::endl;
          dim3 blockSize( cudaBlockSize, cudaBlockSize, cudaBlockSize );
          dim3 gridSize( numBlocksX, numBlocksY, numBlocksZ );
          Devices::Cuda::synchronizeDevice();
          
          tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > > ptr;
          for( int k = 0; k < numBlocksX; k++)
          CudaUpdateCellCaller<<< gridSize, blockSize >>>( ptr,
                                                                                  interfaceMapPtr.template getData< Device >(),
                                                                                  auxPtr.template modifyData< Device>() );
          cudaDeviceSynchronize();
          TNL_CHECK_CUDA_DEVICE;
          aux = *auxPtr;
          interfaceMap = *interfaceMapPtr;
#endif
      }
        
      //aux.save( "aux-8.tnl" );
      iteration++;
      
   }
   aux.save("aux-final.tnl");
}

#ifdef HAVE_CUDA
template < typename Real, typename Device, typename Index >
__global__ void CudaUpdateCellCaller( tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > > ptr,
                                      const Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index >, 3, bool >& interfaceMap,
                                      Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index > >& aux )
{
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    int k = blockDim.z*blockIdx.z + threadIdx.z;
    const Meshes::Grid< 3, Real, Device, Index >& mesh = interfaceMap.template getMesh< Devices::Cuda >();
    
    if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() && k < mesh.getDimensions().z() )
    {
        typedef typename Meshes::Grid< 3, Real, Device, Index >::Cell Cell;
        Cell cell( mesh );
        cell.getCoordinates().x() = i; cell.getCoordinates().y() = j; cell.getCoordinates().z() = k;
        cell.refresh();
        //tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > > ptr;
        for( int l = 0; l < 10; l++ )
        {
            if( ! interfaceMap( cell ) )
            {
               ptr.updateCell( aux, cell );
            }
        }
    }
}
#endif
