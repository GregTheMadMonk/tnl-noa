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
                 
          tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > > ptr;
          
          
          int BlockIterD = 1;
          
          TNL::Containers::Array< int, Devices::Cuda, IndexType > BlockIterDevice;
          BlockIterDevice.setSize( numBlocksX * numBlocksY * numBlocksZ );
          BlockIterDevice.setValue( 1 );
          /*int *BlockIterDevice;
          cudaMalloc(&BlockIterDevice, ( numBlocksX * numBlocksY * numBlocksZ ) * sizeof( int ) );*/
          int nBlocks = ( numBlocksX * numBlocksY * numBlocksZ )/512 + ((( numBlocksX * numBlocksY * numBlocksZ )%512 != 0) ? 1:0);
          
          TNL::Containers::Array< int, Devices::Cuda, IndexType > dBlock;
          dBlock.setSize( nBlocks );
          dBlock.setValue( 0 );
          /*int *dBlock;
          cudaMalloc(&dBlock, nBlocks * sizeof( int ) );*/
          
          while( BlockIterD )
          {
             CudaUpdateCellCaller<<< gridSize, blockSize >>>( ptr,
                                                              interfaceMapPtr.template getData< Device >(),
                                                              auxPtr.template modifyData< Device>(),
                                                              BlockIterDevice );
<<<<<<< HEAD
            cudaDeviceSynchronize();
            TNL_CHECK_CUDA_DEVICE;
            
            CudaParallelReduc<<< nBlocks , 512 >>>( BlockIterDevice, dBlock, ( numBlocksX * numBlocksY * numBlocksZ ) );
            cudaDeviceSynchronize();
            TNL_CHECK_CUDA_DEVICE;
=======
            //CudaParallelReduc<<< nBlocks , 512 >>>( BlockIterDevice, dBlock, ( numBlocksX * numBlocksY * numBlocksZ ) );
            //CudaParallelReduc<<< 1, nBlocks >>>( dBlock, dBlock, nBlocks );
>>>>>>> da336fb8bd927bc927bde8bde5876b18f07a23cf
            
            CudaParallelReduc<<< 1, nBlocks >>>( dBlock, dBlock, nBlocks );
            cudaDeviceSynchronize();
            TNL_CHECK_CUDA_DEVICE;
            cudaMemcpy(&BlockIterD, &dBlock[0], sizeof( int ), cudaMemcpyDeviceToHost);
                                   
            /*for( int i = 1; i < numBlocksX * numBlocksY; i++ )
                BlockIter[ 0 ] = BlockIter[ 0 ] || BlockIter[ i ];*/
            
          }
          //cudaFree( BlockIterDevice );
          //cudaFree( dBlock );
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
                                      Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index > >& aux,
                                      TNL::Containers::Array< int, Devices::Cuda, Index > BlockIterDevice )
{
    int thri = threadIdx.x; int thrj = threadIdx.y; int thrk = threadIdx.z;
    int blIdx = blockIdx.x; int blIdy = blockIdx.y; int blIdz = blockIdx.z;
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    int k = blockDim.z*blockIdx.z + threadIdx.z;
    int currentIndex = thrk * blockDim.x * blockDim.y + thrj * blockDim.x + thri;
    
    __shared__ volatile bool changed[8*8*8];
    changed[ currentIndex ] = false;
    
    if( thrj == 0 && thri == 0 && thrk == 0 )
        changed[ 0 ] = true;
    
    const Meshes::Grid< 3, Real, Device, Index >& mesh = interfaceMap.template getMesh< Devices::Cuda >();
    __shared__ Real hx;
    __shared__ Real hy;
    __shared__ Real hz;
    if( thrj == 1 && thri == 1 && thrk == 1 )
    {
        hx = mesh.getSpaceSteps().x();
        hy = mesh.getSpaceSteps().y();
        hz = mesh.getSpaceSteps().z();
    }
    __shared__ volatile Real sArray[10][10][10];
    sArray[thrk][thrj][thri] = std::numeric_limits< Real >::max();
    if(thri == 0 )
    {
        sArray[8][thrj+1][thrk+1] = std::numeric_limits< Real >::max();
        sArray[9][thrj+1][thrk+1] = std::numeric_limits< Real >::max();
        sArray[thrk+1][thrj+1][8] = std::numeric_limits< Real >::max();
        sArray[thrk+1][thrj+1][9] = std::numeric_limits< Real >::max();
        sArray[thrj+1][8][thrk+1] = std::numeric_limits< Real >::max();
        sArray[thrj+1][9][thrk+1] = std::numeric_limits< Real >::max();
    }
            
    //filling sArray edges
    int dimX = mesh.getDimensions().x(); int dimY = mesh.getDimensions().y();
    int dimZ = mesh.getDimensions().z();
    __shared__ volatile int numOfBlockx;
    __shared__ volatile int numOfBlocky;
    __shared__ volatile int numOfBlockz;
    __shared__ int xkolik;
    __shared__ int ykolik;
    __shared__ int zkolik;
    if( thri == 0 && thrj == 0 && thrk == 0 )
    {
        xkolik = blockDim.x + 1;
        ykolik = blockDim.y + 1;
        zkolik = blockDim.z + 1;
        numOfBlocky = dimY/blockDim.y + ((dimY%blockDim.y != 0) ? 1:0);
        numOfBlockx = dimX/blockDim.x + ((dimX%blockDim.x != 0) ? 1:0);
        numOfBlockz = dimZ/blockDim.z + ((dimZ%blockDim.z != 0) ? 1:0);
        
        if( numOfBlockx - 1 == blIdx )
            xkolik = dimX - (blIdx)*blockDim.x+1;

        if( numOfBlocky -1 == blIdy )
            ykolik = dimY - (blIdy)*blockDim.y+1;
        if( numOfBlockz-1 == blIdz )
            zkolik = dimZ - (blIdz)*blockDim.z+1;
        
        BlockIterDevice[ blIdz * numOfBlockx * numOfBlocky + blIdy * numOfBlockx + blIdx ] = 0;
    }
    __syncthreads();
    
    if( thri == 0 )
    {        
        if( blIdx != 0 && thrj+1 < ykolik && thrk+1 < zkolik )
            sArray[thrk+1][thrj+1][0] = aux[ blIdz*blockDim.z * dimX * dimY + blIdy * blockDim.y*dimX + blIdx*blockDim.x + thrj * dimX -1 + thrk*dimX*dimY ];
        else
            sArray[thrk+1][thrj+1][0] = std::numeric_limits< Real >::max();
    }
    
    if( thri == 1 )
    {
        if( dimX > (blIdx+1) * blockDim.x && thrj+1 < ykolik && thrk+1 < zkolik )
            sArray[thrk+1][thrj+1][9] = aux[ blIdz*blockDim.z * dimX * dimY + blIdy *blockDim.y*dimX+ blIdx*blockDim.x + blockDim.x + thrj * dimX + thrk*dimX*dimY ];
        else
            sArray[thrk+1][thrj+1][9] = std::numeric_limits< Real >::max();
    }
    if( thri == 2 )
    {        
        if( blIdy != 0 && thrj+1 < xkolik && thrk+1 < zkolik )
            sArray[thrk+1][0][thrj+1] = aux[ blIdz*blockDim.z * dimX * dimY + blIdy * blockDim.y*dimX + blIdx*blockDim.x - dimX + thrj + thrk*dimX*dimY ];
        else
            sArray[thrk+1][0][thrj+1] = std::numeric_limits< Real >::max();
    }
    
    if( thri == 3 )
    {
        if( dimY > (blIdy+1) * blockDim.y && thrj+1 < xkolik && thrk+1 < zkolik )
            sArray[thrk+1][9][thrj+1] = aux[ blIdz*blockDim.z * dimX * dimY + (blIdy+1) * blockDim.y*dimX + blIdx*blockDim.x + thrj + thrk*dimX*dimY ];
        else
            sArray[thrk+1][9][thrj+1] = std::numeric_limits< Real >::max();
    }
    if( thri == 4 )
    {        
        if( blIdz != 0 && thrj+1 < ykolik && thrk+1 < xkolik )
            sArray[0][thrj+1][thrk+1] = aux[ blIdz*blockDim.z * dimX * dimY + blIdy * blockDim.y*dimX + blIdx*blockDim.x - dimX * dimY + thrj * dimX + thrk ];
        else
            sArray[0][thrj+1][thrk+1] = std::numeric_limits< Real >::max();
    }
    
    if( thri == 5 )
    {
        if( dimZ > (blIdz+1) * blockDim.z && thrj+1 < ykolik && thrk+1 < xkolik )
            sArray[9][thrj+1][thrk+1] = aux[ (blIdz+1)*blockDim.z * dimX * dimY + blIdy * blockDim.y*dimX + blIdx*blockDim.x + thrj * dimX + thrk ];
        else
            sArray[9][thrj+1][thrk+1] = std::numeric_limits< Real >::max();
    }
    
    if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() && k < mesh.getDimensions().z() )
    {
        sArray[thrk+1][thrj+1][thri+1] = aux[ k*dimX*dimY + j*dimX + i ];
    }
    __shared__ volatile int loopcounter;
    loopcounter = 0;
    __syncthreads(); 
    while( changed[ 0 ] )
    {
        __syncthreads();
        
        changed[ currentIndex ] = false;
        
    //calculation of update cell
        if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() && k < dimZ )
        {
            if( ! interfaceMap[ k*dimX*dimY + j * mesh.getDimensions().x() + i ] )
            {
                changed[ currentIndex ] = ptr.updateCell( sArray, thri+1, thrj+1, thrk+1, hx,hy,hz);
            }
        }
        __syncthreads();
        
    //pyramid reduction
        if( blockDim.x*blockDim.y*blockDim.z == 1024 )
        {
            if( currentIndex < 512 )
            {
                changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 512 ];
            }
        }
        __syncthreads();
        if( blockDim.x*blockDim.y*blockDim.z >= 512 )
        {
            if( currentIndex < 256 )
            {
                changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 256 ];
            }
        }
        __syncthreads();
        if( blockDim.x*blockDim.y*blockDim.z >= 256 )
        {
            if( currentIndex < 128 )
            {
                changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 128 ];
            }
        }
        __syncthreads();
        if( blockDim.x*blockDim.y*blockDim.z >= 128 )
        {
            if( currentIndex < 64 )
            {
                changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 64 ];
            }
        }
        __syncthreads();
        if( currentIndex < 32 ) //POUZE IF JSOU SINCHRONNI NA JEDNOM WARPU
        {
            if( true ) changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 32 ];
            if( currentIndex < 16 ) changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 16 ];
            if( currentIndex < 8 ) changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 8 ];
            if( currentIndex < 4 ) changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 4 ];
            if( currentIndex < 2 ) changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 2 ];
            if( currentIndex < 1 ) changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 1 ];
        }
        __syncthreads();
        
        /*if(thri == 0 && thrj ==0 && thrk ==0 && blIdx == 0 && blIdy == 0 && blIdz == 0)
        {
            for(int m = 0; m < 8; m++){
                for(int n = 0; n<8; n++){
                    for(int b=0; b<8; b++)
                        printf(" %i ", changed[m*64 + n*8 + b]);
                    printf("\n");
                }
                printf("\n \n");
            }
        }*/
        if( changed[ 0 ] && thri == 0 && thrj == 0 && thrk == 0 )
        {
            //loopcounter++;
            BlockIterDevice[ blIdz * numOfBlockx * numOfBlocky + blIdy * numOfBlockx + blIdx ] = 1;
        }
        __syncthreads();
        /*if(thri == 0 && thrj==0 && thrk==0)
            printf("%i \n",loopcounter);
        if(loopcounter == 500)
            break;*/
    }
  
    if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() && k < dimZ && (!interfaceMap[ k*dimX*dimY+j * mesh.getDimensions().x() + i ]) )
        aux[ k*dimX*dimY + j * mesh.getDimensions().x() + i ] = sArray[thrk+1][ thrj + 1 ][ thri + 1 ];
}   
#endif
