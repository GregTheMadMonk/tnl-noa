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


#include <iostream>
#include <fstream>

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
   /*MeshFunctionType v;
   v.setMesh(mesh);
   double A[320][320];
    for (int i = 0; i < 320; i++)
        for (int j = 0; j < 320; j++)
            A[i][j] = 0;
    
    std::ifstream file("/home/maty/Downloads/mapa2.txt");

    for (int i = 0; i < 320; i++)
        for (int j = 0; j < 320; j++)
            file >> A[i][j];
    file.close();
    for (int i = 0; i < 320; i++)
        for (int j = 0; j < 320; j++)
            v[i*320 + j] = A[i][j];
   v.save("mapa.tnl");*/
   
       
   MeshFunctionPointer auxPtr;
   InterfaceMapPointer interfaceMapPtr;
   auxPtr->setMesh( mesh );
   interfaceMapPtr->setMesh( mesh );
   std::cout << "Initiating the interface cells ..." << std::endl;
   BaseType::initInterface( u, auxPtr, interfaceMapPtr );
        
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
          
          tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > > ptr;
          
          
          
          //int BlockIter = 1;// = (bool*)malloc( ( numBlocksX * numBlocksY ) * sizeof( bool ) );

          int *BlockIterDevice;
          int BlockIterD = 1;
          
          cudaMalloc(&BlockIterDevice, ( numBlocksX * numBlocksY ) * sizeof( int ) );
          int nBlocks = ( numBlocksX * numBlocksY )/512 + ((( numBlocksX * numBlocksY )%512 != 0) ? 1:0);
          int *dBlock;
          cudaMalloc(&dBlock, nBlocks * sizeof( int ) );
          
          while( BlockIterD )
          {
           /*for( int i = 0; i < numBlocksX * numBlocksY; i++ )
                BlockIter[ i ] = false;*/
                       
            CudaUpdateCellCaller<<< gridSize, blockSize >>>( ptr,
                                                             interfaceMapPtr.template getData< Device >(),
                                                             auxPtr.template modifyData< Device>(),
                                                             BlockIterDevice );
            
            CudaParallelReduc<<< nBlocks , 512 >>>( BlockIterDevice, dBlock, ( numBlocksX * numBlocksY ) );
            CudaParallelReduc<<< 1, nBlocks >>>( dBlock, dBlock, nBlocks );
            
            cudaMemcpy(&BlockIterD, &dBlock[0], sizeof( int ), cudaMemcpyDeviceToHost);
                                   
            /*for( int i = 1; i < numBlocksX * numBlocksY; i++ )
                BlockIter[ 0 ] = BlockIter[ 0 ] || BlockIter[ i ];*/
            
          }
          cudaFree( BlockIterDevice );
          cudaFree( dBlock );
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
__global__ void CudaParallelReduc( int *BlockIterDevice, int *dBlock, int nBlocks )
{
    int i = threadIdx.x;
    int blId = blockIdx.x;
    __shared__ volatile int sArray[ 512 ];
    sArray[ i ] = false;
    if(blId * 1024 + i < nBlocks )
        sArray[ i ] = BlockIterDevice[ blId * 1024 + i ];
    
    if (blockDim.x * blockDim.y == 1024) {
        if (i < 512)
            sArray[ i ] += sArray[ i ];
    }
    __syncthreads();
    if (blockDim.x * blockDim.y >= 512) {
        if (i < 256) {
            sArray[ i ] += sArray[ i ];
        }
    }
    if (blockDim.x * blockDim.y >= 256) {
        if (i < 128) {
            sArray[ i ] += sArray[ i + 128 ];
        }
    }
    __syncthreads();
    if (blockDim.x * blockDim.y >= 128) {
        if (i < 64) {
            sArray[ i ] += sArray[ i + 64 ];
        }
    }
    __syncthreads();
    if (i < 32 )
    {
        if(  blockDim.x * blockDim.y >= 64 ) sArray[ i ] += sArray[ i + 32 ];
        if(  blockDim.x * blockDim.y >= 32 )  sArray[ i ] += sArray[ i + 16 ];
        if(  blockDim.x * blockDim.y >= 16 )  sArray[ i ] += sArray[ i + 8 ];
        if(  blockDim.x * blockDim.y >= 8 )  sArray[ i ] += sArray[ i + 4 ];
        if(  blockDim.x * blockDim.y >= 4 )  sArray[ i ] += sArray[ i + 2 ];
        if(  blockDim.x * blockDim.y >= 2 )  sArray[ i ] += sArray[ i + 1 ];
    }
    
    if( i == 0 )
        dBlock[ blId ] = sArray[ 0 ];
}



template < typename Real, typename Device, typename Index >
__global__ void CudaUpdateCellCaller( tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > > ptr,
                                      const Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index >, 2, bool >& interfaceMap,
                                      Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& aux,
                                      int *BlockIterDevice )
{
    int thri = threadIdx.x; int thrj = threadIdx.y;
    int blIdx = blockIdx.x; int blIdy = blockIdx.y;
    int i = thri + blockDim.x*blIdx;
    int j = blockDim.y*blIdy + thrj;
    int currentIndex = thrj * 16 + thri;
    
    __shared__ volatile bool changed[256];
    changed[ currentIndex ] = false;
    
    if( thrj == 0 && thri == 0 )
        changed[ 0 ] = true;
    
    const Meshes::Grid< 2, Real, Device, Index >& mesh = interfaceMap.template getMesh< Devices::Cuda >();
    __shared__ Real hx;
    __shared__ Real hy;
    if( thrj == 1 && thri == 1 )
    {
        hx = mesh.getSpaceSteps().x();
        hy = mesh.getSpaceSteps().y();
    }
    
    __shared__ volatile Real sArray[ 18 ][ 18 ];
    sArray[thrj][thri] = TypeInfo< Real >::getMaxValue();
    
    //filling sArray edges
    int dimX = mesh.getDimensions().x(); int dimY = mesh.getDimensions().y();
    __shared__ volatile int numOfBlockx;
    __shared__ volatile int numOfBlocky;
    __shared__ int xkolik;
    __shared__ int ykolik;
    if( thri == 0 && thrj == 0 )
    {
        xkolik = blockDim.x + 1;
        ykolik = blockDim.y + 1;
        numOfBlocky = dimY/blockDim.y + ((dimY%blockDim.y != 0) ? 1:0);
        numOfBlockx = dimX/blockDim.x + ((dimX%blockDim.x != 0) ? 1:0);
    
        if( numOfBlockx - 1 == blIdx )
            xkolik = dimX - (blIdx)*blockDim.x+1;

        if( numOfBlocky -1 == blIdy )
            ykolik = dimY - (blIdy)*blockDim.y+1;
        BlockIterDevice[ blIdy * numOfBlockx + blIdx ] = 0;
    }
    __syncthreads();
    
    if( thri == 0 )
    {        
        if( dimX > (blIdx+1) * blockDim.x  && thrj+1 < ykolik )
            sArray[thrj+1][xkolik] = aux[ blIdy*blockDim.y*dimX - dimX + blIdx*blockDim.x - 1 + (thrj+1)*dimX + xkolik ];
        else
            sArray[thrj+1][xkolik] = TypeInfo< Real >::getMaxValue();
    }
    
    if( thri == 1 )
    {
        if( blIdx != 0 && thrj+1 < ykolik )
            sArray[thrj+1][0] = aux[ blIdy*blockDim.y*dimX - dimX + blIdx*blockDim.x - 1 + (thrj+1)*dimX ];
        else
            sArray[thrj+1][0] = TypeInfo< Real >::getMaxValue();
    }
    
    if( thri == 2 )
    {
        if( dimY > (blIdy+1) * blockDim.y  && thri+1 < xkolik )
            sArray[ykolik][thrj+1] = aux[ blIdy*blockDim.y*dimX - dimX + blIdx*blockDim.x - 1 + ykolik*dimX + thrj+1 ];
        else
           sArray[ykolik][thrj+1] = TypeInfo< Real >::getMaxValue();
    }
    
    if( thri == 3 )
    {
        if( blIdy != 0 && thrj+1 < xkolik )
            sArray[0][thrj+1] = aux[ blIdy*blockDim.y*dimX - dimX + blIdx*blockDim.x - 1 + thrj+1 ];
        else
            sArray[0][thrj+1] = TypeInfo< Real >::getMaxValue();
    }
    
        
    if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() )
    {    
        sArray[thrj+1][thri+1] = aux[ j*mesh.getDimensions().x() + i ];
    }
    __syncthreads();  

    while( changed[ 0 ] )
    {
        __syncthreads();
        
        changed[ currentIndex] = false;
        
    //calculation of update cell
        if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() )
        {
            if( ! interfaceMap[ j * mesh.getDimensions().x() + i ] )
            {
                changed[ currentIndex ] = ptr.updateCell( sArray, thri+1, thrj+1, hx, hy );
            }
        }
        __syncthreads();
        
        /*if( thri == 0 && thrj == 0 && blIdx == 1 && blIdy == 3 ){
            for( int h = 15; h > -1; h-- ){
                for( int g = 0; g < 16; g++ ){
                    printf( "%d\t", changed[h*16+g] );
                }
                printf("\n");
            }
            printf("\n");
        }
        __syncthreads();*/
        
    //pyramid reduction
        if( blockDim.x*blockDim.y >= 256 )
        {
            if( currentIndex < 128 )
            {
                changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 128 ];
            }
        }
        __syncthreads();
        if( blockDim.x*blockDim.y >= 128 )
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
        if( changed[ 0 ] && thri == 0 && thrj == 0 )
            BlockIterDevice[ blIdy * numOfBlockx + blIdx ] = 1;
        __syncthreads();
    }
  
    if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() && (!interfaceMap[ j * mesh.getDimensions().x() + i ]) )
        aux[ j * mesh.getDimensions().x() + i ] = sArray[ thrj + 1 ][ thri + 1 ];
}
#endif
