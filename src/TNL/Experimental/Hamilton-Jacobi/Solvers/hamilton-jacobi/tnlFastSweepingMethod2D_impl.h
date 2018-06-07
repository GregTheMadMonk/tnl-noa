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
          
          
          /*int nBlockIter = numBlocksX > numBlocksY ? numBlocksX : numBlocksY;
          nBlockIter = numBlocksX == numBlocksY ? nBlockIter + 1 : nBlockIter;*/
          
          bool isNotDone = true; numBlocksX = 16; numBlocksY = 16;
          double* BlockIter = (double*)malloc( ( numBlocksX * numBlocksY ) * sizeof( double ) );
          
          double *BlockIterDevice;
          cudaMalloc(&BlockIterDevice, ( numBlocksX * numBlocksY ) * sizeof( double ) );
          
          //while( isNotDone )
          {
           for( int i = 0; i < numBlocksX * numBlocksY; i++ )
                BlockIter[ i ] = 0.0;
           cudaMemcpy(BlockIterDevice, BlockIter, ( numBlocksX * numBlocksY ) * sizeof( double ), cudaMemcpyHostToDevice);
           
            isNotDone = false;
            
            CudaUpdateCellCaller<<< gridSize, blockSize >>>( ptr,
                                                             interfaceMapPtr.template getData< Device >(),
                                                             auxPtr.template modifyData< Device>(),
                                                             BlockIterDevice );
            cudaDeviceSynchronize();         
            TNL_CHECK_CUDA_DEVICE;
            cudaMemcpy(BlockIter, BlockIterDevice, ( numBlocksX * numBlocksY ) * sizeof( double ), cudaMemcpyDeviceToHost);
            
            for( int j = numBlocksY-1; j > -1; j-- )
            {    
                for( int i = 0; i < numBlocksX; i++ )
                {
                    //if( BlockIter[ j * numBlocksX + i ] )
                        std::cout << BlockIter[ j * numBlocksY + i ] << "\t";
                    //else
                    //    std::cout << "0" << " ";    
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
                         
            for( int i = 0; i < numBlocksX * numBlocksY; i++ )
                isNotDone = isNotDone || BlockIter[ i ];
            
          }
          delete[] BlockIter;
          cudaFree( BlockIterDevice );
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
                                      Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& aux,
                                      Real *BlockIterDevice )
{
    int thri = threadIdx.x; int thrj = threadIdx.y;
    int blIdx = blockIdx.x; int blIdy = blockIdx.y;
    int i = thri + blockDim.x*blIdx;
    int j = blockDim.y*blIdy + thrj;
    int currentIndex = thrj * 16 + thri;
    
    __shared__ volatile Real changed[256];
    changed[ currentIndex ] = 0.0;
    __syncthreads();
    
    if( thrj == 0 && thri == 0 )
        changed[ 0 ] = 10.0;
    __syncthreads();
        //SmallCycleBoolin = true;
    
    
    const Meshes::Grid< 2, Real, Device, Index >& mesh = interfaceMap.template getMesh< Devices::Cuda >();
    __shared__ Real hx;
    __shared__ Real hy;
    if( thrj == 0 && thri == 0 )
    {
        hx = mesh.getSpaceSteps().x();
        hy = mesh.getSpaceSteps().y();
    }
    __syncthreads();
    
    __shared__ volatile Real sArray[ 18 ][ 18 ];
    sArray[thrj][thri] = TypeInfo< Real >::getMaxValue();
    __syncthreads();
    
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
    
    //filling BlockIterDevice
        //BlockIterDevice[ blIdy * numOfBlockx + blIdx ] = 0.0;
     }
    /*if( blIdx == 2 && blIdy == 3 )
        BlockIterDevice[ currentIndex ] = 0.0;*/
    __syncthreads();
    
    if( thri == 0 )
    {        
        if( dimX > (blIdx+1) * blockDim.x  && thrj+1 < ykolik )
            sArray[thrj+1][xkolik] = aux[ blIdy*blockDim.y*dimX - dimX + blIdx*blockDim.x - 1 + (thrj+1)*dimX + xkolik ];
        else
            sArray[thrj+1][xkolik] = TypeInfo< Real >::getMaxValue();
    }
    __syncthreads();
    
    if( thri == 1 )
    {
        if( blIdx != 0 && thrj+1 < ykolik )
            sArray[thrj+1][0] = aux[ blIdy*blockDim.y*dimX - dimX + blIdx*blockDim.x - 1 + (thrj+1)*dimX ];
        else
            sArray[thrj+1][0] = TypeInfo< Real >::getMaxValue();
    }
    __syncthreads();
    
    if( thrj == 0 )
    {
        if( dimY > (blIdy+1) * blockDim.y  && thri+1 < xkolik )
            sArray[ykolik][thri+1] = aux[ blIdy*blockDim.y*dimX - dimX + blIdx*blockDim.x - 1 + ykolik*dimX + thri+1 ];
        else
           sArray[ykolik][thri+1] = TypeInfo< Real >::getMaxValue();
    }
    __syncthreads();
    
    if( thrj == 1 )
    {
        if( blIdy != 0 && thri+1 < xkolik )
            sArray[0][thri+1] = aux[ blIdy*blockDim.y*dimX - dimX + blIdx*blockDim.x - 1 + thri+1 ];
        else
            sArray[0][thri+1] = TypeInfo< Real >::getMaxValue();
    }
    __syncthreads();
    
        
    if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() )
    {    
        sArray[thrj+1][thri+1] = aux[ j*mesh.getDimensions().x() + i ];
        __syncthreads();
    }  
    
    int loopcounter;
    loopcounter = 0;
    __syncthreads();
    
  /*if( ( blIdx == 4 && blIdy == 5 ) || ( blIdx == 2 && blIdy == 2 ) || ( blIdx == 2 && blIdy == 3 ) ||
      ( blIdx == 5 && blIdy == 5 ) || ( blIdx == 6 && blIdy == 5 ) || ( blIdx == 5 && blIdy == 4 ) ||
      ( blIdx == 5 && blIdy == 2 ) || ( blIdx == 3 && blIdy == 2 ) || ( blIdx == 4 && blIdy == 2 ) )*/
    while( changed[ 0 ] > 0.1 )
    {
        changed[ currentIndex] = 0.0;
        __syncthreads();
        
    //calculation of update cell
        if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() )
        {
            if( ! interfaceMap[ j * mesh.getDimensions().x() + i ] )
            {
                changed[ currentIndex ] = ptr.updateCell( sArray, thri+1, thrj+1, hx, hy );  
    __syncthreads();
            }
        }
        __syncthreads();
        
    //pyramid reduction
        if( blockDim.x*blockDim.y >= 256 )
        {
            if( currentIndex < 128 )
            {
                changed[ currentIndex ] = changed[ currentIndex ] + changed[ currentIndex + 128 ];
            }
        }
        __syncthreads();
        if( blockDim.x*blockDim.y >= 128 )
        {
            if( currentIndex < 64 )
            {
                changed[ currentIndex ] = changed[ currentIndex ] + changed[ currentIndex + 64 ];
            }
        }
        __syncthreads();
        if( currentIndex < 32 ) //POUZE IF JSOU SINCHRONNI NA JEDNOM WARPU
        {
            if( true ) changed[ currentIndex ] = changed[ currentIndex ] + changed[ currentIndex + 32 ];
            if( currentIndex < 16 ) changed[ currentIndex ] = changed[ currentIndex ] + changed[ currentIndex + 16 ];
            if( currentIndex < 8 ) changed[ currentIndex ] = changed[ currentIndex ] + changed[ currentIndex + 8 ];
            if( currentIndex < 4 ) changed[ currentIndex ] = changed[ currentIndex ] + changed[ currentIndex + 4 ];
            if( currentIndex < 2 ) changed[ currentIndex ] = changed[ currentIndex ] + changed[ currentIndex + 2 ];
            if( currentIndex < 1 ) changed[ currentIndex ] = changed[ currentIndex ] + changed[ currentIndex + 1 ];
        }
        __syncthreads();
        
        
               
        loopcounter++;
        __syncthreads();
        if( currentIndex != 0 )
            changed[ currentIndex ] = 0.0;
        __syncthreads();
        
        if( loopcounter > 200 )
            changed[ currentIndex ] = 0.0;
        __syncthreads();
    }
    /*__syncthreads();
    if( blIdy == 3 && blIdx == 1 )
    {
       BlockIterDevice[ currentIndex ] = changed[ currentIndex ];
       BlockIterDevice[ 0 ] = loopcounter;
    }*/
    __syncthreads();
    if( blIdx == 2 && blIdy == 3 )
    { 
        //SmallCycleBoolin = changed[ 0 ];
        //if( changed[ 0 ] )
             BlockIterDevice[ currentIndex ] = changed[ currentIndex ];// loopcounter;
             BlockIterDevice[ 0 ] = loopcounter;

    }
      
    
    if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() && (!interfaceMap[ j * mesh.getDimensions().x() + i ]) )
        aux[ j * mesh.getDimensions().x() + i ] = sArray[ thrj + 1 ][ thri + 1 ];
    __syncthreads();
}
#endif
