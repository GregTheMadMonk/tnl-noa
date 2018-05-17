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
          
          bool isNotDone = true;
          bool* BlockIter = (bool*)malloc( ( numBlocksX * numBlocksY ) * sizeof( bool ) );
          
          bool *BlockIterDevice;
          cudaMalloc(&BlockIterDevice, ( numBlocksX * numBlocksY ) * sizeof( bool ) );
          
          while( isNotDone )
          {
           for( int i = 0; i < numBlocksX * numBlocksY; i++ )
                BlockIter[ i ] = false;   
           
            isNotDone = false;
            
            CudaUpdateCellCaller<<< gridSize, blockSize, 18 * 18 * sizeof( Real ) >>>( ptr,
                                                                                    interfaceMapPtr.template getData< Device >(),
                                                                                    auxPtr.template modifyData< Device>(),
                                                                                    BlockIterDevice );
            cudaMemcpy(BlockIter, BlockIterDevice, ( numBlocksX * numBlocksY ) * sizeof( bool ), cudaMemcpyDeviceToHost);
            for( int i = 0; i < numBlocksX; i++ )
            {    for( int j = 0; j < numBlocksY; j++ )
                {
                    if( BlockIter[ j * numBlocksY + i ] )
                        std::cout << "true." << "\t";
                    else
                        std::cout << "false." << "\t";    
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
                                      bool *BlockIterDevice )
{
    int thri = threadIdx.x; int thrj = threadIdx.y; // nelze ke stejnym pristupovat znovu pres threadIdx (vzdy da jine hodnoty)
    int blIdx = blockIdx.x; int blIdy = blockIdx.y;
    int i = thri + blockDim.x*blIdx;
    int j = blockDim.y*blIdy + thrj;
    
    __shared__ volatile bool changed[256];
    changed[ thrj * blockDim.x + thri ] = false;
     __shared__ volatile bool SmallCycleBoolin;
    if( thrj == 0 && thri == 0 )
        SmallCycleBoolin = true;
    
    
    const Meshes::Grid< 2, Real, Device, Index >& mesh = interfaceMap.template getMesh< Devices::Cuda >();
    const Real hx = mesh.getSpaceSteps().x();
    const Real hy = mesh.getSpaceSteps().y();
    
    __shared__ volatile Real sArray[ 18 ][ 18 ];
    sArray[thrj][thri] = TypeInfo< Real >::getMaxValue();
    
    //filling sArray edges
    int dimX = mesh.getDimensions().x(); int dimY = mesh.getDimensions().y();
    int numOfBlockx = dimX/16 + ((dimX%16 != 0) ? 1:0);
    int numOfBlocky = dimY/16 + ((dimY%16 != 0) ? 1:0);
    int xkolik = 17;
    int ykolik = 17;
    
    
    if( numOfBlockx - 1 == blIdx )
        xkolik = dimX - (blIdx)*16+1;

    if( numOfBlocky -1 == blIdy )
        ykolik = dimY - (blIdy)*16+1;
       
    
    if( thri == 0 )
    {        
        if( dimX > (blIdx+1) * 16  && thrj+1 < ykolik )
            sArray[thrj+1][xkolik] = aux[ blIdy*16*dimX - dimX + blIdx*16 - 1 + (thrj+1)*dimX + 17 ];
        else
            sArray[thrj+1][xkolik] = TypeInfo< Real >::getMaxValue();
    }
    
    if( thri == 1 )
    {
        if( blIdx != 0 && thrj+1 < ykolik )
            sArray[thrj+1][0] = aux[ blIdy*16*dimX - dimX + blIdx*16 - 1 + (thrj+1)*dimX + 0];
        else
            sArray[thrj+1][0] = TypeInfo< Real >::getMaxValue();
    }
    
    if( thrj == 0 )
    {
        if( dimY > (blIdy+1) * 16  && thri+1 < xkolik )
            sArray[ykolik][thri+1] = aux[ blIdy*16*dimX - dimX + blIdx*16 - 1 + 17*dimX + thri+1 ];
        else
           sArray[ykolik][thri+1] = TypeInfo< Real >::getMaxValue();
    }
    
    if( thrj == 1 )
    {
        if( blIdy != 0 && thri+1 < xkolik )
            sArray[0][thri+1] = aux[ blIdy*16*dimX - dimX + blIdx*16 - 1 + 0*dimX + thri+1 ];
        else
            sArray[0][thri+1] = TypeInfo< Real >::getMaxValue();
    }
    
    //filling BlockIterDevice
    BlockIterDevice[ blIdy * numOfBlockx + blIdx ] = false;
    
    
    if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() )
    {    
        sArray[thrj+1][thri+1] = aux[ j*mesh.getDimensions().x() + i ];
        __syncthreads();
    }    
    
    __shared__ int loopcounter;
    if( thri == 0 && thrj == 0 )
        loopcounter= 0;
  //if( blIdx == 5 && blIdy == 4 )
    while( SmallCycleBoolin )   
    {
        if( thri == 1 && thrj == 1 )
            SmallCycleBoolin = false;
        
        changed[ thrj * 16 + thri ] = false;
        __syncthreads();
        
    //calculation of update cell
        if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() )
        { 
            if( ! interfaceMap[ j * mesh.getDimensions().x() + i ] )
            {
                changed[ thrj * 16 + thri ] = ptr.updateCell( sArray, thri+1, thrj+1, hx, hy );
                __syncthreads();   
            }
        }
        
    //pyramid reduction
        if( blockDim.x*blockDim.y >= 256 )
        {
            if( (thrj * 16 + thri) < 128 )
            {
                changed[ thrj * 16 + thri ] = changed[ thrj * 16 + thri ] || changed[ thrj * 16 + thri + 128 ];
            }
        __syncthreads();
        }
        if( blockDim.x*blockDim.y >= 128 )
        {
            if( (thrj * 16 + thri) < 64 )
            {
                changed[ thrj * 16 + thri ] = changed[ thrj * 16 + thri ] || changed[ thrj * 16 + thri + 64 ];
            }
        __syncthreads();
        }
        if( (thrj * 16 + thri) < 32 )
        {
            changed[ thrj * 16 + thri ] = changed[ thrj * 16 + thri ] || changed[ thrj * 16 + thri + 32 ];
            changed[ thrj * 16 + thri ] = changed[ thrj * 16 + thri ] || changed[ thrj * 16 + thri + 16 ];
            changed[ thrj * 16 + thri ] = changed[ thrj * 16 + thri ] || changed[ thrj * 16 + thri + 8 ];
            changed[ thrj * 16 + thri ] = changed[ thrj * 16 + thri ] || changed[ thrj * 16 + thri + 4 ];
            changed[ thrj * 16 + thri ] = changed[ thrj * 16 + thri ] || changed[ thrj * 16 + thri + 2 ];
            changed[ thrj * 16 + thri ] = changed[ thrj * 16 + thri ] || changed[ thrj * 16 + thri + 1 ];
        }
        __syncthreads();

        if( thrj == 1 && thri == 1 )
        {
            loopcounter++;
            if( loopcounter > 1000 )
                break;
            SmallCycleBoolin = changed[ 0 ];
            //if( SmallCycleBoolin )
                BlockIterDevice[ blIdy * numOfBlockx + blIdx ] = SmallCycleBoolin;
        }
        __syncthreads();
    }
        
    if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() )
        aux[ j * mesh.getDimensions().x() + i ] = sArray[ thrj + 1 ][ thri + 1 ];
        
}
#endif
