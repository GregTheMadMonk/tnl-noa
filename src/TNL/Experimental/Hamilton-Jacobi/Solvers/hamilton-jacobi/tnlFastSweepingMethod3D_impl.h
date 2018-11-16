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
  auxPtr->save( "aux-ini.tnl" );   
  
  typename MeshType::Cell cell( *mesh );
  
  IndexType iteration( 0 );
  MeshFunctionType aux = *auxPtr;
  InterfaceMapType interfaceMap = * interfaceMapPtr;
  while( iteration < this->maxIterations )
  {
    if( std::is_same< DeviceType, Devices::Host >::value )
    {
      int numThreadsPerBlock = 64;
      
      
      int numBlocksX = mesh->getDimensions().x() / numThreadsPerBlock + (mesh->getDimensions().x() % numThreadsPerBlock != 0 ? 1:0);
      int numBlocksY = mesh->getDimensions().y() / numThreadsPerBlock + (mesh->getDimensions().y() % numThreadsPerBlock != 0 ? 1:0);
      int numBlocksZ = mesh->getDimensions().z() / numThreadsPerBlock + (mesh->getDimensions().z() % numThreadsPerBlock != 0 ? 1:0);
      //std::cout << "numBlocksX = " << numBlocksX << std::endl;
      
      /*Real **sArray = new Real*[numBlocksX*numBlocksY];
       for( int i = 0; i < numBlocksX * numBlocksY; i++ )
       sArray[ i ] = new Real [ (numThreadsPerBlock + 2)*(numThreadsPerBlock + 2)];*/
      
      ArrayContainer BlockIterHost;
      BlockIterHost.setSize( numBlocksX * numBlocksY * numBlocksZ );
      BlockIterHost.setValue( 1 );
      int IsCalculationDone = 1;
      
      MeshFunctionPointer helpFunc( mesh );
      MeshFunctionPointer helpFunc1( mesh );
      helpFunc1 = auxPtr;
      auxPtr = helpFunc;
      helpFunc = helpFunc1;
      //std::cout<< "Size = " << BlockIterHost.getSize() << std::endl;
      /*for( int k = numBlocksX-1; k >-1; k-- ){
       for( int l = 0; l < numBlocksY; l++ ){
       std::cout<< BlockIterHost[ l*numBlocksX  + k ];
       }
       std::cout<<std::endl;
       }
       std::cout<<std::endl;*/
      unsigned int numWhile = 0;
      while( IsCalculationDone  )
      {      
        IsCalculationDone = 0;
        helpFunc1 = auxPtr;
        auxPtr = helpFunc;
        helpFunc = helpFunc1;
        this->template updateBlocks< 66 >( interfaceMap, *auxPtr, *helpFunc, BlockIterHost, numThreadsPerBlock/*, sArray*/ );
        
        //Reduction      
        for( int i = 0; i < BlockIterHost.getSize(); i++ ){
          if( IsCalculationDone == 0 ){
            IsCalculationDone = IsCalculationDone || BlockIterHost[ i ];
            //break;
          }
        }
        numWhile++;
        std::cout <<"numWhile = "<< numWhile <<std::endl;
        /*for( int k = 0; k < numBlocksZ; k++ ){
          for( int j = numBlocksY-1; j>-1; j-- ){
            for( int i = 0; i < numBlocksX; i++ ){
              //std::cout << (*auxPtr)[ k * numBlocksX * numBlocksY + j * numBlocksX + i ] << " ";
              std::cout << BlockIterHost[ k * numBlocksX * numBlocksY + j * numBlocksX + i ];
            }
            std::cout << std::endl;
          }
          std::cout << std::endl;
        }
        std::cout << std::endl;*/
        
        this->getNeighbours( BlockIterHost, numBlocksX, numBlocksY, numBlocksZ );
        
        /*for( int k = 0; k < numBlocksZ; k++ ){
          for( int j = numBlocksY-1; j>-1; j-- ){
            for( int i = 0; i < numBlocksX; i++ ){
              //std::cout << (*auxPtr)[ k * numBlocksX * numBlocksY + j * numBlocksX + i ] << " ";
              std::cout << BlockIterHost[ k * numBlocksX * numBlocksY + j * numBlocksX + i ];
            }
            std::cout << std::endl;
          }
          std::cout << std::endl;
        }*/
        
        /*for( int j = numBlocksY-1; j>-1; j-- ){
         for( int i = 0; i < numBlocksX; i++ )
         std::cout << "BlockIterHost = "<< j*numBlocksX + i<< " ," << BlockIterHost[ j * numBlocksX + i ];
         std::cout << std::endl;
         }
         std::cout << std::endl;*/
        
        //std::cout<<std::endl;
        //string s( "aux-"+ std::to_string(numWhile) + ".tnl");
        //aux.save( s );
      }
      if( numWhile == 1 ){
        auxPtr = helpFunc;
      }
      aux = *auxPtr;
      
      /*for( cell.getCoordinates().z() = 0;
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
       }*/
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
      TNL::Containers::Array< int, Devices::Cuda, IndexType > BlockIterPom;
      BlockIterPom.setSize( numBlocksX * numBlocksY * numBlocksZ );
      BlockIterPom.setValue( 0 );
      /*int *BlockIterDevice;
       cudaMalloc(&BlockIterDevice, ( numBlocksX * numBlocksY * numBlocksZ ) * sizeof( int ) );*/
      int nBlocks = ( numBlocksX * numBlocksY * numBlocksZ )/512 + ((( numBlocksX * numBlocksY * numBlocksZ )%512 != 0) ? 1:0);
      
      TNL::Containers::Array< int, Devices::Cuda, IndexType > dBlock;
      dBlock.setSize( nBlocks );
      dBlock.setValue( 0 );
      
      int nBlocksNeigh = ( numBlocksX * numBlocksY * numBlocksZ )/1024 + ((( numBlocksX * numBlocksY * numBlocksZ )%1024 != 0) ? 1:0);
      /*int *dBlock;
       cudaMalloc(&dBlock, nBlocks * sizeof( int ) );*/
      MeshFunctionPointer helpFunc1( mesh );      
      MeshFunctionPointer helpFunc( mesh );
      
      helpFunc1 = auxPtr;
      auxPtr = helpFunc;
      helpFunc = helpFunc1;
      int numIter = 0;
      
      while( BlockIterD )
      {
        helpFunc1 = auxPtr;
        auxPtr = helpFunc;
        helpFunc = helpFunc1;
        TNL_CHECK_CUDA_DEVICE;
        
        CudaUpdateCellCaller< 10 ><<< gridSize, blockSize >>>( ptr,
                interfaceMapPtr.template getData< Device >(),
                auxPtr.template getData< Device>(),
                helpFunc.template modifyData< Device>(),
                BlockIterDevice );
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
        
        GetNeighbours3D<<< nBlocksNeigh, 1024 >>>( BlockIterDevice, BlockIterPom, numBlocksX, numBlocksY, numBlocksZ );
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
        BlockIterDevice = BlockIterPom;
        
        CudaParallelReduc<<< nBlocks , 512 >>>( BlockIterDevice, dBlock, ( numBlocksX * numBlocksY * numBlocksZ ) );
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
        
        CudaParallelReduc<<< 1, nBlocks >>>( dBlock, dBlock, nBlocks );
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
        cudaMemcpy(&BlockIterD, &dBlock[0], sizeof( int ), cudaMemcpyDeviceToHost);
        numIter++;
        /*for( int i = 1; i < numBlocksX * numBlocksY; i++ )
         BlockIter[ 0 ] = BlockIter[ 0 ] || BlockIter[ i ];*/
        
      }
      if( numIter == 1 ){
        auxPtr = helpFunc;
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
template < typename Index >
__global__ void GetNeighbours3D( TNL::Containers::Array< int, Devices::Cuda, Index > BlockIterDevice,
        TNL::Containers::Array< int, Devices::Cuda, Index > BlockIterPom,
        int numBlockX, int numBlockY, int numBlockZ )
{
  int i = blockIdx.x * 1024 + threadIdx.x;
  
  if( i < numBlockX * numBlockY * numBlockZ )
  {
    int pom = 0;//BlockIterPom[ i ] = 0;
    int m=0, l=0, k=0;
    l = i/( numBlockX * numBlockY );
    k = (i-l*numBlockX * numBlockY )/(numBlockX );
    m = (i-l*numBlockX * numBlockY )%( numBlockX );
    if( m > 0 && BlockIterDevice[ i - 1 ] ){
      pom = 1;//BlockIterPom[ i ] = 1;
    }else if( m < numBlockX -1 && BlockIterDevice[ i + 1 ] ){
      pom = 1;//BlockIterPom[ i ] = 1;
    }else if( k > 0 && BlockIterDevice[ i - numBlockX ] ){
      pom = 1;// BlockIterPom[ i ] = 1;
    }else if( k < numBlockY -1 && BlockIterDevice[ i + numBlockX ] ){
      pom = 1;//BlockIterPom[ i ] = 1;
    }else if( l > 0 && BlockIterDevice[ i - numBlockX*numBlockY ] ){
      pom = 1;
    }else if( l < numBlockZ-1 && BlockIterDevice[ i + numBlockX*numBlockY ] ){
      pom = 1;
    }
    
    BlockIterPom[ i ] = pom;//BlockIterPom[ i ];
  }
}

template < int sizeSArray, typename Real, typename Device, typename Index >
__global__ void CudaUpdateCellCaller( tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > > ptr,
        const Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index >, 3, bool >& interfaceMap,
        const Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index > >& aux,
        Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index > >& helpFunc,
        TNL::Containers::Array< int, Devices::Cuda, Index > BlockIterDevice )
{
  int thri = threadIdx.x; int thrj = threadIdx.y; int thrk = threadIdx.z;
  int blIdx = blockIdx.x; int blIdy = blockIdx.y; int blIdz = blockIdx.z;
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = blockDim.y*blockIdx.y + threadIdx.y;
  int k = blockDim.z*blockIdx.z + threadIdx.z;
  int currentIndex = thrk * blockDim.x * blockDim.y + thrj * blockDim.x + thri;
  
  if( BlockIterDevice[ blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x ] )
  {
    __syncthreads();
    
    __shared__ volatile bool changed[ 8*8*8/*(sizeSArray - 2)*(sizeSArray - 2)*(sizeSArray - 2)*/];
    
    changed[ currentIndex ] = false;
    if( thrj == 0 && thri == 0 && thrk == 0 )
      changed[ 0 ] = true;
    
    const Meshes::Grid< 3, Real, Device, Index >& mesh = interfaceMap.template getMesh< Devices::Cuda >();
    __shared__ Real hx; __shared__ int dimX;
    __shared__ Real hy; __shared__ int dimY;
    __shared__ Real hz; __shared__ int dimZ;
    
    if( thrj == 1 && thri == 1 && thrk == 1 )
    {
      //printf( "We are in the calculation. Block = %d.\n",blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x  );
      hx = mesh.getSpaceSteps().x();
      hy = mesh.getSpaceSteps().y();
      hz = mesh.getSpaceSteps().z();
      dimX = mesh.getDimensions().x();
      dimY = mesh.getDimensions().y();
      dimZ = mesh.getDimensions().z();
      BlockIterDevice[ blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x ] = 0;
    }
    __shared__ volatile Real sArray[ 10*10*10/*sizeSArray * sizeSArray * sizeSArray*/ ];
    sArray[(thrk+1)* sizeSArray * sizeSArray + (thrj+1) *sizeSArray + thri+1] = std::numeric_limits< Real >::max();
    
    //filling sArray edges
    int numOfBlockx;
    int numOfBlocky;
    int numOfBlockz;
    int xkolik;
    int ykolik;
    int zkolik;
    xkolik = blockDim.x + 1;
    ykolik = blockDim.y + 1;
    zkolik = blockDim.z + 1;
    numOfBlockx = gridDim.x;
    numOfBlocky = gridDim.y;
    numOfBlockz = gridDim.z;
    __syncthreads();
    
    if( numOfBlockx - 1 == blIdx )
      xkolik = dimX - (blIdx)*blockDim.x+1;
    if( numOfBlocky -1 == blIdy )
      ykolik = dimY - (blIdy)*blockDim.y+1;
    if( numOfBlockz-1 == blIdz )
      zkolik = dimZ - (blIdz)*blockDim.z+1;
    __syncthreads();
    
    if( thri == 0 )
    {        
      if( blIdx != 0 && thrj+1 < ykolik && thrk+1 < zkolik )
        sArray[(thrk+1 )* sizeSArray * sizeSArray + (thrj+1)*sizeSArray + 0] = aux[ blIdz*blockDim.z * dimX * dimY + blIdy * blockDim.y*dimX + blIdx*blockDim.x + thrj * dimX -1 + thrk*dimX*dimY ];
      else
        sArray[(thrk+1)* sizeSArray * sizeSArray + (thrj+1)*sizeSArray + 0] = std::numeric_limits< Real >::max();
    }
    
    if( thri == 1 )
    {
      if( dimX > (blIdx+1) * blockDim.x && thrj+1 < ykolik && thrk+1 < zkolik )
        sArray[ (thrk+1) * sizeSArray * sizeSArray + (thrj+1) *sizeSArray + xkolik ] = aux[ blIdz*blockDim.z * dimX * dimY + blIdy *blockDim.y*dimX+ blIdx*blockDim.x + blockDim.x + thrj * dimX + thrk*dimX*dimY ];
      else
        sArray[ (thrk+1) * sizeSArray * sizeSArray + (thrj+1)*sizeSArray + xkolik] = std::numeric_limits< Real >::max();
    }
    if( thri == 2 )
    {        
      if( blIdy != 0 && thrj+1 < xkolik && thrk+1 < zkolik )
        sArray[ (thrk+1) * sizeSArray * sizeSArray +0*sizeSArray + thrj+1] = aux[ blIdz*blockDim.z * dimX * dimY + blIdy * blockDim.y*dimX + blIdx*blockDim.x - dimX + thrj + thrk*dimX*dimY ];
      else
        sArray[ (thrk+1) * sizeSArray * sizeSArray + 0*sizeSArray + thrj+1] = std::numeric_limits< Real >::max();
    }
    
    if( thri == 3 )
    {
      if( dimY > (blIdy+1) * blockDim.y && thrj+1 < xkolik && thrk+1 < zkolik )
        sArray[ (thrk+1) * sizeSArray * sizeSArray + ykolik*sizeSArray + thrj+1] = aux[ blIdz*blockDim.z * dimX * dimY + (blIdy+1) * blockDim.y*dimX + blIdx*blockDim.x + thrj + thrk*dimX*dimY ];
      else
        sArray[ (thrk+1) * sizeSArray * sizeSArray + ykolik*sizeSArray + thrj+1] = std::numeric_limits< Real >::max();
    }
    if( thri == 4 )
    {        
      if( blIdz != 0 && thrj+1 < ykolik && thrk+1 < xkolik )
        sArray[ 0 * sizeSArray * sizeSArray +(thrj+1 )* sizeSArray + thrk+1] = aux[ blIdz*blockDim.z * dimX * dimY + blIdy * blockDim.y*dimX + blIdx*blockDim.x - dimX * dimY + thrj * dimX + thrk ];
      else
        sArray[0 * sizeSArray * sizeSArray + (thrj+1) *sizeSArray + thrk+1] = std::numeric_limits< Real >::max();
    }
    
    if( thri == 5 )
    {
      if( dimZ > (blIdz+1) * blockDim.z && thrj+1 < ykolik && thrk+1 < xkolik )
        sArray[zkolik * sizeSArray * sizeSArray + (thrj+1) * sizeSArray + thrk+1] = aux[ (blIdz+1)*blockDim.z * dimX * dimY + blIdy * blockDim.y*dimX + blIdx*blockDim.x + thrj * dimX + thrk ];
      else
        sArray[zkolik * sizeSArray * sizeSArray + (thrj+1) * sizeSArray + thrk+1] = std::numeric_limits< Real >::max();
    }
    
    if( i < dimX && j < dimY && k < dimZ )
    {
      sArray[(thrk+1) * sizeSArray * sizeSArray + (thrj+1) *sizeSArray + thri+1] = aux[ k*dimX*dimY + j*dimX + i ];
    }
    __syncthreads(); 
    
    while( changed[ 0 ] )
    {
      __syncthreads();
      
      changed[ currentIndex ] = false;
      
      //calculation of update cell
      if( i < dimX && j < dimY && k < dimZ )
      {
        if( ! interfaceMap[ k*dimX*dimY + j * dimX + i ] )
        {
          changed[ currentIndex ] = ptr.updateCell3D< sizeSArray >( sArray, thri+1, thrj+1, thrk+1, hx,hy,hz);
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
      if( currentIndex < 32 )
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
       //for(int m = 0; m < 8; m++){
       int m = 4;
       for(int n = 0; n<8; n++){
       for(int b=0; b<8; b++)
       printf(" %i ", changed[m*64 + n*8 + b]);
       printf("\n");
       }
       printf("\n \n");
       }
       //}*/
      
      if( changed[ 0 ] && thri == 0 && thrj == 0 && thrk == 0 )
      {
        //printf( "Setting block calculation. Block = %d.\n",blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x  );
        BlockIterDevice[ blIdz * gridDim.x * gridDim.y + blIdy * gridDim.x + blIdx ] = 1;
      }
      __syncthreads();
    }
    
    if( i < dimX && j < dimY && k < dimZ )
      helpFunc[ k*dimX*dimY + j * dimX + i ] = sArray[ (thrk+1) * sizeSArray * sizeSArray + (thrj+1) * sizeSArray + thri+1 ];
    
  } 
}  
#endif
