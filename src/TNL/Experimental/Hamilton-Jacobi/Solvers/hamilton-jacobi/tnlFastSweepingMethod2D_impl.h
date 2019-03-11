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
#include "tnlDirectEikonalProblem.h"
#include <TNL/Devices/Cuda.h>
#include <TNL/Communicators/MpiDefs.h>
#include "tnlDirectEikonalProblem.h"




#include <string.h>
#include <iostream>
#include <fstream>

template< typename Real,
        typename Device,
        typename Index,
        typename Communicator,
        typename Anisotropy >
FastSweepingMethod< Meshes::Grid< 2, Real, Device, Index >, Communicator, Anisotropy >::
FastSweepingMethod()
: maxIterations( 1 )
{
  
}

template< typename Real,
        typename Device,
        typename Index,
        typename Communicator,
        typename Anisotropy >
const Index&
FastSweepingMethod< Meshes::Grid< 2, Real, Device, Index >, Communicator, Anisotropy >::
getMaxIterations() const
{
  
}

template< typename Real,
        typename Device,
        typename Index,
        typename Communicator,
        typename Anisotropy >
void
FastSweepingMethod< Meshes::Grid< 2, Real, Device, Index >, Communicator, Anisotropy >::
setMaxIterations( const IndexType& maxIterations )
{
  
}

template< typename Real,
        typename Device,
        typename Index,
        typename Communicator,
        typename Anisotropy > 
void
FastSweepingMethod< Meshes::Grid< 2, Real, Device, Index >, Communicator, Anisotropy >::
solve( const MeshPointer& mesh,
        MeshFunctionPointer& Aux,
        const AnisotropyPointer& anisotropy,
        const MeshFunctionPointer& u )
{  
  MeshFunctionPointer auxPtr;
  InterfaceMapPointer interfaceMapPtr;
  auxPtr->setMesh( mesh );
  interfaceMapPtr->setMesh( mesh );
  
  //Distributed mesh for MPI overlaps (without MPI null pointer)
  Meshes::DistributedMeshes::DistributedMesh< MeshType >* meshPom = mesh->getDistributedMesh();
  
  int i = MPI::GetRank( MPI::AllGroup ); // number that identifies rank
  
  // getting overlaps ( WITHOUT MPI SHOULD BE 0 )
  Containers::StaticVector< 2, IndexType > vLower;
  vLower[0] = 0; vLower[1] = 0;
  Containers::StaticVector< 2, IndexType > vUpper;
  vUpper[0] = 0; vUpper[1] = 0;
#ifdef HAVE_MPI
  if( CommunicatorType::isDistributed() ) //If we started solver with MPI
  {
    vLower = meshPom->getLowerOverlap();
    vUpper = meshPom->getUpperOverlap();
  }
#endif
  
  std::cout << "Initiating the interface cells ..." << std::endl;
  BaseType::initInterface( u, auxPtr, interfaceMapPtr, vLower, vUpper );
  
  auxPtr->save( "aux-ini.tnl" );
  
  typename MeshType::Cell cell( *mesh );
  
  IndexType iteration( 0 );
  InterfaceMapType interfaceMap = *interfaceMapPtr;
  MeshFunctionType aux = *auxPtr;
  aux.template synchronize< Communicator >();
  
  
#ifdef HAVE_MPI
  int i = Communicators::MpiCommunicator::GetRank( Communicators::MpiCommunicator::AllGroup );
  //printf( "Hello world from rank: %d ", i );
  //Communicators::MpiCommunicator::Request r = Communicators::MpiCommunicator::ISend( auxPtr, 0, 0, Communicators::MpiCommunicator::AllGroup );
  if( i == 1 ) {
    /*for( int k = 0; k < 16*16; k++ )
      aux[ k ] = 10;*/
    printf( "1: mesh x: %d\n", mesh->getDimensions().x() );
    printf( "1: mesh y: %d\n", mesh->getDimensions().y() );
    //aux.save("aux_proc1.tnl");
  }
  if( i == 0 ) {
    printf( "0: mesh x: %d\n", mesh->getDimensions().x() );
    printf( "0: mesh y: %d\n", mesh->getDimensions().y() );
    //aux.save("aux_proc0.tnl");
    /*for( int k = 0; k < mesh->getDimensions().x()*mesh->getDimensions().y(); k++ )
      aux[ k ] = 10;
    for( int k = 0; k < mesh->getDimensions().x(); k++ ){
      for( int l = 0; l < mesh->getDimensions().y(); l++ )
        printf("%f.2\t",aux[ k * 16 + l ] );
    printf("\n");
    }*/
  }
    
  /*bool a = Communicators::MpiCommunicator::IsInitialized();
  if( a )
    printf("Je Init\n");
  else
    printf("Neni Init\n");*/
#endif
  
  while( iteration < this->maxIterations )
  {    
#if  ForDebug 
    int WhileCount = 0; // number of passages of while cycle with condition calculated
    printf( "%d: meshDimensions are (x,y) = (%d,%d).\n",i, mesh->getDimensions().x(), mesh->getDimensions().y() );
    printf( "%d: owerlaps are ([x1,x2],[y1,y2]) = ([%d,%d],[%d,%d]).\n",i, vLower[0], vUpper[0], vLower[1], vUpper[1] );
    /*if( std::is_same< DeviceType, Devices::Host >::value && i == 0 )
    {
      for( int j = mesh->getDimensions().y()-1; j>-1; j-- ){
        for( int m = 0; m < mesh->getDimensions().x(); m++ )
          std::cout << aux[ j * mesh->getDimensions().x() + m ] << " ";
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }*/
    
    // TO SEE CUDA OVERLAPS
    /*const int cudaBlockSize( 16 );
    int numBlocksXWithoutOverlaps = Devices::Cuda::getNumberOfBlocks( mesh->getDimensions().x(), cudaBlockSize );
    int numBlocksYWithoutOverlaps = Devices::Cuda::getNumberOfBlocks( mesh->getDimensions().y(), cudaBlockSize );
    dim3 gridSizeWithoutOverlaps( numBlocksXWithoutOverlaps, numBlocksYWithoutOverlaps );
    dim3 blockSize( cudaBlockSize, cudaBlockSize );
    MeshFunctionPointer helpFunc( mesh );
    DeepCopy<<< gridSizeWithoutOverlaps, blockSize >>>( helpFunc.template getData< Device>(),
            auxPtr.template modifyData< Device>(), 1, i ); */
    
#endif
    
    int calculated = 1; // indicates weather we calculated in the last passage of the while cycle 
    // calculated is same for all ranks 
    // without MPI should be FALSE at the end of while cycle body
    int calculate = 1; // indicates if the thread should calculate again in upcoming passage of cycle
    // calculate is a value that can differ in every rank
    // without MPI should be FALSE at the end of while cycle body
    
    while( calculated )
    {
      calculated = 0;
#if ForDebug
      WhileCount++;
      /*if( std::is_same< DeviceType, Devices::Cuda >::value )
      {
        DeepCopy<<< gridSizeWithoutOverlaps, blockSize >>>( auxPtr.template getData< Device>(),
                helpFunc.template modifyData< Device>(), 0, i );
      }*/
#endif
      
      if( std::is_same< DeviceType, Devices::Host >::value && calculate ) // should we calculate in Host?
      {
        calculate = 0;
        
        /**--HERE-IS-PARALLEL-OMP-CODE--!!!WITHOUT MPI!!!--------------------**/
        /*
         int numThreadsPerBlock = -1;
         
         numThreadsPerBlock = ( mesh->getDimensions().x()/2 + (mesh->getDimensions().x() % 2 != 0 ? 1:0));
         //printf("numThreadsPerBlock = %d\n", numThreadsPerBlock);
         if( numThreadsPerBlock <= 16 )
         numThreadsPerBlock = 16;
         else if(numThreadsPerBlock <= 32 )
         numThreadsPerBlock = 32;
         else if(numThreadsPerBlock <= 64 )
         numThreadsPerBlock = 64;
         else if(numThreadsPerBlock <= 128 )
         numThreadsPerBlock = 128;
         else if(numThreadsPerBlock <= 256 )
         numThreadsPerBlock = 256;
         else if(numThreadsPerBlock <= 512 )
         numThreadsPerBlock = 512;
         else
         numThreadsPerBlock = 1024;
         //printf("numThreadsPerBlock = %d\n", numThreadsPerBlock);
         
         if( numThreadsPerBlock == -1 ){
         printf("Fail in setting numThreadsPerBlock.\n");
         break;
         }
         
         
         
         int numBlocksX = mesh->getDimensions().x() / numThreadsPerBlock + (mesh->getDimensions().x() % numThreadsPerBlock != 0 ? 1:0);
         int numBlocksY = mesh->getDimensions().y() / numThreadsPerBlock + (mesh->getDimensions().y() % numThreadsPerBlock != 0 ? 1:0);
         
         //std::cout << "numBlocksX = " << numBlocksX << std::endl;
         
         //Real **sArray = new Real*[numBlocksX*numBlocksY];
         //for( int i = 0; i < numBlocksX * numBlocksY; i++ )
         // sArray[ i ] = new Real [ (numThreadsPerBlock + 2)*(numThreadsPerBlock + 2)];
         
         ArrayContainer BlockIterHost;
         BlockIterHost.setSize( numBlocksX * numBlocksY );
         BlockIterHost.setValue( 1 );
         int IsCalculationDone = 1;
         
         MeshFunctionPointer helpFunc( mesh );
         MeshFunctionPointer helpFunc1( mesh );
         helpFunc1 = auxPtr;
         auxPtr = helpFunc;
         helpFunc = helpFunc1;
         //std::cout<< "Size = " << BlockIterHost.getSize() << std::endl;
         //for( int k = numBlocksX-1; k >-1; k-- ){
         // for( int l = 0; l < numBlocksY; l++ ){
         // std::cout<< BlockIterHost[ l*numBlocksX  + k ];
         // }
         // std::cout<<std::endl;
         // }
         // std::cout<<std::endl;
         unsigned int numWhile = 0;
         while( IsCalculationDone )
         {      
         IsCalculationDone = 0;
         helpFunc1 = auxPtr;
         auxPtr = helpFunc;
         helpFunc = helpFunc1;
         switch ( numThreadsPerBlock ){
         case 16:
         this->template updateBlocks< 18 >( interfaceMap, *auxPtr, *helpFunc, BlockIterHost, numThreadsPerBlock );
         case 32:
         this->template updateBlocks< 34 >( interfaceMap, *auxPtr, *helpFunc, BlockIterHost, numThreadsPerBlock );
         case 64:
         this->template updateBlocks< 66 >( interfaceMap, *auxPtr, *helpFunc, BlockIterHost, numThreadsPerBlock );
         case 128:
         this->template updateBlocks< 130 >( interfaceMap, *auxPtr, *helpFunc, BlockIterHost, numThreadsPerBlock );
         case 256:
         this->template updateBlocks< 258 >( interfaceMap, *auxPtr, *helpFunc, BlockIterHost, numThreadsPerBlock );
         case 512:
         this->template updateBlocks< 514 >( interfaceMap, *auxPtr, *helpFunc, BlockIterHost, numThreadsPerBlock );
         default:
         this->template updateBlocks< 1028 >( interfaceMap, *auxPtr, *helpFunc, BlockIterHost, numThreadsPerBlock );
         }
         
         
         //Reduction      
         for( int i = 0; i < BlockIterHost.getSize(); i++ ){
         if( IsCalculationDone == 0 ){
         IsCalculationDone = IsCalculationDone || BlockIterHost[ i ];
         //break;
         }
         }
         numWhile++;
         //std::cout <<"numWhile = "<< numWhile <<std::endl;
         
         // for( int j = numBlocksY-1; j>-1; j-- ){
         // for( int i = 0; i < numBlocksX; i++ )
         // std::cout << BlockIterHost[ j * numBlocksX + i ];
         // std::cout << std::endl;
         // }
         // std::cout << std::endl;
         
         this->getNeighbours( BlockIterHost, numBlocksX, numBlocksY );
         
         //std::cout<<std::endl;
         //String s( "aux-"+ std::to_string(numWhile) + ".tnl");
         //aux.save( s );
         }
         if( numWhile == 1 ){
         auxPtr = helpFunc;
         }
         */
        /**-END-OF-OMP-PARALLEL------------------------------------------------**/
        
        
        /*if( i == 1 )
         {
         for( int k = 0; k < mesh->getDimensions().y(); k++ ){
         for( int l = 0; l < mesh->getDimensions().x(); l++ )
         printf("%.2f\t",aux[ k * mesh->getDimensions().x() + l ] );
         printf("\n");
         }
         }*/
        
  // FSM FOR MPI and WITHOUT MPI
        for( cell.getCoordinates().y() = 0 + vLower[1];
                cell.getCoordinates().y() < mesh->getDimensions().y() - vUpper[1];
                cell.getCoordinates().y()++ )
        {
          for( cell.getCoordinates().x() = 0 + vLower[0];
                  cell.getCoordinates().x() < mesh->getDimensions().x() - vUpper[0];
                  cell.getCoordinates().x()++ )
          {
            cell.refresh();
            if( ! interfaceMap( cell ) )
            {
              calculated = this->updateCell( aux, cell ) || calculated;
            }
          }
        }
        
        for( cell.getCoordinates().y() = 0 + vLower[1];
                cell.getCoordinates().y() < mesh->getDimensions().y()-vUpper[1];
                cell.getCoordinates().y()++ )
        {
          for( cell.getCoordinates().x() = mesh->getDimensions().x() - 1 - vUpper[0];
                  cell.getCoordinates().x() >= 0 + vLower[0];
                  cell.getCoordinates().x()-- )		
          {
            //std::cerr << "2 -> ";
            cell.refresh();
            if( ! interfaceMap( cell ) )            
              this->updateCell( aux, cell );
          }
        }
        
        //aux.save( "aux-2.tnl" );
        
        for( cell.getCoordinates().y() = mesh->getDimensions().y() - 1 -vUpper[1];
                cell.getCoordinates().y() >= 0 + vLower[1] ;
                cell.getCoordinates().y()-- )
        {
          for( cell.getCoordinates().x() = 0 + vLower[0];
                  cell.getCoordinates().x() < mesh->getDimensions().x() - vUpper[0];
                  cell.getCoordinates().x()++ )
          {
            //std::cerr << "3 -> ";
            cell.refresh();
            if( ! interfaceMap( cell ) )            
              this->updateCell( aux, cell );
          }
        }
        
        //aux.save( "aux-3.tnl" );
        
        for( cell.getCoordinates().y() = mesh->getDimensions().y() - 1 - vUpper[1];
                cell.getCoordinates().y() >= 0 + vLower[1];
                cell.getCoordinates().y()-- )
        {
          for( cell.getCoordinates().x() = mesh->getDimensions().x() - 1 - vUpper[0];
                  cell.getCoordinates().x() >= 0 + vLower[0];
                  cell.getCoordinates().x()-- )		
          {
            //std::cerr << "4 -> ";
            cell.refresh();
            if( ! interfaceMap( cell ) )            
              this->updateCell( aux, cell );
          }
        }
      }
      
      if( std::is_same< DeviceType, Devices::Cuda >::value && calculate ) // should we calculate on CUDA?
      {
        calculate = 0;
        
#if ForDebug 
        printf("%d: We are in Cuda code start.\n", i);
#endif
          
#ifdef HAVE_CUDA
        TNL_CHECK_CUDA_DEVICE;
        // Maximum cudaBlockSite is 32. Because of maximum num. of threads in kernel.
        // IF YOU CHANGE THIS, YOU NEED TO CHANGE THE TEMPLATE PARAMETER IN CudaUpdateCellCaller (The Number + 2)
        const int cudaBlockSize( 16 );
        
        // Setting number of threads and blocks for kernel
        int numBlocksX = Devices::Cuda::getNumberOfBlocks( mesh->getDimensions().x() - vLower[0] - vUpper[0], cudaBlockSize );
        int numBlocksY = Devices::Cuda::getNumberOfBlocks( mesh->getDimensions().y() - vLower[1] - vUpper[1], cudaBlockSize );
        dim3 blockSize( cudaBlockSize, cudaBlockSize );
        dim3 gridSize( numBlocksX, numBlocksY );
        
        // Need for calling functions from kernel
        BaseType ptr;
        
        int BlockIterD = 1;
        /*auxPtr = helpFunc;
         
         CudaUpdateCellCaller<18><<< gridSize, blockSize >>>( ptr,
         interfaceMapPtr.template getData< Device >(),
         auxPtr.template getData< Device>(),
         helpFunc.template modifyData< Device>(),
         BlockIterDevice,
         oddEvenBlock.getView() );
         cudaDeviceSynchronize();
         TNL_CHECK_CUDA_DEVICE;
         auxPtr = helpFunc;
         
         oddEvenBlock= (oddEvenBlock == 0) ? 1: 0;
         
         CudaUpdateCellCaller<18><<< gridSize, blockSize >>>( ptr,
         interfaceMapPtr.template getData< Device >(),
         auxPtr.template getData< Device>(),
         helpFunc.template modifyData< Device>(),
         BlockIterDevice,
         oddEvenBlock.getView() );
         cudaDeviceSynchronize();
         TNL_CHECK_CUDA_DEVICE;
         auxPtr = helpFunc;
         
         oddEvenBlock= (oddEvenBlock == 0) ? 1: 0;
         
         CudaParallelReduc<<< nBlocks , 1024 >>>( BlockIterDevice.getView(), dBlock.getView(), ( numBlocksX * numBlocksY ) );
         cudaDeviceSynchronize();
         TNL_CHECK_CUDA_DEVICE;
         CudaParallelReduc<<< 1, nBlocks >>>( dBlock.getView(), dBlock.getView(), nBlocks );
         cudaDeviceSynchronize();
         TNL_CHECK_CUDA_DEVICE;
         
         BlockIterD = dBlock.getElement( 0 );*/
        
        // Array that identifies which blocks should be calculated.
        TNL::Containers::Array< int, Devices::Cuda, IndexType > BlockIterDevice;
        BlockIterDevice.setSize( numBlocksX * numBlocksY );
        BlockIterDevice.setValue( 1 );
        TNL_CHECK_CUDA_DEVICE;
        
        // Array into which we identify the neighbours and then copy it into BlockIterDevice
        TNL::Containers::Array< int, Devices::Cuda, IndexType > BlockIterPom;
        BlockIterPom.setSize( numBlocksX * numBlocksY  );
        BlockIterPom.setValue( 0 );
        
#if ForDebug // For printf of BlockIterDevice
        TNL::Containers::Array< int, Devices::Host, IndexType > BlockIterPom1;
        BlockIterPom1.setSize( numBlocksX * numBlocksY  );
        BlockIterPom1.setValue( 0 );
#endif   
        int nBlocksNeigh = ( numBlocksX * numBlocksY )/1024 + ((( numBlocksX * numBlocksY )%1024 != 0) ? 1:0);
        // for CudaPrallelReduc (replaced with .containsValue(1))
        //int nBlocks = ( numBlocksX * numBlocksY )/1024 + ((( numBlocksX * numBlocksY )%1024 != 0) ? 1:0);
        //TNL::Containers::Array< int, Devices::Cuda, IndexType > dBlock;
        //dBlock.setSize( nBlocks );
        //TNL::Containers::Array< int, Devices::Host, IndexType > dBlock1;
        //dBlock1.setSize( nBlocks );
        //TNL_CHECK_CUDA_DEVICE;
        
        // Helping meshFunction that switches with AuxPtr in every calculation of CudaUpdateCellCaller<<<>>>()
        MeshFunctionPointer helpFunc( mesh );
        helpFunc.template modifyData() = auxPtr.template getData();
        Devices::Cuda::synchronizeDevice(); 
        //MeshFunctionPointer helpFunc1( mesh );
        
        // Setting number of threads and blocks in grid for DeepCopy of meshFunction
        /*int numBlocksXWithoutOverlaps = Devices::Cuda::getNumberOfBlocks( mesh->getDimensions().x(), cudaBlockSize );
        int numBlocksYWithoutOverlaps = Devices::Cuda::getNumberOfBlocks( mesh->getDimensions().y(), cudaBlockSize );
        dim3 gridSizeWithoutOverlaps( numBlocksXWithoutOverlaps, numBlocksYWithoutOverlaps );
        
        
          Devices::Cuda::synchronizeDevice();
        DeepCopy<<< gridSizeWithoutOverlaps, blockSize >>>( auxPtr.template getData< Device>(),
                helpFunc.template modifyData< Device>(), 1, i );
          cudaDeviceSynchronize();
          TNL_CHECK_CUDA_DEVICE;
          Devices::Cuda::synchronizeDevice();
        DeepCopy<<< gridSizeWithoutOverlaps, blockSize >>>( auxPtr.template getData< Device>(),
                helpFunc.template modifyData< Device>(), 0, i );
          cudaDeviceSynchronize();
          TNL_CHECK_CUDA_DEVICE;*/
        
#if ForDebug
        /*int numBlocksXWithoutOverlaps = Devices::Cuda::getNumberOfBlocks( mesh->getDimensions().x(), cudaBlockSize );
        int numBlocksYWithoutOverlaps = Devices::Cuda::getNumberOfBlocks( mesh->getDimensions().y(), cudaBlockSize );
        dim3 gridSizeWithoutOverlaps( numBlocksXWithoutOverlaps, numBlocksYWithoutOverlaps );*/
        DeepCopy<<< gridSizeWithoutOverlaps, blockSize >>>( auxPtr.template getData< Device>(),
                helpFunc.template modifyData< Device>(), 0, i );
#endif
        
        //int pocBloku = 0;
        Devices::Cuda::synchronizeDevice();
        CudaUpdateCellCaller<18><<< gridSize, blockSize >>>( ptr,
                interfaceMapPtr.template getData< Device >(),
                auxPtr.template modifyData< Device>(),
                helpFunc.template modifyData< Device>(),
                BlockIterDevice.getView() );
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
        
        //int oddEvenBlock = 0;
        //int numberWhile = 0;
        while( BlockIterD )
        {
          //numberWhile++;
          /** HERE IS CHESS METHOD (NO MPI) **/
          
          /*
           CudaUpdateCellCaller<18><<< gridSize, blockSize >>>( ptr,
           interfaceMapPtr.template getData< Device >(),
           auxPtr.template getData< Device>(),
           helpFunc.template modifyData< Device>(),
           BlockIterDevice,
           oddEvenBlock );
           cudaDeviceSynchronize();
           TNL_CHECK_CUDA_DEVICE;
           
           oddEvenBlock= (oddEvenBlock == 0) ? 1: 0;
           
           CudaUpdateCellCaller<18><<< gridSize, blockSize >>>( ptr,
           interfaceMapPtr.template getData< Device >(),
           helpFunc.template getData< Device>(),
           auxPtr.template modifyData< Device>(),
           BlockIterDevice, vLower, vUpper, 
           oddEvenBlock );
           cudaDeviceSynchronize();
           TNL_CHECK_CUDA_DEVICE;
           
           oddEvenBlock= (oddEvenBlock == 0) ? 1: 0;
           
           CudaParallelReduc<<< nBlocks , 1024 >>>( BlockIterDevice, dBlock, ( numBlocksX * numBlocksY ) );
           cudaDeviceSynchronize();
           TNL_CHECK_CUDA_DEVICE;
           CudaParallelReduc<<< 1, nBlocks >>>( dBlock, dBlock, nBlocks );
           cudaDeviceSynchronize();
           TNL_CHECK_CUDA_DEVICE;
           
           BlockIterD = dBlock.getElement( 0 );*/
          
          /**------------------------------------------------------------------------------------------------*/
          
          
     /** HERE IS FIM FOR MPI AND WITHOUT MPI **/
          Devices::Cuda::synchronizeDevice();
          CudaUpdateCellCaller<18><<< gridSize, blockSize >>>( ptr,
                  interfaceMapPtr.template getData< Device >(),
                  auxPtr.template getData< Device>(),
                  helpFunc.template modifyData< Device>(),
                  BlockIterDevice, vLower, vUpper, i );
          cudaDeviceSynchronize();
          TNL_CHECK_CUDA_DEVICE;
        
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
        
        
        aux = *auxPtr;
        interfaceMap = *interfaceMapPtr;
#endif
      }

      
/**----------------------MPI-TO-DO---------------------------------------------**/
        
#ifdef HAVE_MPI
        //int i = MPI::GetRank( MPI::AllGroup );
        //TNL::Meshes::DistributedMeshes::DistributedMesh< MeshType > Mesh;
        int neighCount = 0; // should this thread calculate again?
        int calculpom[4] = {0,0,0,0};
        
          if( i == 0 ){
            BlockIterPom1 = BlockIterDevice;
            for( int i =0; i< numBlocksX; i++ ){
              for( int j = 0; j < numBlocksY; j++ )
              {
                std::cout << BlockIterPom1[j*numBlocksX + i];
              }
              std::cout << std::endl;
            }
            std::cout << std::endl;
          }
#endif
          
          // Getting blocks that should calculate in next passage. These blocks are neighbours of those that were calculated now.
          Devices::Cuda::synchronizeDevice(); 
          GetNeighbours<<< nBlocksNeigh, 1024 >>>( BlockIterDevice, BlockIterPom, numBlocksX, numBlocksY );
          cudaDeviceSynchronize();
          TNL_CHECK_CUDA_DEVICE;
          BlockIterDevice = BlockIterPom;
          cudaDeviceSynchronize();
          TNL_CHECK_CUDA_DEVICE;
#if ForDebug          
          if( i == 1 ){
            BlockIterPom1 = BlockIterDevice;
            for( int i =0; i< numBlocksX; i++ ){
              for( int j = 0; j < numBlocksY; j++ )
              {
                std::cout << BlockIterPom1[j*numBlocksX + i];
              }
              std::cout << std::endl;
            }
            std::cout << std::endl;
          }
#endif
          // "Parallel reduction" to see if we should calculate again BlockIterD
          BlockIterD = BlockIterDevice.containsValue(1);
          /*Devices::Cuda::synchronizeDevice();
          CudaParallelReduc<<< nBlocks , 1024 >>>( BlockIterDevice, dBlock, ( numBlocksX * numBlocksY ) );
          cudaDeviceSynchronize();
          TNL_CHECK_CUDA_DEVICE;
          
          // Parallel reduction on dBlock because of too large number of blocks (more than maximum number of threads)
          Devices::Cuda::synchronizeDevice();
          CudaParallelReduc<<< 1, 1024 >>>( dBlock, dBlock, nBlocks );
          cudaDeviceSynchronize();
          TNL_CHECK_CUDA_DEVICE;*/
          
          // Copy of the first element which is result of parallel reduction
          /*Devices::Cuda::synchronizeDevice();
          BlockIterD = dBlock.getElement( 0 );
          cudaDeviceSynchronize();
          TNL_CHECK_CUDA_DEVICE;*/
          
          // When we change something then we should caclucate again in the next passage of MPI ( calculated = true )
         
          
          if( BlockIterD ){
            calculated = 1;
          }
          
          /**-----------------------------------------------------------------------------------------------------------*/
       
          numIter ++;
        }
        if( numIter%2  == 1 ){
          auxPtr = helpFunc;
        }
        /*cudaFree( BlockIterDevice );
         cudaFree( dBlock );
         delete BlockIter;*/
        
        if( neigh[1] != -1 )
        {
          req[neighCount] = MPI::ISend( &calculated, 1, neigh[1], 0, MPI::AllGroup ); 
          neighCount++;
          
          
          req[neighCount] = MPI::IRecv( &calculpom[1], 1, neigh[1], 0, MPI::AllGroup );
          neighCount++;
        }
        
        if( neigh[2] != -1 )
        {
          req[neighCount] = MPI::ISend( &calculated, 1, neigh[2], 0, MPI::AllGroup );
          neighCount++;
          
          req[neighCount] = MPI::IRecv( &calculpom[2], 1, neigh[2], 0, MPI::AllGroup  );
          neighCount++;
        }
        
        if( neigh[5] != -1 )
        {
          req[neighCount] = MPI::ISend( &calculated, 1, neigh[5], 0, MPI::AllGroup );
          neighCount++;
          
          req[neighCount] = MPI::IRecv( &calculpom[3], 1, neigh[5], 0, MPI::AllGroup );
          neighCount++;
        }
        
        MPI::WaitAll(req,neighCount);
#if ForDebug
        printf( "%d: Sending Calculated = %d.\n", i, calculated );
#endif        
        MPI::Allreduce( &calculated, &calculated, 1, MPI_LOR,  MPI::AllGroup );
        aux.template synchronize< Communicator >();
        calculate = calculpom[0] || calculpom[1] || calculpom[2] || calculpom[3];
#if ForDebug
        printf( "%d: Receved Calculated = %d.\n%d: Calculate = %d\n", i, calculated, i, calculate);
#endif
        
#if ForDebug 
        if( i == 1 )
          printf("WhileCount = %d\n",WhileCount);
        //calculated = 0; // DEBUG;
#endif
      }
#endif
      if( !CommunicatorType::isDistributed() ) // If we start the solver without MPI, we need calculated 0!
        calculated = 0;
    }
    iteration++;
  }
  //String s( "aux-" + std::to_string( i ) + ".tnl" );
  //aux.save( s );   
  Aux=auxPtr; // copy it for MakeSnapshot
  
  aux.save("aux-final.tnl");
}


#ifdef HAVE_CUDA
// DeepCopy nebo pracne kopirovat kraje v zavislosti na vLower,vUpper z sArray do helpFunc.
template< typename Real, typename Device, typename Index >
__global__ void DeepCopy( const Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& aux,
        Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& helpFunc, int copy, int k )
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = blockDim.y*blockIdx.y + threadIdx.y;
  const Meshes::Grid< 2, Real, Device, Index >& mesh = aux.template getMesh< Devices::Cuda >();
  if( copy ){
    if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() )
      helpFunc[ j * mesh.getDimensions().x() + i ] = 1;//aux[ j * mesh.getDimensions().x() + i ];
  }
  else
  {
    if( i==0 && j == 0 && blockIdx.x == 0 && blockIdx.y == 0 && k == 3 )
    {
      for( int m = mesh.getDimensions().y()-1; m>-1; m-- ){
        for( int l = 0; l < 17; l++ ){
          printf( "%.2f ", aux[ m * mesh.getDimensions().x() + l ]);
        }
        printf( "\n");
      }
      printf( "\n");
    }
    if( i==0 && j == 0 && blockIdx.x == 0 && blockIdx.y == 0 && k == 3 )
    {
      for( int m = mesh.getDimensions().y()-1; m>-1; m-- ){
        for( int l = 0; l < 17; l++ ){
          printf( "%.2f ", helpFunc[ m * mesh.getDimensions().x() + l ]);
        }
        printf( "\n");
      }
      printf( "\n");
    }
  }
}

template < typename Index >
__global__ void GetNeighbours( TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterDevice,
        TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterPom, int numBlockX, int numBlockY )
{
  int i = blockIdx.x * 1024 + threadIdx.x;
  
  if( i < numBlockX * numBlockY )
  {
    int pom = 0;//BlockIterPom[ i ] = 0;
    int m=0, k=0;
    m = i%numBlockX;
    k = i/numBlockX;
    if( m > 0 && BlockIterDevice[ i - 1 ] ){
      pom = 1;//BlockIterPom[ i ] = 1;
    }else if( m < numBlockX -1 && BlockIterDevice[ i + 1 ] ){
      pom = 1;//BlockIterPom[ i ] = 1;
    }else if( k > 0 && BlockIterDevice[ i - numBlockX ] ){
      pom = 1;// BlockIterPom[ i ] = 1;
    }else if( k < numBlockY -1 && BlockIterDevice[ i + numBlockX ] ){
      pom = 1;//BlockIterPom[ i ] = 1;
    }
    
    if( BlockIterDevice[ i ] != 1 )
      BlockIterPom[ i ] = pom;//BlockIterPom[ i ];
    else
      BlockIterPom[ i ] = 1;
  }
}

template < typename Index >
__global__ void CudaParallelReduc( TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterDevice,
        TNL::Containers::ArrayView< int, Devices::Cuda, Index > dBlock, int nBlocks )
{
  int i = threadIdx.x;
  int blId = blockIdx.x;
  int blockSize = blockDim.x;
  /*if ( i == 0 && blId == 0 ){
    printf( "nBlocks = %d\n", nBlocks );
    for( int j = nBlocks-1; j > -1 ; j--){
      printf( "%d: cislo = %d \n", j, BlockIterDevice[ j ] );
    }
  }*/
  __shared__ int sArray[ 1024 ];
  sArray[ i ] = 0;
  if( blId * 1024 + i < nBlocks )
    sArray[ i ] = BlockIterDevice[ blId * 1024 + i ];
  __syncthreads();
  /*if ( i == 0 && blId == 0 ){
   printf( "nBlocks = %d\n", nBlocks );
   for( int j = 4; j > -1 ; j--){
   printf( "%d: cislo = %d \n", j, sArray[ j ] );
   }
  }*/
  /*extern __shared__ volatile int sArray[];
   unsigned int i = threadIdx.x;
   unsigned int gid = blockIdx.x * blockSize * 2 + threadIdx.x;
   unsigned int gridSize = blockSize * 2 * gridDim.x;
   sArray[ i ] = 0;
   while( gid < nBlocks )
   {
   sArray[ i ] += BlockIterDevice[ gid ] + BlockIterDevice[ gid + blockSize ];
   gid += gridSize;
   }
   __syncthreads();*/
  
  if ( blockSize == 1024) {
    if (i < 512)
      sArray[ i ] += sArray[ i + 512 ];
  }
  __syncthreads();
  if (blockSize >= 512) {
    if (i < 256) {
      sArray[ i ] += sArray[ i + 256 ];
    }
  }
  __syncthreads();
  if (blockSize >= 256) {
    if (i < 128) {
      sArray[ i ] += sArray[ i + 128 ];
    }
  }
  __syncthreads();
  if (blockSize >= 128) {
    if (i < 64) {
      sArray[ i ] += sArray[ i + 64 ];
    }
  }
  __syncthreads();
  if (i < 32 )
  {
    if(  blockSize >= 64 ){ sArray[ i ] += sArray[ i + 32 ];}
  __syncthreads();
    if(  blockSize >= 32 ){  sArray[ i ] += sArray[ i + 16 ];}
  __syncthreads();
    if(  blockSize >= 16 ){  sArray[ i ] += sArray[ i + 8 ];}
    if(  blockSize >= 8 ){  sArray[ i ] += sArray[ i + 4 ];}
  __syncthreads();
    if(  blockSize >= 4 ){  sArray[ i ] += sArray[ i + 2 ];}
  __syncthreads();
    if(  blockSize >= 2 ){  sArray[ i ] += sArray[ i + 1 ];}
  __syncthreads();
  }
  __syncthreads();
  
  if( i == 0 )
    dBlock[ blId ] = sArray[ 0 ];
}



template < int sizeSArray, typename Real, typename Device, typename Index >
__global__ void CudaUpdateCellCaller( tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > > ptr,
        const Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index >, 2, bool >& interfaceMap,
        const Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& aux,
        Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& helpFunc,
        CudaParallelReduc<<< nBlocks , 1024 >>>( BlockIterDevice.getView(), dBlock.getView(), ( numBlocksX * numBlocksY ) );
        TNL_CHECK_CUDA_DEVICE;
        
        CudaParallelReduc<<< 1, nBlocks >>>( dBlock.getView(), dBlock.getView(), nBlocks );
        TNL_CHECK_CUDA_DEVICE;
{
  int thri = threadIdx.x; int thrj = threadIdx.y;
  int i = threadIdx.x + blockDim.x*blockIdx.x + vLower[0];
  int j = blockDim.y*blockIdx.y + threadIdx.y + vLower[1];
  const Meshes::Grid< 2, Real, Device, Index >& mesh = aux.template getMesh< Devices::Cuda >();
  /** FOR CHESS METHOD */
  //if( (blockIdx.y%2  + blockIdx.x) % 2 == oddEvenBlock )
  //{
  /**------------------------------------------*/
  
  
  /** FOR FIM METHOD */
  
  if( BlockIterDevice[ blockIdx.y * gridDim.x + blockIdx.x ] )
  { 
    __syncthreads();
    
    /**-----------------------------------------*/
    __shared__ int dimX;
    __shared__ int dimY;
    __shared__ Real hx;
    __shared__ Real hy;
    if( thri==0 && thrj == 0)
    {
      dimX = mesh.getDimensions().x();
      dimY = mesh.getDimensions().y();
      hx = mesh.getSpaceSteps().x();
      hy = mesh.getSpaceSteps().y();
      BlockIterDevice[ blockIdx.y * gridDim.x + blockIdx.x ] = 0;
    }
    __syncthreads();
    int numOfBlockx;
    int numOfBlocky;
    int xkolik;
    int ykolik;
    
    xkolik = blockDim.x + 1;
    ykolik = blockDim.y + 1;
    numOfBlocky = gridDim.y;//(dimY-vUpper[1]-vLower[1])/blockDim.y + (((dimY-vUpper[1]-vLower[1])%blockDim.y != 0) ? 1:0);
    numOfBlockx = gridDim.x;//(dimX-vUpper[0]-vLower[0])/blockDim.x + (((dimX-vUpper[0]-vLower[0])%blockDim.x != 0) ? 1:0);
    
    if( numOfBlockx - 1 == blockIdx.x )
      xkolik = (dimX-vUpper[0]-vLower[0]) - (blockIdx.x)*blockDim.x+1;
    
    if( numOfBlocky -1 == blockIdx.y )
      ykolik = (dimY-vUpper[1]-vLower[1]) - (blockIdx.y)*blockDim.y+1;
    __syncthreads();
    
#if ForDebug
    /*if( thri==0 && thrj == 0 )
    {
      printf("%d: DimX = %d, DimY = %d, xKolik = %d, yKolik = %d, numOfBlockX = %d, numOfBlockY = %d, blockIdx.x = %d, blockIdx.y = %d.\n",
              k, dimX, dimY, xkolik, ykolik, numOfBlockx, numOfBlocky, blockIdx.x, blockIdx.y);
    }*/
#endif
    
    int currentIndex = thrj * blockDim.x + thri;
    //__shared__ volatile bool changed[ blockDim.x*blockDim.y ];
    __shared__ volatile bool changed[ (sizeSArray-2)*(sizeSArray-2)];
    changed[ currentIndex ] = false;
    if( thrj == 0 && thri == 0 )
      changed[ 0 ] = true;
    
    
    //__shared__ volatile Real sArray[ blockDim.y+2 ][ blockDim.x+2 ];
    __shared__ volatile Real sArray[ sizeSArray * sizeSArray ];
    sArray[ (thrj+1) * sizeSArray + thri +1 ] = std::numeric_limits< Real >::max();
    
       
        //filling sArray edges
    if( thri == 0 )
    {      
      if( dimX - vLower[ 0 ] > (blockIdx.x+1) * blockDim.x  && thrj+1 < ykolik )
        sArray[(thrj+1)*sizeSArray + xkolik] = aux[ (blockIdx.y*blockDim.y+vLower[1])*dimX - dimX + blockIdx.x*blockDim.x - 1 + (thrj+1)*dimX + xkolik + vLower[0] ];
      else
        sArray[(thrj+1)*sizeSArray + xkolik] = std::numeric_limits< Real >::max();
    }
        
    if( thri == 1 )
    { 
      if( ( blockIdx.x != 0 || vLower[0] != 0 ) && thrj+1 < ykolik )
        sArray[(thrj+1)*sizeSArray + 0] = aux[ (blockIdx.y*blockDim.y+vLower[1])*dimX - dimX + blockIdx.x*blockDim.x - 1 + (thrj+1)*dimX  + vLower[0] ];
      else
        sArray[(thrj+1)*sizeSArray + 0] = std::numeric_limits< Real >::max();
    }
    
    if( thri == 2 )
    {
      if( dimY - vLower[ 1 ] > (blockIdx.y+1) * blockDim.y  && thrj+1 < xkolik )
        sArray[ ykolik*sizeSArray + thrj+1 ] = aux[ (blockIdx.y*blockDim.y+vLower[1])*dimX - dimX + blockIdx.x*blockDim.x - 1 + ykolik*dimX + thrj+1 + vLower[0] ];
      else
        sArray[ ykolik*sizeSArray + thrj+1 ] = std::numeric_limits< Real >::max();
      
    }
        
    if( thri == 3 )
    {
      if( ( blockIdx.y != 0 || vLower[1] != 0 ) && thrj+1 < xkolik )
        sArray[0*sizeSArray + thrj+1] = aux[ (blockIdx.y*blockDim.y+vLower[1])*dimX - dimX + blockIdx.x*blockDim.x - 1 + thrj+1 + vLower[0] ];
      else
        sArray[0*sizeSArray + thrj+1] = std::numeric_limits< Real >::max();
    }
    /*__syncthreads();
    if( thri==0 && thrj == 0 && blockIdx.x == 0 && blockIdx.y == 0 && k == 1 )
    {
      printf( "Kraje: \n");
      for( int k = sizeSArray-1; k>-1; k-- ){
        for( int l = 0; l < sizeSArray; l++ )
          printf( "%.4f ", sArray[k * sizeSArray + l]);
        printf( "\n");
      }
      printf( "\n");
    }
    __syncthreads();*/
    
    
    if( i-vLower[0] < dimX && j-vLower[1] < dimY && thri+1 < xkolik + vUpper[0] && thrj+1 < ykolik + vUpper[1] )
    {  
      /*if( k == 1 && blockIdx.x == 0 && blockIdx.y == 0 )
        printf("at index = %d\n", j*dimX + i);*/
      sArray[(thrj+1)*sizeSArray + thri+1] = aux[ (j)*dimX + i ];
    }
    __syncthreads();  
#if ForDebug    
    if( thri==0 && thrj == 0 && blockIdx.x == 0 && blockIdx.y == 0 && k == 3 )
    {
      printf( "všechno před výpočtem: \n");
      for( int m = sizeSArray-1; m>-1; m-- ){
        for( int l = 0; l < sizeSArray; l++ )
          printf( "%.2f ", sArray[m * sizeSArray + l]);
        printf( "\n");
      }
      printf( "\n");
    }
    
    if( thri==0 && thrj == 0 && blockIdx.x == 0 && blockIdx.y == 0 && k == 3 )
    {
      for( int m = mesh.getDimensions().y()-1; m>-1; m-- ){
        for( int l = 0; l < 17; l++ )
          printf( "%.2f ", aux[ m * mesh.getDimensions().x() + l ]);
        printf( "\n");
      }
      printf( "\n");
    }
#endif 
    //main while cycle
    //if( i == 0 && j == 0 )
    //  printf("Overlaps [x1,y1],[x2,y2] = [%d,%d],[%d,%d]",vLower[0], vLower[1], vUpper[0], vUpper[1] );
    
    while( changed[ 0 ] )
    {
      __syncthreads();
      
      changed[ currentIndex] = false;
      
      //calculation of update cell
      if( i < dimX - vUpper[0] && j < dimY - vUpper[1] /*&& i > vLower[0]-1 && j > vLower[1]-1*/ )
      {
        if( ! interfaceMap[ j * dimX + i ] )
        {
          /*if( k == 1 && blockIdx.x == 1 && blockIdx.y == 0 )
            printf( "thri = %d, thrj = %d \n", thri, thrj );*/
          changed[ currentIndex ] = ptr.updateCell<sizeSArray>( sArray, thri+1, thrj+1, hx,hy);
        }
      }
      __syncthreads();
      
      //pyramid reduction
      if( blockDim.x*blockDim.y == 1024 )
      {
        if( currentIndex < 512 )
        {
          changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 512 ];
        }
      }
      __syncthreads();
      if( blockDim.x*blockDim.y >= 512 )
      {
        if( currentIndex < 256 )
        {
          changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 256 ];
        }
      }
      __syncthreads();
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
      if( currentIndex < 32 ) 
      {
        if( true ) changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 32 ];
        if( currentIndex < 16 ) changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 16 ];
        if( currentIndex < 8 ) changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 8 ];
        if( currentIndex < 4 ) changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 4 ];
        if( currentIndex < 2 ) changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 2 ];
        if( currentIndex < 1 ) changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 1 ];
      }
      if( thri == 0 && thrj == 0 && changed[ 0 ] ){
        BlockIterDevice[ blockIdx.y * gridDim.x + blockIdx.x ] = 1;
      }
      __syncthreads();
    }
    
    
      
    if( i < dimX && j < dimY && thri+1 < xkolik && thrj+1 < ykolik )
      helpFunc[ j * dimX + i ] = sArray[ ( thrj + 1 ) * sizeSArray + thri + 1 ];
    __syncthreads();
#if ForDebug
    if( thri==0 && thrj == 0 && blockIdx.x == 0 && blockIdx.y == 0 && k == 3 )
    {
      printf( "všechno po výpočtu: \n");
      for( int m = sizeSArray-1; m>-1; m-- ){
        for( int l = 0; l < sizeSArray; l++ )
          printf( "%.2f ", sArray[m * sizeSArray + l]);
        printf( "\n");
      }
      printf( "\n");
    }
    
    if( thri==0 && thrj == 0 && blockIdx.x == 0 && blockIdx.y == 0 && k == 3 )
    {
      printf( "8: \n");
      for( int m = mesh.getDimensions().y()-1; m>-1; m-- ){
        for( int l = 0; l < mesh.getDimensions().x(); l++ )
          printf( "%.2f ", helpFunc[ m * mesh.getDimensions().x() + l ]);
        printf("\n");
      }
      printf( "\n");
    }
#endif
  }
  else
  {
    if( i < mesh.getDimensions().x() - vUpper[0] && j < mesh.getDimensions().y() - vUpper[1] )
      helpFunc[ j * mesh.getDimensions().x() + i ] = aux[ j * mesh.getDimensions().x() + i ];
  }
}
#endif
        TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterDevice, int oddEvenBlock )
