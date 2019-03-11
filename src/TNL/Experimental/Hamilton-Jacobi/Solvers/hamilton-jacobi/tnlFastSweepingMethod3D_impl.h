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
        typename Communicator,
        typename Anisotropy >
FastSweepingMethod< Meshes::Grid< 3, Real, Device, Index >, Communicator, Anisotropy >::
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
FastSweepingMethod< Meshes::Grid< 3, Real, Device, Index >, Communicator, Anisotropy >::
getMaxIterations() const
{
  
}

template< typename Real,
        typename Device,
        typename Index,
        typename Communicator,
        typename Anisotropy >
void
FastSweepingMethod< Meshes::Grid< 3, Real, Device, Index >, Communicator, Anisotropy >::
setMaxIterations( const IndexType& maxIterations )
{
  
}

template< typename Real,
        typename Device,
        typename Index,
        typename Communicator,
        typename Anisotropy >
void
FastSweepingMethod< Meshes::Grid< 3, Real, Device, Index >, Communicator, Anisotropy >::
solve( const MeshPointer& mesh,
        MeshFunctionPointer& Aux,
        const AnisotropyPointer& anisotropy,
        MeshFunctionPointer& u )
{
  MeshFunctionPointer auxPtr;
  InterfaceMapPointer interfaceMapPtr;
  auxPtr->setMesh( mesh );
  interfaceMapPtr->setMesh( mesh );
  
  //Distributed mesh for overlaps (without MPI is null pointer)
  Meshes::DistributedMeshes::DistributedMesh< MeshType >* meshPom = mesh->getDistributedMesh();
  
  // getting overlaps ( WITHOUT MPI SHOULD BE 0 )
  Containers::StaticVector< 3, IndexType > vLower;
  vLower[0] = 0; vLower[1] = 0; vLower[2] = 0;
  Containers::StaticVector< 3, IndexType > vUpper;
  vUpper[0] = 0; vUpper[1] = 0; vUpper[2] = 0;
#ifdef HAVE_MPI
  if( CommunicatorType::isDistributed() )
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
  MeshFunctionType aux = *auxPtr;
  InterfaceMapType interfaceMap = * interfaceMapPtr;
  aux.template synchronize< Communicator >(); //synchronization of intial conditions
  int i = MPI::GetRank( MPI::AllGroup ); //getting identification of MPI thread
#if ForDebug
        if( i == 2 ){
          aux.save("aux-init2.tnl");
          mesh->save("mesh-2.tnl");
        }
        if( i == 1 ){
          aux.save("aux-init1.tnl");
          mesh->save("mesh-1.tnl");
        }
        if( i == 3 ){
          aux.save("aux-init3.tnl");
          mesh->save("mesh-3.tnl");
        }
        if( i == 0 ){
          aux.save("aux-init0.tnl");
          mesh->save("mesh-0.tnl");
        }
#endif
  
  while( iteration < this->maxIterations )
  {    
#if  ForDebug 
    int WhileCount = 0; // number of passages of while cycle with condition calculated
    printf( "%d: meshDimensions are (x,y,z) = (%d,%d,%d).\n",i, mesh->getDimensions().x(), mesh->getDimensions().y(), mesh->getDimensions().z() );
    printf( "%d: owerlaps are ([x1,x2],[y1,y2],[z1,z2]) = ([%d,%d],[%d,%d],[%d,%d]).\n",i, vLower[0], vUpper[0], vLower[1], vUpper[1], vUpper[2], vLower[2] );
    /*if( std::is_same< DeviceType, Devices::Host >::value && i == 2 )
    {
      for( int j = mesh->getDimensions().y()-1; j>-1; j-- ){
        for( int m = 0; m < mesh->getDimensions().x(); m++ )
          printf( "%.2f " , aux[ j*mesh->getDimensions().x() + m ]);
        printf("\n");
      }
      printf("\n");
    }*/    
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
#endif
      if( std::is_same< DeviceType, Devices::Host >::value && calculate ) // should we calculate in Host?
      {
        calculate = 0;
        
/** HERE IS FSM FOR OPENMP (NO MPI) - isnt worthy */
        /*int numThreadsPerBlock = 64;
         
         
         int numBlocksX = mesh->getDimensions().x() / numThreadsPerBlock + (mesh->getDimensions().x() % numThreadsPerBlock != 0 ? 1:0);
         int numBlocksY = mesh->getDimensions().y() / numThreadsPerBlock + (mesh->getDimensions().y() % numThreadsPerBlock != 0 ? 1:0);
         int numBlocksZ = mesh->getDimensions().z() / numThreadsPerBlock + (mesh->getDimensions().z() % numThreadsPerBlock != 0 ? 1:0);
         //std::cout << "numBlocksX = " << numBlocksX << std::endl;
         
         //Real **sArray = new Real*[numBlocksX*numBlocksY];
         // for( int i = 0; i < numBlocksX * numBlocksY; i++ )
         // sArray[ i ] = new Real [ (numThreadsPerBlock + 2)*(numThreadsPerBlock + 2)];
         
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
         //for( int k = numBlocksX-1; k >-1; k-- ){
         //for( int l = 0; l < numBlocksY; l++ ){
         // std::cout<< BlockIterHost[ l*numBlocksX  + k ];
         // }
         // std::cout<<std::endl;
         // }
         // std::cout<<std::endl;
         unsigned int numWhile = 0;
         while( IsCalculationDone  )
         {      
         IsCalculationDone = 0;
         helpFunc1 = auxPtr;
         auxPtr = helpFunc;
         helpFunc = helpFunc1;
         this->template updateBlocks< 66 >( interfaceMap, *auxPtr, *helpFunc, BlockIterHost, numThreadsPerBlock );
         
         //Reduction      
         for( int i = 0; i < BlockIterHost.getSize(); i++ ){
         if( IsCalculationDone == 0 ){
         IsCalculationDone = IsCalculationDone || BlockIterHost[ i ];
         //break;
         }
         }
         numWhile++;
         
         
         this->getNeighbours( BlockIterHost, numBlocksX, numBlocksY, numBlocksZ );
         
         //string s( "aux-"+ std::to_string(numWhile) + ".tnl");
         //aux.save( s );
         }
         if( numWhile == 1 ){
         auxPtr = helpFunc;
         }
         aux = *auxPtr;*/
/**------------------------------------------------------------------------------*/
        
        
/** HERE IS FSM WITH MPI AND WITHOUT MPI */
        
#if ForDebug
        if( i == 1 ){
          aux.save("aux-final10.tnl");
        }
#endif
        for( cell.getCoordinates().z() = 0 + vLower[2];
                cell.getCoordinates().z() < mesh->getDimensions().z() - vUpper[2];
                cell.getCoordinates().z()++ )
        {
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
                //getting information weather we calculated in this passage
                calculated = this->updateCell( aux, cell ) || calculated;
              }
            }
          }
        }
#if ForDebug
        if( i == 1 ){
          aux.save("aux-final11.tnl");
        }
        int pocNull = 0;
        for( cell.getCoordinates().z() = 0 + vLower[2];
                cell.getCoordinates().z() < mesh->getDimensions().z() - vUpper[2];
                cell.getCoordinates().z()++ )
        {
          for( cell.getCoordinates().y() = 0 + vLower[1];
                  cell.getCoordinates().y() < mesh->getDimensions().y() - vUpper[1];
                  cell.getCoordinates().y()++ )
          {
            for( cell.getCoordinates().x() = 0 + vLower[0];
                    cell.getCoordinates().x() < mesh->getDimensions().x() - vUpper[0];
                    cell.getCoordinates().x()++ )
            {
              cell.refresh();
              if( fabs( aux(cell) ) < 0.002 )
                pocNull++;
            }
          }
        }
        printf("%d: 1. pocNull = %d\n", i , pocNull);
#endif        
        for( cell.getCoordinates().z() = 0 + vLower[2];
                cell.getCoordinates().z() < mesh->getDimensions().z() - vUpper[2];
                cell.getCoordinates().z()++ )
        {
          for( cell.getCoordinates().y() = 0 + vLower[1];
                  cell.getCoordinates().y() < mesh->getDimensions().y() - vUpper[1];
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
        }
#if ForDebug
        if( i == 1 ){
          aux.save("aux-final12.tnl");
        }
        pocNull = 0;
        for( cell.getCoordinates().z() = 0 + vLower[2];
                cell.getCoordinates().z() < mesh->getDimensions().z() - vUpper[2];
                cell.getCoordinates().z()++ )
        {
          for( cell.getCoordinates().y() = 0 + vLower[1];
                  cell.getCoordinates().y() < mesh->getDimensions().y() - vUpper[1];
                  cell.getCoordinates().y()++ )
          {
            for( cell.getCoordinates().x() = 0 + vLower[0];
                    cell.getCoordinates().x() < mesh->getDimensions().x() - vUpper[0];
                    cell.getCoordinates().x()++ )
            {
              cell.refresh();
              if( fabs( aux(cell) ) < 0.002 )
                pocNull++;
            }
          }
        }
        printf("%d: 2. pocNull = %d\n", i , pocNull);
#endif        
        //aux.save( "aux-2.tnl" );
        for( cell.getCoordinates().z() = 0 + vLower[2];
                cell.getCoordinates().z() < mesh->getDimensions().z() - vUpper[2];
                cell.getCoordinates().z()++ )
        {
          for( cell.getCoordinates().y() = mesh->getDimensions().y() - 1 - vUpper[1];
                  cell.getCoordinates().y() >= 0 + vLower[1];
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
        }
#if ForDebug
        if( i == 1 ){
          aux.save("aux-final13.tnl");
        }
        pocNull = 0;
        for( cell.getCoordinates().z() = 0 + vLower[2];
                cell.getCoordinates().z() < mesh->getDimensions().z() - vUpper[2];
                cell.getCoordinates().z()++ )
        {
          for( cell.getCoordinates().y() = 0 + vLower[1];
                  cell.getCoordinates().y() < mesh->getDimensions().y() - vUpper[1];
                  cell.getCoordinates().y()++ )
          {
            for( cell.getCoordinates().x() = 0 + vLower[0];
                    cell.getCoordinates().x() < mesh->getDimensions().x() - vUpper[0];
                    cell.getCoordinates().x()++ )
            {
              cell.refresh();
              if( fabs( aux(cell) ) < 0.002 )
                pocNull++;
            }
          }
        }
        printf("%d: 3. pocNull = %d\n", i , pocNull);
#endif        
        //aux.save( "aux-3.tnl" );
        
        for( cell.getCoordinates().z() = 0 + vLower[2];
                cell.getCoordinates().z() < mesh->getDimensions().z() - vUpper[2];
                cell.getCoordinates().z()++ )
        {
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
#if ForDebug
        if( i == 1 ){
          aux.save("aux-final14.tnl");
        }
        pocNull = 0;
        for( cell.getCoordinates().z() = 0 + vLower[2];
                cell.getCoordinates().z() < mesh->getDimensions().z() - vUpper[2];
                cell.getCoordinates().z()++ )
        {
          for( cell.getCoordinates().y() = 0 + vLower[1];
                  cell.getCoordinates().y() < mesh->getDimensions().y() - vUpper[1];
                  cell.getCoordinates().y()++ )
          {
            for( cell.getCoordinates().x() = 0 + vLower[0];
                    cell.getCoordinates().x() < mesh->getDimensions().x() - vUpper[0];
                    cell.getCoordinates().x()++ )
            {
              cell.refresh();
              if( fabs( aux(cell) ) < 0.002 )
                pocNull++;
            }
          }
        }
        printf("%d: 4. pocNull = %d\n", i , pocNull);
#endif        
        //aux.save( "aux-4.tnl" );
        
        for( cell.getCoordinates().z() = mesh->getDimensions().z() - 1 - vUpper[2];
                cell.getCoordinates().z() >= 0 + vLower[2];
                cell.getCoordinates().z()-- )
        {
          for( cell.getCoordinates().y() = 0 + vLower[1];
                  cell.getCoordinates().y() < mesh->getDimensions().y() - vUpper[1];
                  cell.getCoordinates().y()++ )
          {
            for( cell.getCoordinates().x() = 0 + vLower[0];
                    cell.getCoordinates().x() < mesh->getDimensions().x() - vUpper[0];
                    cell.getCoordinates().x()++ )
            {
              //std::cerr << "5 -> ";
              cell.refresh();
              if( ! interfaceMap( cell ) )
                this->updateCell( aux, cell );
            }
          }
        }
#if ForDebug
        if( i == 1 ){
          aux.save("aux-final15.tnl");
        }
        pocNull = 0;
        for( cell.getCoordinates().z() = 0 + vLower[2];
                cell.getCoordinates().z() < mesh->getDimensions().z() - vUpper[2];
                cell.getCoordinates().z()++ )
        {
          for( cell.getCoordinates().y() = 0 + vLower[1];
                  cell.getCoordinates().y() < mesh->getDimensions().y() - vUpper[1];
                  cell.getCoordinates().y()++ )
          {
            for( cell.getCoordinates().x() = 0 + vLower[0];
                    cell.getCoordinates().x() < mesh->getDimensions().x() - vUpper[0];
                    cell.getCoordinates().x()++ )
            {
              cell.refresh();
              if( fabs( aux(cell) ) < 0.002 )
                pocNull++;
            }
          }
        }
        printf("%d: 5. pocNull = %d\n", i , pocNull);
 #endif       
        //aux.save( "aux-5.tnl" );
        
        for( cell.getCoordinates().z() = mesh->getDimensions().z() - 1 - vUpper[2];
                cell.getCoordinates().z() >= 0 + vLower[2];
                cell.getCoordinates().z()-- )
        {
          for( cell.getCoordinates().y() = 0 + vLower[1];
                  cell.getCoordinates().y() < mesh->getDimensions().y() - vUpper[1];
                  cell.getCoordinates().y()++ )
          {
            for( cell.getCoordinates().x() = mesh->getDimensions().x() - 1 - vUpper[0];
                    cell.getCoordinates().x() >= 0 + vLower[0];
                    cell.getCoordinates().x()-- )		
            {
              //std::cerr << "6 -> ";
              cell.refresh();
              if( ! interfaceMap( cell ) )            
                this->updateCell( aux, cell );
            }
          }
        }
#if ForDebug
        if( i == 1 ){
          aux.save("aux-final16.tnl");
        }
        pocNull = 0;
        for( cell.getCoordinates().z() = 0 + vLower[2];
                cell.getCoordinates().z() < mesh->getDimensions().z() - vUpper[2];
                cell.getCoordinates().z()++ )
        {
          for( cell.getCoordinates().y() = 0 + vLower[1];
                  cell.getCoordinates().y() < mesh->getDimensions().y() - vUpper[1];
                  cell.getCoordinates().y()++ )
          {
            for( cell.getCoordinates().x() = 0 + vLower[0];
                    cell.getCoordinates().x() < mesh->getDimensions().x() - vUpper[0];
                    cell.getCoordinates().x()++ )
            {
              cell.refresh();
              if( fabs( aux(cell) ) < 0.002 )
                pocNull++;
            }
          }
        }
        printf("%d: 6. pocNull = %d\n", i , pocNull);
#endif        
        //aux.save( "aux-6.tnl" );
        
        for( cell.getCoordinates().z() = mesh->getDimensions().z() - 1 - vUpper[2];
                cell.getCoordinates().z() >= 0 + vLower[2];
                cell.getCoordinates().z()-- )
        {
          for( cell.getCoordinates().y() = mesh->getDimensions().y() - 1 - vUpper[1];
                  cell.getCoordinates().y() >= 0 + vLower[1];
                  cell.getCoordinates().y()-- )
          {
            for( cell.getCoordinates().x() = 0 + vLower[0];
                    cell.getCoordinates().x() < mesh->getDimensions().x() - vUpper[0];
                    cell.getCoordinates().x()++ )
            {
              //std::cerr << "7 -> ";
              cell.refresh();
              if( ! interfaceMap( cell ) )            
                this->updateCell( aux, cell );
            }
          }
        }
        
#if ForDebug
        if( i == 1 ){
          aux.save("aux-final17.tnl");
        }
        pocNull = 0;
        for( cell.getCoordinates().z() = 0 + vLower[2];
                cell.getCoordinates().z() < mesh->getDimensions().z() - vUpper[2];
                cell.getCoordinates().z()++ )
        {
          for( cell.getCoordinates().y() = 0 + vLower[1];
                  cell.getCoordinates().y() < mesh->getDimensions().y() - vUpper[1];
                  cell.getCoordinates().y()++ )
          {
            for( cell.getCoordinates().x() = 0 + vLower[0];
                    cell.getCoordinates().x() < mesh->getDimensions().x() - vUpper[0];
                    cell.getCoordinates().x()++ )
            {
              cell.refresh();
              if( fabs( aux(cell) ) < 0.002 )
                pocNull++;
            }
          }
        }
        printf("%d: 7. pocNull = %d\n", i , pocNull);
#endif        
        //aux.save( "aux-7.tnl" );
        
        for( cell.getCoordinates().z() = mesh->getDimensions().z() - 1 - vUpper[2];
                cell.getCoordinates().z() >= 0 + vLower[2];
                cell.getCoordinates().z()-- )
        {
          for( cell.getCoordinates().y() = mesh->getDimensions().y() - 1 - vUpper[1];
                  cell.getCoordinates().y() >= 0 + vLower[1];
                  cell.getCoordinates().y()-- )
          {
            for( cell.getCoordinates().x() = mesh->getDimensions().x() - 1 - vUpper[0];
                    cell.getCoordinates().x() >= 0 + vLower[0];
                    cell.getCoordinates().x()-- )		
            {
              //std::cerr << "8 -> ";
              cell.refresh();
              if( ! interfaceMap( cell ) )            
                this->updateCell( aux, cell );
            }
          }
        }
#if ForDebug
        if( i == 1 ){
          aux.save("aux-final18.tnl");
        }
        pocNull = 0;
        for( cell.getCoordinates().z() = 0 + vLower[2];
                cell.getCoordinates().z() < mesh->getDimensions().z() - vUpper[2];
                cell.getCoordinates().z()++ )
        {
          for( cell.getCoordinates().y() = 0 + vLower[1];
                  cell.getCoordinates().y() < mesh->getDimensions().y() - vUpper[1];
                  cell.getCoordinates().y()++ )
          {
            for( cell.getCoordinates().x() = 0 + vLower[0];
                    cell.getCoordinates().x() < mesh->getDimensions().x() - vUpper[0];
                    cell.getCoordinates().x()++ )
            {
              cell.refresh();
              if( fabs( aux(cell) ) < 0.002 )
                pocNull++;
            }
          }
        }
        printf("%d: 8. pocNull = %d\n", i , pocNull);
        for( cell.getCoordinates().z() = 0 + vLower[2];
                cell.getCoordinates().z() < mesh->getDimensions().z() - vUpper[2];
                cell.getCoordinates().z()++ )
        {
          for( cell.getCoordinates().y() = 0 + vLower[1];
                  cell.getCoordinates().y() < mesh->getDimensions().y() - vUpper[1];
                  cell.getCoordinates().y()++ )
          {
            for( cell.getCoordinates().x() = 0 + vLower[0];
                    cell.getCoordinates().x() < mesh->getDimensions().x() - vUpper[0];
                    cell.getCoordinates().x()++ )
            {
              cell.refresh();
              printf("%.2f ", aux(cell));
            }
            printf("\n");
          }
          printf("\n");
        }
#endif
        
        /**----------------------------------------------------------------------------------*/
      }
      if( std::is_same< DeviceType, Devices::Cuda >::value && calculate )
      {
#ifdef HAVE_CUDA
        // cudaBlockSize is a size of blocks. It's the number raised to the 3 power.
        // the number should be less than 10^3 (num of threads in one grid is maximally 1024)
        // IF YOU CHANGE THIS, YOU NEED TO CHANGE THE TEMPLATE PARAMETER IN CudaUpdateCellCaller (The Number + 2)
        const int cudaBlockSize( 8 );
        
        CudaUpdateCellCaller< 10 ><<< gridSize, blockSize >>>( ptr,
                interfaceMapPtr.template getData< Device >(),
                auxPtr.template getData< Device>(),
                helpFunc.template modifyData< Device>(),
                BlockIterDevice.getView() );
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
        
        GetNeighbours3D<<< nBlocksNeigh, 1024 >>>( BlockIterDevice.getView(), BlockIterPom.getView(), numBlocksX, numBlocksY, numBlocksZ );
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
        BlockIterDevice = BlockIterPom;
        
        CudaParallelReduc<<< nBlocks , 512 >>>( BlockIterDevice.getView(), dBlock.getView(), ( numBlocksX * numBlocksY * numBlocksZ ) );
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
        
        CudaParallelReduc<<< 1, nBlocks >>>( dBlock.getView(), dBlock.getView(), nBlocks );
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
        aux = *auxPtr;
        interfaceMap = *interfaceMapPtr;
#endif
      }
      
#ifdef HAVE_MPI
      if( CommunicatorType::isDistributed() ){
        
        const int *neigh = meshPom->getNeighbors(); // Getting nembers of distributed mesh
        MPI::Request *req;
        req = new MPI::Request[meshPom->getNeighborsCount()];  
        
        int neighCount = 0; // we know the number in runtime and it can differ for every MPI thread
        // Getting information weather some of six neghbours (top, bottom, right, left, ahead, behind) calculated
        int calculpom[6] = {0,0,0,0,0,0}; 
        
        
        if( neigh[0] != -1 ) // if you have west neighbour
        {
          // if we have this neighbour, we send calculated, one number, to him, ...
          req[neighCount] = MPI::ISend( &calculated, 1, neigh[0], 0, MPI::AllGroup );
          neighCount++;
          // and we recive the same information from him
          req[neighCount] = MPI::IRecv( &calculpom[0], 1, neigh[0], 0, MPI::AllGroup );
          neighCount++;
        }
        
        if( neigh[1] != -1 ) // east
        {
          req[neighCount] = MPI::ISend( &calculated, 1, neigh[1], 0, MPI::AllGroup ); 
          neighCount++;
          
          
          req[neighCount] = MPI::IRecv( &calculpom[1], 1, neigh[1], 0, MPI::AllGroup );
          neighCount++;
        }
        
        if( neigh[2] != -1 ) // north
        {
          req[neighCount] = MPI::ISend( &calculated, 1, neigh[2], 0, MPI::AllGroup );
          neighCount++;
          
          req[neighCount] = MPI::IRecv( &calculpom[2], 1, neigh[2], 0, MPI::AllGroup  );
          neighCount++;
        }
        
        if( neigh[5] != -1 ) //south
        {
          req[neighCount] = MPI::ISend( &calculated, 1, neigh[5], 0, MPI::AllGroup );
          neighCount++;
          
          req[neighCount] = MPI::IRecv( &calculpom[3], 1, neigh[5], 0, MPI::AllGroup );
          neighCount++;
        }
        
        if( neigh[8] != -1 ) // top 
        {
          req[neighCount] = MPI::ISend( &calculated, 1, neigh[8], 0, MPI::AllGroup );
          neighCount++;
          
          req[neighCount] = MPI::IRecv( &calculpom[4], 1, neigh[8], 0, MPI::AllGroup );
          neighCount++;
        }
        
        if( neigh[17] != -1 ) //bottom
        {
          req[neighCount] = MPI::ISend( &calculated, 1, neigh[17], 0, MPI::AllGroup );
          neighCount++;
          
          req[neighCount] = MPI::IRecv( &calculpom[5], 1, neigh[17], 0, MPI::AllGroup );
          neighCount++;
        }
        
        MPI::WaitAll(req,neighCount); //waiting for all to have all the information
#if ForDebug
        printf( "%d: Sending Calculated = %d.\n", i, calculated );
        printf( "%d: calculpom[0] = %d, calculpom[1] = %d, calculpom[2] = %d, calculpom[3] = %d, calculpom[4] = %d,"
                "calculpom[5] = %d", i ,calculpom[0],calculpom[1],calculpom[2],calculpom[3],calculpom[4],calculpom[5]);
#endif        
        // if one of the MPI thread had calculated = 1, then all get 1. Otherwise all get 0
        MPI::Allreduce( &calculated, &calculated, 1, MPI_LOR,  MPI::AllGroup ); 
        // synchronizate the overlaps 
        aux.template synchronize< Communicator >();
        // if any of my neighbours had calculated = 1, than I should calculate again (but all of us has to go throw while(calculated))
        calculate = calculpom[0] || calculpom[1] || calculpom[2] ||
                    calculpom[3] || calculpom[4] || calculpom[5];
#if ForDebug
        printf( "%d: Receved Calculated = %d.\n%d: Calculate = %d\n", i, calculated, i, calculate);
#endif
        
#if ForDebug 
        if( i == 1 )
          printf("WhileCount = %d\n",WhileCount);
        if( i == 2 ){
          aux.save("aux-final2.tnl");
          mesh->save("mesh-2.tnl");
        }
        if( i == 1 ){
          aux.save("aux-final1.tnl");
          mesh->save("mesh-1.tnl");
        }
        if( i == 3 ){
          aux.save("aux-final3.tnl");
          mesh->save("mesh-3.tnl");
        }
        if( i == 0 ){
          aux.save("aux-final0.tnl");
          mesh->save("mesh-0.tnl");
        }
        //calculated = 0; // DEBUG;
#endif
      }
#endif
      if( !CommunicatorType::isDistributed() ) // If we start the solver without MPI, we need calculated 0!
        calculated = 0; //otherwise we would go throw the FSM code and CUDA FSM code again uselessly
    }
    //aux.save( "aux-8.tnl" );
    iteration++;
    
  }
  // Saving the results into Aux for MakeSnapshot function.
  Aux = auxPtr; 
  aux.save("aux-final.tnl");
}

#ifdef HAVE_CUDA
// DeepCopy nebo pracne kopirovat kraje v zavislosti na vLower,vUpper z sArray do helpFunc.
template< typename Real, typename Device, typename Index >
__global__ void DeepCopy( const Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index > >& aux,
        Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index > >& helpFunc, int copy, int k )
{
  int thri = threadIdx.x + blockDim.x*blockIdx.x;
  int thrj = blockDim.y*blockIdx.y + threadIdx.y;
  int thrk = blockDim.z*blockIdx.z + threadIdx.z;
  
  const Meshes::Grid< 3, Real, Device, Index >& mesh = aux.template getMesh< Devices::Cuda >();
  if( copy ){
    if( thri < mesh.getDimensions().x() && thrj < mesh.getDimensions().y() && thrk < mesh.getDimensions().z() )
    {
      helpFunc[ thrk * mesh.getDimensions().x() * mesh.getDimensions().y() + thrj * mesh.getDimensions().x() + thri ] =
              aux[ thrk * mesh.getDimensions().x() * mesh.getDimensions().y() + thrj * mesh.getDimensions().x() + thri ];
    }
  }
  else // for debug, values can be printed only from cuda kernel
  {
    if( thrk == 0 && thri==0 && thrj == 0 && blockIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && k == 0 )
    {
      printf("%d: DimX = %d, DimY = %d, DimZ = %d\n", k,mesh.getDimensions().x(),mesh.getDimensions().y(),mesh.getDimensions().z() );
      for( int z = mesh.getDimensions().z()-1; z > mesh.getDimensions().z()-2; z-- )
      {
        for( int y = 0; y < mesh.getDimensions().y(); y++ )
        {
          for( int x = 0; x < mesh.getDimensions().x(); x++ )
          {
            printf("%.2f ", helpFunc[ z *mesh.getDimensions().y()*mesh.getDimensions().x() + y*mesh.getDimensions().x() + x ]);
          }
          printf("\n");
        }
        printf("\n");
      }
      printf("\n");
    }
    if( thrk == 0 && thri==0 && thrj == 0 && blockIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && k == 1 )
    {
      printf("%d: DimX = %d, DimY = %d, DimZ = %d\n", k,mesh.getDimensions().x(),mesh.getDimensions().y(),mesh.getDimensions().z() );
      
      if( k == 1 )
      {
        for( int z = 1; z < 2; z++ )
        {
          for( int y = 0; y < mesh.getDimensions().y(); y++ )
          {
            for( int x = 0; x < mesh.getDimensions().x(); x++ )
            {
              printf("%.2f ", aux[ z *mesh.getDimensions().y()*mesh.getDimensions().x() + y*mesh.getDimensions().x() + x ]);
            }
            printf("\n");
          }
          printf("\n");
        }
        printf("\n");
      }
    }
  }
}

template < typename Index >
__global__ void GetNeighbours3D( TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterDevice,
        TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterPom,
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
    if( m > 0 && BlockIterDevice[ i - 1 ] ){ // left neighbour
      pom = 1;//BlockIterPom[ i ] = 1;
    }else if( m < numBlockX -1 && BlockIterDevice[ i + 1 ] ){ // right neighbour
      pom = 1;//BlockIterPom[ i ] = 1;
    }else if( k > 0 && BlockIterDevice[ i - numBlockX ] ){ // bottom neighbour
      pom = 1;// BlockIterPom[ i ] = 1;
    }else if( k < numBlockY -1 && BlockIterDevice[ i + numBlockX ] ){ // top neighbour
      pom = 1;//BlockIterPom[ i ] = 1;
    }else if( l > 0 && BlockIterDevice[ i - numBlockX*numBlockY ] ){ // neighbour behind 
      pom = 1;
    }else if( l < numBlockZ-1 && BlockIterDevice[ i + numBlockX*numBlockY ] ){ // neighbour in front
      pom = 1;
    }
    
    if( !BlockIterDevice[ i ] ) // only in CudaUpdateCellCaller can BlockIterDevice gain 0
      BlockIterPom[ i ] = pom;
    else
      BlockIterPom[ i ] = 1;
  }
}

template < int sizeSArray, typename Real, typename Device, typename Index >
__global__ void CudaUpdateCellCaller( tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > > ptr,
        const Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index >, 3, bool >& interfaceMap,
        const Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index > >& aux,
        Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index > >& helpFunc,
        TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterDevice )
{
  int thri = threadIdx.x; int thrj = threadIdx.y; int thrk = threadIdx.z;
  int blIdx = blockIdx.x; int blIdy = blockIdx.y; int blIdz = blockIdx.z;
  int i = threadIdx.x + blockDim.x*blockIdx.x + vLower[0]; // WITH OVERLAPS!!! i,j,k aren't coordinates of all values
  int j = blockDim.y*blockIdx.y + threadIdx.y + vLower[1];
  int k = blockDim.z*blockIdx.z + threadIdx.z + vLower[2];
  int currentIndex = thrk * blockDim.x * blockDim.y + thrj * blockDim.x + thri;
  const Meshes::Grid< 3, Real, Device, Index >& mesh = interfaceMap.template getMesh< Devices::Cuda >();
  
  // should this block calculate?
  if( BlockIterDevice[ blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x ] ) 
  {
    __syncthreads();
    
    // Array indicates weather some threads calculated (for parallel reduction)
    __shared__ volatile bool changed[ (sizeSArray - 2)*(sizeSArray - 2)*(sizeSArray - 2) ];
    changed[ currentIndex ] = false;
    
    if( thrj == 0 && thri == 0 && thrk == 0 )
      changed[ 0 ] = true; // first indicates weather we should calculate again (princip of parallel reduction)
    
    __shared__ Real hx; __shared__ int dimX; //getting stepps and size of mesh
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
      // we dont know if we will calculate in here, more info down in code
      BlockIterDevice[ blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x ] = 0;
    }
    
    // sArray contains values of one block (coppied from aux) and edges (not MPI) of those blocks
    __shared__ volatile Real sArray[ sizeSArray * sizeSArray * sizeSArray ];
    sArray[(thrk+1)* sizeSArray * sizeSArray + (thrj+1) *sizeSArray + thri+1] = std::numeric_limits< Real >::max();
    
    // getting some usefull information 
    int numOfBlockx;
    int numOfBlocky;
    int numOfBlockz;
    int xkolik; // maximum of threads in x direction (for all blocks different)
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
      xkolik = (dimX-vUpper[0]-vLower[0]) - (blIdx)*blockDim.x+1;
    if( numOfBlocky -1 == blIdy )
      ykolik = (dimY-vUpper[1]-vLower[1]) - (blIdy)*blockDim.y+1;
    if( numOfBlockz-1 == blIdz )
      zkolik = (dimZ-vUpper[2]-vLower[2]) - (blIdz)*blockDim.z+1;
    __syncthreads();
    
     //filling sArray edges
    if( thri == 0 ) //x bottom
    {        
      if( (blIdx != 0 || vLower[0] !=0) && thrj+1 < ykolik && thrk+1 < zkolik )
        sArray[ (thrk+1 )* sizeSArray * sizeSArray + (thrj+1)*sizeSArray + 0 ] = 
                aux[ (blIdz*blockDim.z + vLower[2]) * dimX * dimY + (blIdy * blockDim.y+vLower[1])*dimX 
                + blIdx*blockDim.x + thrj * dimX -1 + thrk*dimX*dimY + vLower[0] ];
    else
        sArray[(thrk+1)* sizeSArray * sizeSArray + (thrj+1)*sizeSArray + 0] = std::numeric_limits< Real >::max();
    }
    
    if( thri == 1 ) //xtop
    {
      if( dimX - vLower[ 0 ] > (blIdx+1) * blockDim.x && thrj+1 < ykolik && thrk+1 < zkolik )
        sArray[ (thrk+1) * sizeSArray * sizeSArray + (thrj+1) *sizeSArray + xkolik ] =
                aux[ (blIdz*blockDim.z + vLower[2]) * dimX * dimY + (blIdy * blockDim.y+vLower[1])*dimX
                + blIdx*blockDim.x + blockDim.x + thrj * dimX + thrk*dimX*dimY + vLower[0] ];
     else
        sArray[ (thrk+1) * sizeSArray * sizeSArray + (thrj+1)*sizeSArray + xkolik] = std::numeric_limits< Real >::max();
    }
    if( thri == 2 ) //y bottom
    {        
      if( (blIdy != 0 || vLower[1] !=0) && thrj+1 < xkolik && thrk+1 < zkolik )
        sArray[ (thrk+1) * sizeSArray * sizeSArray +0*sizeSArray + thrj+1] =
                aux[ (blIdz*blockDim.z + vLower[2]) * dimX * dimY + (blIdy * blockDim.y+vLower[1])*dimX
                + blIdx*blockDim.x - dimX + thrj + thrk*dimX*dimY + vLower[0] ];
      else
        sArray[ (thrk+1) * sizeSArray * sizeSArray + 0*sizeSArray + thrj+1] = std::numeric_limits< Real >::max();
    }
    
    if( thri == 3 ) //y top
    {
      if( dimY - vLower[ 1 ] > (blIdy+1) * blockDim.y && thrj+1 < xkolik && thrk+1 < zkolik )
        sArray[ (thrk+1) * sizeSArray * sizeSArray + ykolik*sizeSArray + thrj+1] =
                aux[ (blIdz*blockDim.z + vLower[2]) * dimX * dimY + ((blIdy+1) * blockDim.y+vLower[1])*dimX
                + blIdx*blockDim.x + thrj + thrk*dimX*dimY + vLower[0] ];
     else
        sArray[ (thrk+1) * sizeSArray * sizeSArray + ykolik*sizeSArray + thrj+1] = std::numeric_limits< Real >::max();
    }
    if( thri == 4 ) //z bottom
    {        
      if( (blIdz != 0 || vLower[2] !=0) && thrj+1 < ykolik && thrk+1 < xkolik )
        sArray[ 0 * sizeSArray * sizeSArray +(thrj+1 )* sizeSArray + thrk+1] =
                aux[ (blIdz*blockDim.z + vLower[2]) * dimX * dimY + (blIdy * blockDim.y+vLower[1])*dimX
                + blIdx*blockDim.x - dimX * dimY + thrj * dimX + thrk + vLower[0] ];
     else
        sArray[0 * sizeSArray * sizeSArray + (thrj+1) *sizeSArray + thrk+1] = std::numeric_limits< Real >::max();
    }
    
    if( thri == 5 ) //z top
    {
      if( dimZ - vLower[ 2 ] > (blIdz+1) * blockDim.z && thrj+1 < ykolik && thrk+1 < xkolik )
        sArray[ zkolik * sizeSArray * sizeSArray + (thrj+1) * sizeSArray + thrk+1] =
                aux[ ((blIdz+1)*blockDim.z + vLower[2]) * dimX * dimY + (blIdy * blockDim.y+vLower[1])*dimX
                + blIdx*blockDim.x + thrj * dimX + thrk + vLower[0] ];
     else
        sArray[zkolik * sizeSArray * sizeSArray + (thrj+1) * sizeSArray + thrk+1] = std::numeric_limits< Real >::max();
    }
    
    // Copy all other values that aren't edges
    if( i - vLower[0] < dimX && j - vLower[1] < dimY && k - vLower[2] < dimZ &&
        thri+1 < xkolik + vUpper[0] && thrj+1 < ykolik + vUpper[1] && thrk+1 < zkolik + vUpper[2] )
    {
      sArray[(thrk+1) * sizeSArray * sizeSArray + (thrj+1) *sizeSArray + thri+1] = aux[ k*dimX*dimY + j*dimX + i ];
    }
    __syncthreads(); 
    
#if ForDebug    
    /*if( thri==0 && thrj == 0 && thrk == 0 && blockIdx.z == 0 && blockIdx.x == 2 && blockIdx.y == 2 && MPIthread == 1 )
    {
      printf( "všechno před výpočtem: \n");
      for( int m = sizeSArray-1; m>-1; m-- ){
        for( int l = 0; l < sizeSArray; l++ )
          printf( "%.2f ", sArray[4*sizeSArray * sizeSArray + m * sizeSArray + l]);
        printf( "\n");
      }
      printf( "\n");
    }
    
    if(thri==0 && thrj == 0 && thrk == 0 && blockIdx.z == 0 && blockIdx.x == 2 && blockIdx.y == 2 && MPIthread == 1 )
    {
      for( int m = 24; m>14; m-- ){
        for( int l = 15; l < 25; l++ )  
          printf("%.2f ", aux[ 4 *mesh.getDimensions().y()*mesh.getDimensions().x() + m*mesh.getDimensions().x() + l ]);
        printf( "\n");
      }
      printf( "\n");
    }*/
#endif 
    
    //main while cycle. each value can get information only from neighbour but that information has to spread there
    while( changed[ 0 ] )
    {
      __syncthreads();
      
      changed[ currentIndex ] = false;
      
      //calculation of update cell
      if( i < dimX - vUpper[0] && j < dimY - vUpper[1] && k < dimZ - vUpper[2] )
      {
        if( ! interfaceMap[ k*dimX*dimY + j * dimX + i ] )
        {
          // calculate new value depending on neighbours in sArray on (thri+1, thrj+1) coordinates
          changed[ currentIndex ] = ptr.updateCell3D< sizeSArray >( sArray, thri+1, thrj+1, thrk+1, hx,hy,hz); 
        }
      }
      __syncthreads();
      
      //pyramid reduction (parallel reduction)
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
      
      // if we calculated, then the BlockIterDevice should contain the info about this whole block! (only one number for one block)
      if( changed[ 0 ] && thri == 0 && thrj == 0 && thrk == 0 )
      {
        BlockIterDevice[ blIdz * gridDim.x * gridDim.y + blIdy * gridDim.x + blIdx ] = 1;
      }
      __syncthreads();
    }
    
    // copy results into helpFunc (not into aux bcs of conflicts)
    if( i < dimX && j < dimY && k < dimZ && thri+1 < xkolik && thrj+1 < ykolik && thrk+1 < zkolik )
      helpFunc[ k*dimX*dimY + j * dimX + i ] = sArray[ (thrk+1) * sizeSArray * sizeSArray + (thrj+1) * sizeSArray + thri+1 ];
    
  }
  else // if not, then it should at least copy the values from aux to helpFunc.
  {
    if( i < mesh.getDimensions().x() - vUpper[0] && j < mesh.getDimensions().y() - vUpper[1]
            && k < mesh.getDimensions().z() - vUpper[2])
      helpFunc[ k * mesh.getDimensions().x() * mesh.getDimensions().y() + j * mesh.getDimensions().x() + i ] =
              aux[ k * mesh.getDimensions().x() * mesh.getDimensions().y() + j * mesh.getDimensions().x() + i ];
  }
}  
#endif
