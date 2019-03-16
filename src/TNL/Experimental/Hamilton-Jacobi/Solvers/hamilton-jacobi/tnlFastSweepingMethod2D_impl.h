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
  
  // Setting overlaps ( WITHOUT MPI SHOULD BE 0 )
  StaticVector vecLowerOverlaps, vecUpperOverlaps;
  setOverlaps( vecLowerOverlaps, vecUpperOverlaps, mesh );
  
  std::cout << "Initiating the interface cells ..." << std::endl;
  BaseType::initInterface( u, auxPtr, interfaceMapPtr, vecLowerOverlaps, vecUpperOverlaps );
  
  //auxPtr->save( "aux-ini.tnl" );
  
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
    // calculatedBefore indicates weather we calculated in the last passage of the while cycle 
    // calculatedBefore is same for all ranks 
    // without MPI should be FALSE at the end of while cycle body
    int calculatedBefore = 1;
    
    // calculateMPIAgain indicates if the thread should calculate again in upcoming passage of while cycle
    // calculateMPIAgain is a value that can differ in every rank
    // without MPI should be FALSE at the end of while cycle body
    int calculateMPIAgain = 1;  
    
    while( calculatedBefore )
    {
      calculatedBefore = 0;
      
      if( std::is_same< DeviceType, Devices::Host >::value && calculateMPIAgain ) // should we calculate in Host?
      {
        calculateMPIAgain = 0;
        
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
        
        
  // FSM FOR MPI and WITHOUT MPI
        StaticVector boundsFrom; StaticVector boundsTo;
    // UP and RIGHT
        boundsFrom[1] = vecLowerOverlaps[1]; boundsTo[1] = mesh->getDimensions().y() - vecUpperOverlaps[1];
        boundsFrom[0] = vecLowerOverlaps[0]; boundsTo[0] = mesh->getDimensions().x() - vecUpperOverlaps[0];
        calculatedBefore = goThroughSweep( boundsFrom, boundsTo, aux, interfaceMap, anisotropy );
        //aux.save("aux-1.tnl");
        
    // UP and LEFL
        boundsFrom[1] = vecLowerOverlaps[1]; boundsTo[1] = mesh->getDimensions().y() - vecUpperOverlaps[1];
        boundsFrom[0] = mesh->getDimensions().x() - 1 - vecUpperOverlaps[0]; boundsTo[0] = -1 + vecLowerOverlaps[0];
        goThroughSweep( boundsFrom, boundsTo, aux, interfaceMap, anisotropy );
        //aux.save( "aux-2.tnl" );
        
    // DOWN and RIGHT
        boundsFrom[1] = mesh->getDimensions().y() - 1 - vecUpperOverlaps[1]; boundsTo[1] = - 1 + vecLowerOverlaps[1];
        boundsFrom[0] = vecLowerOverlaps[0]; boundsTo[0] = mesh->getDimensions().x() - vecUpperOverlaps[0];
        goThroughSweep( boundsFrom, boundsTo, aux, interfaceMap, anisotropy );
        //aux.save( "aux-3.tnl" );
        
    // DOWN and LEFT
        boundsFrom[1] = mesh->getDimensions().y() - 1 - vecUpperOverlaps[1]; boundsTo[1] = - 1 + vecLowerOverlaps[1];
        boundsFrom[0] = mesh->getDimensions().x() - 1 - vecUpperOverlaps[0]; boundsTo[0] = - 1 + vecLowerOverlaps[0];
        goThroughSweep( boundsFrom, boundsTo, aux, interfaceMap, anisotropy );
        
      }
      if( std::is_same< DeviceType, Devices::Cuda >::value && calculateMPIAgain ) // should we calculate on CUDA?
      {
        calculateMPIAgain = 0;
          
#ifdef HAVE_CUDA
        TNL_CHECK_CUDA_DEVICE;
        // Maximum cudaBlockSite is 32. Because of maximum num. of threads in kernel.
        // IF YOU CHANGE THIS, YOU NEED TO CHANGE THE TEMPLATE PARAMETER IN CudaUpdateCellCaller (The Number + 2)
        const int cudaBlockSize( 16 );
        
        // Setting number of threads and blocks for kernel
        int numBlocksX = Devices::Cuda::getNumberOfBlocks( mesh->getDimensions().x() - vecLowerOverlaps[0] - vecUpperOverlaps[0], cudaBlockSize );
        int numBlocksY = Devices::Cuda::getNumberOfBlocks( mesh->getDimensions().y() - vecLowerOverlaps[1] - vecUpperOverlaps[1], cudaBlockSize );
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
        // All blocks should calculate in first passage ( setValue(1) )
        TNL::Containers::Array< int, Devices::Cuda, IndexType > blockCalculationIndicator( numBlocksX * numBlocksY );
        blockCalculationIndicator.setValue( 1 );
        TNL_CHECK_CUDA_DEVICE;
        
        // Array into which we identify the neighbours and then copy it into blockCalculationIndicator
        TNL::Containers::Array< int, Devices::Cuda, IndexType > blockCalculationIndicatorHelp(numBlocksX * numBlocksY );
        blockCalculationIndicatorHelp.setValue( 0 );
        
        // number of Blocks for kernel that calculates neighbours.
        int nBlocksNeigh = ( numBlocksX * numBlocksY )/1024 + ((( numBlocksX * numBlocksY )%1024 != 0) ? 1:0);
        
        // Helping meshFunction that switches with AuxPtr in every calculation of CudaUpdateCellCaller<<<>>>()
        MeshFunctionPointer helpFunc( mesh );
        helpFunc.template modifyData() = auxPtr.template getData(); 
        
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
        while( calculateCudaBlocksAgain )
        {
  /** HERE IS CHESS METHOD (NO MPI) **/
          
          /*
           CudaUpdateCellCaller<18><<< gridSize, blockSize >>>( ptr,
           interfaceMapPtr.template getData< Device >(),
           auxPtr.template getData< Device>(),
           helpFunc.template modifyData< Device>(),
           blockCalculationIndicator,
           oddEvenBlock );
           cudaDeviceSynchronize();
           TNL_CHECK_CUDA_DEVICE;
           
           oddEvenBlock= (oddEvenBlock == 0) ? 1: 0;
           
           CudaUpdateCellCaller<18><<< gridSize, blockSize >>>( ptr,
           interfaceMapPtr.template getData< Device >(),
           helpFunc.template getData< Device>(),
           auxPtr.template modifyData< Device>(),
           blockCalculationIndicator, vecLowerOverlaps, vecUpperOverlaps, 
           oddEvenBlock );
           cudaDeviceSynchronize();
           TNL_CHECK_CUDA_DEVICE;
           
           oddEvenBlock= (oddEvenBlock == 0) ? 1: 0;
           
           CudaParallelReduc<<< nBlocks , 1024 >>>( blockCalculationIndicator, dBlock, ( numBlocksX * numBlocksY ) );
           cudaDeviceSynchronize();
           TNL_CHECK_CUDA_DEVICE;
           CudaParallelReduc<<< 1, nBlocks >>>( dBlock, dBlock, nBlocks );
           cudaDeviceSynchronize();
           TNL_CHECK_CUDA_DEVICE;
           
           BlockIterD = dBlock.getElement( 0 );*/
          
  /**------------------------------------------------------------------------------------------------*/
          
          
  /** HERE IS FIM FOR MPI AND WITHOUT MPI **/
          Devices::Cuda::synchronizeDevice();
          CudaUpdateCellCaller<18><<< gridSize, blockSize >>>( ptr, interfaceMapPtr.template getData< Device >(),
                  auxPtr.template getData< Device>(), helpFunc.template modifyData< Device>(),
                  blockCalculationIndicator, vecLowerOverlaps, vecUpperOverlaps );
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
          GetNeighbours<<< nBlocksNeigh, 1024 >>>( blockCalculationIndicator, blockCalculationIndicatorHelp, numBlocksX, numBlocksY );
          cudaDeviceSynchronize();
          TNL_CHECK_CUDA_DEVICE;
          blockCalculationIndicator = blockCalculationIndicatorHelp;
          cudaDeviceSynchronize();
          TNL_CHECK_CUDA_DEVICE;
          
          // "Parallel reduction" to see if we should calculate again BlockIterD
          calculateCudaBlocksAgain = blockCalculationIndicator.containsValue(1);
          
          // When we change something then we should caclucate again in the next passage of MPI ( calculated = true )
         if( calculateCudaBlocksAgain ){
            calculatedBefore = 1;
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
      }
#endif
      if( !CommunicatorType::isDistributed() ) // If we start the solver without MPI, we need calculated 0!
        calculatedBefore = 0;
    }
    iteration++;
  }
  Aux=auxPtr; // copy it for MakeSnapshot
}


// PROTECTED FUNCTIONS:

template< typename Real, typename Device, typename Index, 
          typename Communicator, typename Anisotropy >
void 
FastSweepingMethod< Meshes::Grid< 2, Real, Device, Index >, Communicator, Anisotropy >::
setOverlaps( StaticVector& vecLowerOverlaps, StaticVector& vecUpperOverlaps,
              const MeshPointer& mesh)
{
  vecLowerOverlaps[0] = 0; vecLowerOverlaps[1] = 0; vecUpperOverlaps[0] = 0; vecUpperOverlaps[1] = 0;
#ifdef HAVE_MPI
  if( CommunicatorType::isDistributed() ) //If we started solver with MPI
  {
    //Distributed mesh for MPI overlaps (without MPI null pointer)
    const Meshes::DistributedMeshes::DistributedMesh< MeshType >* meshPom = mesh->getDistributedMesh();
    vecLowerOverlaps = meshPom->getLowerOverlap();
    vecUpperOverlaps = meshPom->getUpperOverlap();
  }
#endif
}

template < typename Index >
__global__ void GetNeighbours( TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterDevice,
        TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterPom, int numBlockX, int numBlockY )
{
  bool calculated = false;
  const MeshType& mesh = aux.getMesh();
  const IndexType stepX = boundsFrom[0] < boundsTo[0]? 1 : -1;
  const IndexType stepY = boundsFrom[1] < boundsTo[1]? 1 : -1;
  
  typename MeshType::Cell cell( mesh );
  cell.refresh();
  
  for( cell.getCoordinates().y() = boundsFrom[1];
          TNL::abs( cell.getCoordinates().y() - boundsTo[1] ) > 0;
          cell.getCoordinates().y() += stepY )
  {
    for( cell.getCoordinates().x() = boundsFrom[0];
           TNL::abs( cell.getCoordinates().x() - boundsTo[0] ) > 0;
            cell.getCoordinates().x() += stepX )
    {
      cell.refresh();
      if( ! interfaceMap( cell ) )
      {
        calculated = this->updateCell( aux, cell ) || calculated;
      }
    }
  }
  return calculated;
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
  Meshes::DistributedMeshes::DistributedMesh< MeshType >* meshDistr = mesh->getDistributedMesh();
  
  int calculateFromNeighbours[4] = {0,0,0,0};
  const int *neighbours = meshDistr->getNeighbors(); // Getting neighbors of distributed mesh
  MPI::Request *requestsInformation;
  requestsInformation = new MPI::Request[ meshDistr->getNeighborsCount() ];  
  
  int neighCount = 0; // should this thread calculate again?
  
  if( neighbours[0] != -1 ) // LEFT
  {
    requestsInformation[neighCount++] =
            MPI::ISend( &calculatedBefore, 1, neighbours[0], 0, MPI::AllGroup );
    requestsInformation[neighCount++] = 
            MPI::IRecv( &calculateFromNeighbours[0], 1, neighbours[0], 0, MPI::AllGroup );
  }
  
  if( neighbours[1] != -1 ) // RIGHT
  {
    requestsInformation[neighCount++] =
            MPI::ISend( &calculatedBefore, 1, neighbours[1], 0, MPI::AllGroup ); 
    requestsInformation[neighCount++] = 
            MPI::IRecv( &calculateFromNeighbours[1], 1, neighbours[1], 0, MPI::AllGroup );
  }
  
  if( neighbours[2] != -1 ) //UP
  {
    requestsInformation[neighCount++] = 
            MPI::ISend( &calculatedBefore, 1, neighbours[2], 0, MPI::AllGroup );
    requestsInformation[neighCount++] =
            MPI::IRecv( &calculateFromNeighbours[2], 1, neighbours[2], 0, MPI::AllGroup  );
  }
  
  if( neighbours[5] != -1 ) //DOWN
  {
    requestsInformation[neighCount++] = 
            MPI::ISend( &calculatedBefore, 1, neighbours[5], 0, MPI::AllGroup );
    requestsInformation[neighCount++] = 
            MPI::IRecv( &calculateFromNeighbours[3], 1, neighbours[5], 0, MPI::AllGroup );
  }
  MPI::WaitAll( requestsInformation, neighCount );
  
  MPI::Allreduce( &calculatedBefore, &calculatedBefore, 1, MPI_LOR,  MPI::AllGroup );
  calculateMPIAgain = calculateFromNeighbours[0] || calculateFromNeighbours[1] ||
              calculateFromNeighbours[2] || calculateFromNeighbours[3];
}
#endif
        TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterDevice, int oddEvenBlock )
