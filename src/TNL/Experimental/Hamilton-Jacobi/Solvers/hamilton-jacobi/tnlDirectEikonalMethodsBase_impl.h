/* 
 * File:   tnlDirectEikonalMethodsBase_impl.h
 * Author: oberhuber
 *
 * Created on July 14, 2016, 3:22 PM
 */

#pragma once

#include <limits>

#include <iostream>
#include "tnlFastSweepingMethod.h"
#include "tnlDirectEikonalMethodsBase.h"

template< typename Real,
        typename Device,
        typename Index >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > >::
initInterface( const MeshFunctionPointer& _input,
        MeshFunctionPointer& _output,
        InterfaceMapPointer& _interfaceMap  )
{
  if( std::is_same< Device, Devices::Cuda >::value )
  {
#ifdef HAVE_CUDA
    const MeshType& mesh = _input->getMesh();
    
    const int cudaBlockSize( 16 );
    int numBlocksX = Devices::Cuda::getNumberOfBlocks( mesh.getDimensions().x(), cudaBlockSize );
    dim3 blockSize( cudaBlockSize );
    dim3 gridSize( numBlocksX );
    Devices::Cuda::synchronizeDevice();
    CudaInitCaller<<< gridSize, blockSize >>>( _input.template getData< Device >(),
            _output.template modifyData< Device >(),
            _interfaceMap.template modifyData< Device >() );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
#endif
  }
  if( std::is_same< Device, Devices::Host >::value )
  {
    const MeshType& mesh = _input->getMesh();
    typedef typename MeshType::Cell Cell;
    const MeshFunctionType& input = _input.getData();
    MeshFunctionType& output = _output.modifyData();
    InterfaceMapType& interfaceMap = _interfaceMap.modifyData();
    Cell cell( mesh );
    for( cell.getCoordinates().x() = 0;
            cell.getCoordinates().x() < mesh.getDimensions().x();
            cell.getCoordinates().x() ++ )
    {
      cell.refresh();
      output[ cell.getIndex() ] =
              input( cell ) >= 0 ? std::numeric_limits< RealType >::max() :
                -std::numeric_limits< RealType >::max();
      interfaceMap[ cell.getIndex() ] = false;
    }
    
    
    const RealType& h = mesh.getSpaceSteps().x();
    for( cell.getCoordinates().x() = 0;
            cell.getCoordinates().x() < mesh.getDimensions().x() - 1;
            cell.getCoordinates().x() ++ )
    {
      cell.refresh();
      const RealType& c = input( cell );      
      if( ! cell.isBoundaryEntity()  )
      {
        const auto& neighbors = cell.getNeighborEntities();
        Real pom = 0;
        //const IndexType& c = cell.getIndex();
        const IndexType e = neighbors.template getEntityIndex<  1 >();
        if( c * input[ e ] <= 0 )
        {
          pom = TNL::sign( c )*( h * c )/( c - input[ e ]);
          if( TNL::abs( output[ cell.getIndex() ] ) > TNL::abs( pom ) )
            output[ cell.getIndex() ] = pom;
          
          pom = pom - TNL::sign( c )*h; //output[ e ] = (hx * c)/( c - input[ e ]) - hx;
          if( TNL::abs( output[ e ] ) > TNL::abs( pom ) )
            output[ e ] = pom; 
          
          interfaceMap[ cell.getIndex() ] = true;
          interfaceMap[ e ] = true;
        }
      }
    }
  }
}

template< typename Real,
        typename Device,
        typename Index >
template< int sizeSArray >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
updateBlocks( InterfaceMapType interfaceMap,
        MeshFunctionType aux,
        MeshFunctionType helpFunc,
        ArrayContainer BlockIterHost, int numThreadsPerBlock/*, Real **sArray*/ )
{
#pragma omp parallel for schedule( dynamic )
  for( IndexType i = 0; i < BlockIterHost.getSize(); i++ )
  {
    if( BlockIterHost[ i ] )
    {
      MeshType mesh = interfaceMap.template getMesh< Devices::Host >();
      
      int dimX = mesh.getDimensions().x(); int dimY = mesh.getDimensions().y();
      //std::cout << "dimX = " << dimX << " ,dimY = " << dimY << std::endl;
      int numOfBlocky = dimY/numThreadsPerBlock + ((dimY%numThreadsPerBlock != 0) ? 1:0);
      int numOfBlockx = dimX/numThreadsPerBlock + ((dimX%numThreadsPerBlock != 0) ? 1:0);
      //std::cout << "numOfBlockx = " << numOfBlockx << " ,numOfBlocky = " << numOfBlocky << std::endl;
      int xkolik = numThreadsPerBlock + 1;
      int ykolik = numThreadsPerBlock + 1;
      
      int blIdx = i%numOfBlockx;
      int blIdy = i/numOfBlockx;
      //std::cout << "blIdx = " << blIdx << " ,blIdy = " << blIdy << std::endl;
      
      if( numOfBlockx - 1 == blIdx )
        xkolik = dimX - (blIdx)*numThreadsPerBlock+1;
      
      if( numOfBlocky -1 == blIdy )
        ykolik = dimY - (blIdy)*numThreadsPerBlock+1;
      //std::cout << "xkolik = " << xkolik << " ,ykolik = " << ykolik << std::endl;
      
      
      /*bool changed[numThreadsPerBlock*numThreadsPerBlock];
       changed[ 0 ] = 1;*/
      Real hx = mesh.getSpaceSteps().x();
      Real hy = mesh.getSpaceSteps().y();
      
      bool changed = false;
      BlockIterHost[ blIdy * numOfBlockx + blIdx ] = 0;
      
      
      Real *sArray;
      sArray = new Real[ sizeSArray * sizeSArray ];
      if( sArray == nullptr )
        std::cout << "Error while allocating memory for sArray." << std::endl;
      
      for( IndexType thri = 0; thri < sizeSArray; thri++ ){
        for( IndexType thrj = 0; thrj < sizeSArray; thrj++ )
          sArray[ thri * sizeSArray + thrj ] = std::numeric_limits< Real >::max();
      }
      
      
      //printf("numThreadsPerBlock = %d\n", numThreadsPerBlock);
      for( IndexType thrj = 0; thrj < numThreadsPerBlock + 1; thrj++ )
      {        
        if( dimX > (blIdx+1) * numThreadsPerBlock  && thrj+1 < ykolik )
          sArray[ ( thrj+1 )* sizeSArray +xkolik] = aux[ blIdy*numThreadsPerBlock*dimX - dimX + blIdx*numThreadsPerBlock - 1 + (thrj+1)*dimX + xkolik ];
        
        
        if( blIdx != 0 && thrj+1 < ykolik )
          sArray[(thrj+1)* sizeSArray] = aux[ blIdy*numThreadsPerBlock*dimX - dimX + blIdx*numThreadsPerBlock - 1 + (thrj+1)*dimX ];
        
        if( dimY > (blIdy+1) * numThreadsPerBlock  && thrj+1 < xkolik )
          sArray[ykolik * sizeSArray + thrj+1] = aux[ blIdy*numThreadsPerBlock*dimX - dimX + blIdx*numThreadsPerBlock - 1 + ykolik*dimX + thrj+1 ];
        
        if( blIdy != 0 && thrj+1 < xkolik )
          sArray[thrj+1] = aux[ blIdy*numThreadsPerBlock*dimX - dimX + blIdx*numThreadsPerBlock - 1 + thrj+1 ];
      }
      
      for( IndexType k = 0; k < numThreadsPerBlock; k++ ){
        for( IndexType l = 0; l < numThreadsPerBlock; l++ )
          if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX )
            sArray[(k+1) * sizeSArray + l+1] = aux[ blIdy * numThreadsPerBlock * dimX + numThreadsPerBlock * blIdx  + k*dimX + l ];
      }
      
      for( IndexType k = 0; k < numThreadsPerBlock; k++ ){ 
        for( IndexType l = 0; l < numThreadsPerBlock; l++ ){
          if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX ){
            //std::cout << "proslo i = " << k * numThreadsPerBlock + l << std::endl;
            if( ! interfaceMap[ blIdy * numThreadsPerBlock * dimX + numThreadsPerBlock * blIdx  + k*dimX + l ] )
            {
              changed = this->template updateCell< sizeSArray >( sArray, l+1, k+1, hx,hy) || changed;
              
            }
          }
        }
      }
      /*aux.save( "aux-1pruch.tnl" );
       for( int k = 0; k < sizeSArray; k++ ){ 
       for( int l = 0; l < sizeSArray; l++ ) {
       std::cout << sArray[ k * sizeSArray + l] << " ";
       }
       std::cout << std::endl;
       }*/
      
      for( IndexType k = 0; k < numThreadsPerBlock; k++ ) 
        for( IndexType l = numThreadsPerBlock-1; l >-1; l-- ) { 
          if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX )
          {
            if( ! interfaceMap[ blIdy * numThreadsPerBlock * dimX + numThreadsPerBlock * blIdx  + k*dimX + l ] )
            {
              this->template updateCell< sizeSArray >( sArray, l+1, k+1, hx,hy);
            }
          }
        }
      /*aux.save( "aux-2pruch.tnl" );
       for( int k = 0; k < sizeSArray; k++ ){ 
       for( int l = 0; l < sizeSArray; l++ ) {
       std::cout << sArray[ k * sizeSArray + l] << " ";
       }
       std::cout << std::endl;
       }*/
      
      for( IndexType k = numThreadsPerBlock-1; k > -1; k-- ) 
        for( IndexType l = 0; l < numThreadsPerBlock; l++ ) {
          if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX )
          {
            if( ! interfaceMap[ blIdy * numThreadsPerBlock * dimX + numThreadsPerBlock * blIdx  + k*dimX + l ] )
            {
              this->template updateCell< sizeSArray >( sArray, l+1, k+1, hx,hy);
            }
          }
        }
      /*aux.save( "aux-3pruch.tnl" );
       for( int k = 0; k < sizeSArray; k++ ){ 
       for( int l = 0; l < sizeSArray; l++ ) {
       std::cout << sArray[ k * sizeSArray + l] << " ";
       }
       std::cout << std::endl;
       }*/
      
      for( IndexType k = numThreadsPerBlock-1; k > -1; k-- ){
        for( IndexType l = numThreadsPerBlock-1; l >-1; l-- ) { 
          if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX )
          {
            if( ! interfaceMap[ blIdy * numThreadsPerBlock * dimX + numThreadsPerBlock * blIdx  + k*dimX + l ] )
            {
              this->template updateCell< sizeSArray >( sArray, l+1, k+1, hx, hy, 1.0);
            }
          }
        }
      }
      /*aux.save( "aux-4pruch.tnl" );
       for( int k = 0; k < sizeSArray; k++ ){ 
       for( int l = 0; l < sizeSArray; l++ ) {
       std::cout << sArray[ k * sizeSArray + l] << " ";
       }
       std::cout << std::endl;
       }*/
      
      
      if( changed ){
        BlockIterHost[ blIdy * numOfBlockx + blIdx ] = 1;
      }
      
      
      for( IndexType k = 0; k < numThreadsPerBlock; k++ ){ 
        for( IndexType l = 0; l < numThreadsPerBlock; l++ ) {
          if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX )      
            helpFunc[ blIdy * numThreadsPerBlock * dimX + numThreadsPerBlock * blIdx  + k*dimX + l ] = sArray[ (k + 1)* sizeSArray + l + 1 ];
          //std::cout<< sArray[k+1][l+1];
        }
        //std::cout<<std::endl;
      }
      delete []sArray;
    }
  }
}
template< typename Real,
        typename Device,
        typename Index >
template< int sizeSArray >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >::
updateBlocks( const InterfaceMapType interfaceMap,
        const MeshFunctionType aux,
        MeshFunctionType& helpFunc,
        ArrayContainer BlockIterHost, int numThreadsPerBlock/*, Real **sArray*/ )
{  
//#pragma omp parallel for schedule( dynamic )
  for( IndexType i = 0; i < BlockIterHost.getSize(); i++ )
  {
    if( BlockIterHost[ i ] )
    {
      MeshType mesh = interfaceMap.template getMesh< Devices::Host >();
      
      int dimX = mesh.getDimensions().x(); int dimY = mesh.getDimensions().y();
      int dimZ = mesh.getDimensions().z();
      //std::cout << "dimX = " << dimX << " ,dimY = " << dimY << std::endl;
      int numOfBlocky = dimY/numThreadsPerBlock + ((dimY%numThreadsPerBlock != 0) ? 1:0);
      int numOfBlockx = dimX/numThreadsPerBlock + ((dimX%numThreadsPerBlock != 0) ? 1:0);
      int numOfBlockz = dimZ/numThreadsPerBlock + ((dimZ%numThreadsPerBlock != 0) ? 1:0);
      //std::cout << "numOfBlockx = " << numOfBlockx << " ,numOfBlocky = " << numOfBlocky << std::endl;
      int xkolik = numThreadsPerBlock + 1;
      int ykolik = numThreadsPerBlock + 1;
      int zkolik = numThreadsPerBlock + 1;
      
      
      int blIdz = i/( numOfBlockx * numOfBlocky );
      int blIdy = (i-blIdz*numOfBlockx * numOfBlocky )/(numOfBlockx );
      int blIdx = (i-blIdz*numOfBlockx * numOfBlocky )%( numOfBlockx );
      //std::cout << "blIdx = " << blIdx << " ,blIdy = " << blIdy << std::endl;
      
      if( numOfBlockx - 1 == blIdx )
        xkolik = dimX - (blIdx)*numThreadsPerBlock+1;
      if( numOfBlocky -1 == blIdy )
        ykolik = dimY - (blIdy)*numThreadsPerBlock+1;
      if( numOfBlockz-1 == blIdz )
        zkolik = dimZ - (blIdz)*numThreadsPerBlock+1;
      //std::cout << "xkolik = " << xkolik << " ,ykolik = " << ykolik << std::endl;
      
      
      /*bool changed[numThreadsPerBlock*numThreadsPerBlock];
       changed[ 0 ] = 1;*/
      Real hx = mesh.getSpaceSteps().x();
      Real hy = mesh.getSpaceSteps().y();
      Real hz = mesh.getSpaceSteps().z();
      
      bool changed = false;
      BlockIterHost[ i ] = 0;
      
      
      Real *sArray;
      sArray = new Real[ sizeSArray * sizeSArray * sizeSArray ];
      if( sArray == nullptr )
        std::cout << "Error while allocating memory for sArray." << std::endl;
      
      for( IndexType k = 0; k < sizeSArray; k++ )
        for( IndexType l = 0; l < sizeSArray; l++ )
          for( IndexType m = 0; m < sizeSArray; m++ ){
            sArray[ m * sizeSArray * sizeSArray + k * sizeSArray + l ] = std::numeric_limits< Real >::max();
          }
      
      
      for( IndexType thrk = 0; thrk < numThreadsPerBlock; thrk++ )
        for( IndexType thrj = 0; thrj < numThreadsPerBlock; thrj++ )
        {
          if( blIdx != 0 && thrj+1 < ykolik && thrk+1 < zkolik )
            sArray[(thrk+1 )* sizeSArray * sizeSArray + (thrj+1)*sizeSArray + 0] = 
                    aux[ blIdz*numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock*dimX + blIdx*numThreadsPerBlock + thrj * dimX -1 + thrk*dimX*dimY ];
          
          if( dimX > (blIdx+1) * numThreadsPerBlock && thrj+1 < ykolik && thrk+1 < zkolik )
            sArray[ (thrk+1) * sizeSArray * sizeSArray + (thrj+1) *sizeSArray + xkolik ] = 
                    aux[ blIdz*numThreadsPerBlock * dimX * dimY + blIdy *numThreadsPerBlock*dimX+ blIdx*numThreadsPerBlock + numThreadsPerBlock + thrj * dimX + thrk*dimX*dimY ];
          
          if( blIdy != 0 && thrj+1 < xkolik && thrk+1 < zkolik )
            sArray[ (thrk+1) * sizeSArray * sizeSArray +0*sizeSArray + thrj+1] = 
                    aux[ blIdz*numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock*dimX + blIdx*numThreadsPerBlock - dimX + thrj + thrk*dimX*dimY ];
          
          if( dimY > (blIdy+1) * numThreadsPerBlock && thrj+1 < xkolik && thrk+1 < zkolik )
            sArray[ (thrk+1) * sizeSArray * sizeSArray + ykolik*sizeSArray + thrj+1] = 
                    aux[ blIdz*numThreadsPerBlock * dimX * dimY + (blIdy+1) * numThreadsPerBlock*dimX + blIdx*numThreadsPerBlock + thrj + thrk*dimX*dimY ];
          
          if( blIdz != 0 && thrj+1 < ykolik && thrk+1 < xkolik )
            sArray[ 0 * sizeSArray * sizeSArray +(thrj+1 )* sizeSArray + thrk+1] = 
                    aux[ blIdz*numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock*dimX + blIdx*numThreadsPerBlock - dimX * dimY + thrj * dimX + thrk ];
          
          if( dimZ > (blIdz+1) * numThreadsPerBlock && thrj+1 < ykolik && thrk+1 < xkolik )
            sArray[zkolik * sizeSArray * sizeSArray + (thrj+1) * sizeSArray + thrk+1] = 
                    aux[ (blIdz+1)*numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock*dimX + blIdx*numThreadsPerBlock + thrj * dimX + thrk ];
        }
      
      for( IndexType m = 0; m < numThreadsPerBlock; m++ ){
        for( IndexType k = 0; k < numThreadsPerBlock; k++ ){
          for( IndexType l = 0; l < numThreadsPerBlock; l++ ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )
              sArray[(m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1] = 
                      aux[ blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l ];
          }
        }
      }
      /*string s;
      int numWhile = 0;
      for( int k = 0; k < numThreadsPerBlock; k++ ){
        for( int l = 0; l < numThreadsPerBlock; l++ ) 
          for( int m = 0; m < numThreadsPerBlock; m++ )
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )     
              helpFunc[ m*dimX*dimY + k*dimX + l ] = sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
      } 
      numWhile++;
      s = "helpFunc-"+ std::to_string(numWhile) + ".tnl";
      helpFunc.save( s );*/
      
      for( IndexType m = 0; m < numThreadsPerBlock; m++ ){
        for( IndexType k = 0; k < numThreadsPerBlock; k++ ){ 
          for( IndexType l = 0; l < numThreadsPerBlock; l++ ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ ){
              //std::cout << "proslo i = " << k * numThreadsPerBlock + l << std::endl;
              if( ! interfaceMap[ blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l  ] )
              {
                //printf("In with point m  = %d, k = %d, l = %d\n", m, k, l);
                changed = this->template updateCell3D< sizeSArray >( sArray, l+1, k+1, m+1, hx,hy,hz) || changed;
                
              }
            }
          }
        }
      }
      /*for( int k = 0; k < numThreadsPerBlock; k++ ){
        for( int l = 0; l < numThreadsPerBlock; l++ ) 
          for( int m = 0; m < numThreadsPerBlock; m++ )
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )     
              helpFunc[ m*dimX*dimY + k*dimX + l ] = sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
      } 
      numWhile++;
      s = "helpFunc-"+ std::to_string(numWhile) + ".tnl";
      helpFunc.save( s );*/
      
      for( IndexType m = numThreadsPerBlock-1; m >-1; m-- ){
        for( IndexType k = 0; k < numThreadsPerBlock; k++ ){
          for( IndexType l = 0; l <numThreadsPerBlock; l++ ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )
            {
              if( ! interfaceMap[ blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l  ] )
              {
                this->template updateCell3D< sizeSArray >( sArray, l+1, k+1, m+1, hx,hy,hz);
              }
            }
          }
        }
      }
      /*for( int k = 0; k < numThreadsPerBlock; k++ ){
        for( int l = 0; l < numThreadsPerBlock; l++ ) 
          for( int m = 0; m < numThreadsPerBlock; m++ )
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )     
              helpFunc[ m*dimX*dimY + k*dimX + l ] = sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
      } 
      numWhile++;
      s = "helpFunc-"+ std::to_string(numWhile) + ".tnl";
      helpFunc.save( s );*/
      
      for( IndexType m = 0; m < numThreadsPerBlock; m++ ){
        for( IndexType k = 0; k < numThreadsPerBlock; k++ ){
          for( IndexType l = numThreadsPerBlock-1; l >-1; l-- ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )
            {
              if( ! interfaceMap[ blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l  ] )
              {
                this->template updateCell3D< sizeSArray >( sArray, l+1, k+1, m+1, hx,hy,hz);
              }
            }
          }
        }
      }
      /*for( int k = 0; k < numThreadsPerBlock; k++ ){
        for( int l = 0; l < numThreadsPerBlock; l++ ) 
          for( int m = 0; m < numThreadsPerBlock; m++ )
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )     
              helpFunc[ m*dimX*dimY + k*dimX + l ] = sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
      } 
      numWhile++;
      s = "helpFunc-"+ std::to_string(numWhile) + ".tnl";
      helpFunc.save( s );
      */
      for( IndexType m = numThreadsPerBlock-1; m >-1; m-- ){
        for( IndexType k = 0; k < numThreadsPerBlock; k++ ){
          for( IndexType l = numThreadsPerBlock-1; l >-1; l-- ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )
            {
              if( ! interfaceMap[ blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l  ] )
              {
                this->template updateCell3D< sizeSArray >(  sArray, l+1, k+1, m+1, hx,hy,hz);
              }
            }
          }
        }
      }
      /*for( int k = 0; k < numThreadsPerBlock; k++ ){
        for( int l = 0; l < numThreadsPerBlock; l++ ) 
          for( int m = 0; m < numThreadsPerBlock; m++ )
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )     
              helpFunc[ m*dimX*dimY + k*dimX + l ] = sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
      } 
      numWhile++;
      s = "helpFunc-"+ std::to_string(numWhile) + ".tnl";
      helpFunc.save( s );*/
      
      for( IndexType m = 0; m < numThreadsPerBlock; m++ ){
        for( IndexType k = numThreadsPerBlock-1; k > -1; k-- ){
          for( IndexType l = 0; l <numThreadsPerBlock; l++ ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )
            {
              if( ! interfaceMap[blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l  ] )
              {
                this->template updateCell3D< sizeSArray >( sArray, l+1, k+1, m+1, hx,hy,hz);
              }
            }
          }
        }
      }
      /*for( int k = 0; k < numThreadsPerBlock; k++ ){
        for( int l = 0; l < numThreadsPerBlock; l++ ) 
          for( int m = 0; m < numThreadsPerBlock; m++ )
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )     
              helpFunc[ m*dimX*dimY + k*dimX + l ] = sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
      } 
      numWhile++;
      s = "helpFunc-"+ std::to_string(numWhile) + ".tnl";
      helpFunc.save( s );*/
      
      for( IndexType m = numThreadsPerBlock-1; m >-1; m-- ){
        for( IndexType k = numThreadsPerBlock-1; k > -1; k-- ){
          for( IndexType l = 0; l <numThreadsPerBlock; l++ ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )
            {
              if( ! interfaceMap[ blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l  ] )
              {
                this->template updateCell3D< sizeSArray >( sArray, l+1, k+1, m+1, hx,hy,hz);
              }
            }
          }
        }
      }
      /*for( int k = 0; k < numThreadsPerBlock; k++ ){
        for( int l = 0; l < numThreadsPerBlock; l++ ) 
          for( int m = 0; m < numThreadsPerBlock; m++ )
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )     
              helpFunc[ m*dimX*dimY + k*dimX + l ] = sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
      } 
      numWhile++;
      s = "helpFunc-"+ std::to_string(numWhile) + ".tnl";
      helpFunc.save( s );*/
      
      for( IndexType m = 0; m < numThreadsPerBlock; m++ ){
        for( IndexType k = numThreadsPerBlock-1; k > -1; k-- ){
          for( IndexType l = numThreadsPerBlock-1; l >-1; l-- ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )
            {
              if( ! interfaceMap[ blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l  ] )
              {
                this->template updateCell3D< sizeSArray >( sArray, l+1, k+1, m+1, hx,hy,hz);
              }
            }
          }
        }
      }
      /*for( int k = 0; k < numThreadsPerBlock; k++ ){
        for( int l = 0; l < numThreadsPerBlock; l++ ) 
          for( int m = 0; m < numThreadsPerBlock; m++ )
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )     
              helpFunc[ m*dimX*dimY + k*dimX + l ] = sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
      } 
      numWhile++;
      s = "helpFunc-"+ std::to_string(numWhile) + ".tnl";
      helpFunc.save( s );*/
      
      
      for( IndexType m = numThreadsPerBlock-1; m >-1; m-- ){
        for( IndexType k = numThreadsPerBlock-1; k > -1; k-- ){
          for( IndexType l = numThreadsPerBlock-1; l >-1; l-- ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )
            {
              if( ! interfaceMap[ blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l  ] )
              {
                this->template updateCell3D< sizeSArray >( sArray, l+1, k+1, m+1, hx,hy,hz);
              }
            }
          }
        }
      }
      /*for( int k = 0; k < numThreadsPerBlock; k++ ){
        for( int l = 0; l < numThreadsPerBlock; l++ ) 
          for( int m = 0; m < numThreadsPerBlock; m++ )
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )     
              helpFunc[ m*dimX*dimY + k*dimX + l ] = sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
      } 
      numWhile++;
      s = "helpFunc-"+ std::to_string(numWhile) + ".tnl";
      helpFunc.save( s );*/
      
      if( changed ){
        BlockIterHost[ i ] = 1;
      }
      
      
      for( IndexType k = 0; k < numThreadsPerBlock; k++ ){ 
        for( IndexType l = 0; l < numThreadsPerBlock; l++ ) {
          for( IndexType m = 0; m < numThreadsPerBlock; m++ ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ ){      
              helpFunc[ blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l ] = 
                      sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
              //std::cout << helpFunc[ m*dimX*dimY + k*dimX + l ] << " ";
            }
          }
          //std::cout << std::endl;
        }
        //std::cout << std::endl;
      }
      //helpFunc.save( "helpF.tnl");
      delete []sArray;
    }
  }
}
template< typename Real,
        typename Device,
        typename Index >
void 
tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >::
getNeighbours( ArrayContainer BlockIterHost, int numBlockX, int numBlockY, int numBlockZ )
{
  int* BlockIterPom; 
  BlockIterPom = new int [ numBlockX * numBlockY * numBlockZ ];
  
  for( int i = 0; i< BlockIterHost.getSize(); i++)
  {
    BlockIterPom[ i ] = 0;
    
    int m=0, l=0, k=0;
    l = i/( numBlockX * numBlockY );
    k = (i-l*numBlockX * numBlockY )/(numBlockX );
    m = (i-l*numBlockX * numBlockY )%( numBlockX );
    
    if( m > 0 && BlockIterHost[ i - 1 ] ){
      BlockIterPom[ i ] = 1;
    }else if( m < numBlockX -1 && BlockIterHost[ i + 1 ] ){
      BlockIterPom[ i ] = 1;
    }else if( k > 0 && BlockIterHost[ i - numBlockX ] ){
      BlockIterPom[ i ] = 1;
    }else if( k < numBlockY -1 && BlockIterHost[ i + numBlockX ] ){
      BlockIterPom[ i ] = 1;
    }else if( l > 0 && BlockIterHost[ i - numBlockX*numBlockY ] ){
      BlockIterPom[ i ] = 1;
    }else if( l < numBlockZ-1 && BlockIterHost[ i + numBlockX*numBlockY ] ){
      BlockIterPom[ i ] = 1;
    }
  }
  for( int i = 0; i< BlockIterHost.getSize(); i++)
  { 
    BlockIterHost[ i ] = BlockIterPom[ i ];
  }
}


template< typename Real,
        typename Device,
        typename Index >
void 
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
getNeighbours( ArrayContainer BlockIterHost, int numBlockX, int numBlockY )
{
  int* BlockIterPom; 
  BlockIterPom = new int [numBlockX * numBlockY];
  
  for(int i = 0; i < numBlockX * numBlockY; i++)
  {
    BlockIterPom[ i ] = 0;//BlockIterPom[ i ] = 0;
    int m=0, k=0;
    m = i%numBlockX;
    k = i/numBlockX;
    if( m > 0 && BlockIterHost[ i - 1 ] ){
      BlockIterPom[ i ] = 1;
    }else if( m < numBlockX -1 && BlockIterHost[ i + 1 ] ){
      BlockIterPom[ i ] = 1;
    }else if( k > 0 && BlockIterHost[ i - numBlockX ] ){
      BlockIterPom[ i ] = 1;
    }else if( k < numBlockY -1 && BlockIterHost[ i + numBlockX ] ){
      BlockIterPom[ i ] = 1;
    }
    //BlockIterPom[ i ];
  }
  
  for(int i = 0; i < numBlockX * numBlockY; i++)
  {
    if( !BlockIterHost[ i ] )
      BlockIterHost[ i ] = BlockIterPom[ i ];
  }
  /*else
   BlockIter[ i ] = 0;*/
  /*for( int i = numBlockX-1; i > -1; i-- )
   {
   for( int j = 0; j< numBlockY; j++ )
   std::cout << BlockIterHost[ i*numBlockY + j ];
   std::cout << std::endl;
   }
   std::cout << std::endl;*/
  delete[] BlockIterPom;
}

template< typename Real,
        typename Device,
        typename Index >
template< typename MeshEntity >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
        const MeshEntity& cell, 
        const RealType v )
{
  const auto& neighborEntities = cell.template getNeighborEntities< 1 >();
  const MeshType& mesh = cell.getMesh();
  const RealType& h = mesh.getSpaceSteps().x();
  const RealType value = u( cell );
  RealType a, tmp = std::numeric_limits< RealType >::max();
  
  if( cell.getCoordinates().x() == 0 )
    a = u[ neighborEntities.template getEntityIndex< 1 >() ];
  else if( cell.getCoordinates().x() == mesh.getDimensions().x() - 1 )
    a = u[ neighborEntities.template getEntityIndex< -1 >() ];
  else
  {
    a = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< -1 >() ],
            u[ neighborEntities.template getEntityIndex<  1 >() ] );
  }
  
  if( fabs( a ) == std::numeric_limits< RealType >::max() )
    return;
  
  tmp = a + TNL::sign( value ) * h/v;
  
  u[ cell.getIndex() ] = argAbsMin( value, tmp );
}

template< typename Real,
        typename Device,
        typename Index >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
initInterface( const MeshFunctionPointer& _input,
        MeshFunctionPointer& _output,
        InterfaceMapPointer& _interfaceMap )
{
  
  if( std::is_same< Device, Devices::Cuda >::value )
  {
#ifdef HAVE_CUDA
    const MeshType& mesh = _input->getMesh();
    
    const int cudaBlockSize( 16 );
    int numBlocksX = Devices::Cuda::getNumberOfBlocks( mesh.getDimensions().x(), cudaBlockSize );
    int numBlocksY = Devices::Cuda::getNumberOfBlocks( mesh.getDimensions().y(), cudaBlockSize );
    dim3 blockSize( cudaBlockSize, cudaBlockSize );
    dim3 gridSize( numBlocksX, numBlocksY );
    Devices::Cuda::synchronizeDevice();
    CudaInitCaller<<< gridSize, blockSize >>>( _input.template getData< Device >(),
            _output.template modifyData< Device >(),
            _interfaceMap.template modifyData< Device >() );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
#endif
  }
  if( std::is_same< Device, Devices::Host >::value )
  {
    MeshFunctionType input = _input.getData();
    
    /*double A[320][320];
     std::ifstream fileInit("/home/maty/Downloads/initData.txt");
     
     for (int i = 0; i < 320; i++)
     for (int j = 0; j < 320; j++)
     fileInit >> A[j];
     fileInit.close();
     for (int i = 0; i < 320; i++)
     for (int j = 0; j < 320; j++)
     input[i*320 + j] = A[j];*/
    
    
    MeshFunctionType& output = _output.modifyData();
    InterfaceMapType& interfaceMap = _interfaceMap.modifyData();
    const MeshType& mesh = input.getMesh();
    typedef typename MeshType::Cell Cell;
    Cell cell( mesh );
    for( cell.getCoordinates().y() = 0;
            cell.getCoordinates().y() < mesh.getDimensions().y();
            cell.getCoordinates().y() ++ )
      for( cell.getCoordinates().x() = 0;
              cell.getCoordinates().x() < mesh.getDimensions().x();
              cell.getCoordinates().x() ++ )
      {
        cell.refresh();
        output[ cell.getIndex() ] =
                input( cell ) >= 0 ? std::numeric_limits< RealType >::max() :
                  - std::numeric_limits< RealType >::max();
        interfaceMap[ cell.getIndex() ] = false;
      }
    
    const RealType& hx = mesh.getSpaceSteps().x();
    const RealType& hy = mesh.getSpaceSteps().y();     
    for( cell.getCoordinates().y() = 0;
            cell.getCoordinates().y() < mesh.getDimensions().y();
            cell.getCoordinates().y() ++ )
      for( cell.getCoordinates().x() = 0;
              cell.getCoordinates().x() < mesh.getDimensions().x();
              cell.getCoordinates().x() ++ )
      {
        cell.refresh();
        const RealType& c = input( cell );
        if( ! cell.isBoundaryEntity()  )
        {
          auto neighbors = cell.getNeighborEntities();
          Real pom = 0;
          const IndexType e = neighbors.template getEntityIndex<  1,  0 >();
          const IndexType n = neighbors.template getEntityIndex<  0,  1 >();
          //Try init with exact data:
          /*if( c * input[ n ] <= 0 )
           {
           output[ cell.getIndex() ] = c;
           output[ n ] = input[ n ];
           interfaceMap[ cell.getIndex() ] = true;
           interfaceMap[ n ] = true;
           }   
           if( c * input[ e ] <= 0 )
           {   
           output[ cell.getIndex() ] = c;
           output[ e ] = input[ e ];
           interfaceMap[ cell.getIndex() ] = true;
           interfaceMap[ e ] = true;
           }*/
          if( c * input[ n ] <= 0 )
          {
            /*if( c >= 0 )
             {*/
            pom = TNL::sign( c )*( hy * c )/( c - input[ n ]);
            if( TNL::abs( output[ cell.getIndex() ] ) > TNL::abs( pom ) ) 
              output[ cell.getIndex() ] = pom;
            pom = pom - TNL::sign( c )*hy;
            if( TNL::abs( output[ n ] ) > TNL::abs( pom ) )
              output[ n ] = pom; //( hy * c )/( c - input[ n ]) - hy;
            /*}else
             {
             pom = - ( hy * c )/( c - input[ n ]);
             if( output[ cell.getIndex() ] < pom )
             output[ cell.getIndex() ] = pom;
             if( output[ n ] > hy + pom )
             output[ n ] = hy + pom; //hy - ( hy * c )/( c - input[ n ]);
             }*/
            interfaceMap[ cell.getIndex() ] = true;
            interfaceMap[ n ] = true;
          }
          if( c * input[ e ] <= 0 )
          {
            /*if( c >= 0 )
             {*/
            pom = TNL::sign( c )*( hx * c )/( c - input[ e ]);
            if( TNL::abs( output[ cell.getIndex() ] ) > TNL::abs( pom ) )
              output[ cell.getIndex() ] = pom;
            
            pom = pom - TNL::sign( c )*hx; //output[ e ] = (hx * c)/( c - input[ e ]) - hx;
            if( TNL::abs( output[ e ] ) > TNL::abs( pom ) )
              output[ e ] = pom; 
            /*}else
             {
             pom = - (hx * c)/( c - input[ e ]);
             if( output[ cell.getIndex() ] < pom )
             output[ cell.getIndex() ] = pom;
             
             pom = pom + hx; //output[ e ] = hx - (hx * c)/( c - input[ e ]);
             if( output[ e ] > pom )
             output[ e ] = pom;
             }*/
            interfaceMap[ cell.getIndex() ] = true;
            interfaceMap[ e ] = true;
          }
        }
      }
  }
}

template< typename Real,
        typename Device,
        typename Index >
template< typename MeshEntity >
__cuda_callable__
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
        const MeshEntity& cell,   
        const RealType v)
{
  const auto& neighborEntities = cell.template getNeighborEntities< 2 >();
  const MeshType& mesh = cell.getMesh();
  const RealType& hx = mesh.getSpaceSteps().x();
  const RealType& hy = mesh.getSpaceSteps().y();
  const RealType value = u( cell );
  RealType a, b, tmp = std::numeric_limits< RealType >::max();
  
  if( cell.getCoordinates().x() == 0 )
    a = u[ neighborEntities.template getEntityIndex< 1,  0 >() ];
  else if( cell.getCoordinates().x() == mesh.getDimensions().x() - 1 )
    a = u[ neighborEntities.template getEntityIndex< -1,  0 >() ];
  else
  {
    a = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< -1,  0 >() ],
            u[ neighborEntities.template getEntityIndex<  1,  0 >() ] );
  }
  
  if( cell.getCoordinates().y() == 0 )
    b = u[ neighborEntities.template getEntityIndex< 0,  1 >()];
  else if( cell.getCoordinates().y() == mesh.getDimensions().y() - 1 )
    b = u[ neighborEntities.template getEntityIndex< 0,  -1 >() ];
  else
  {
    b = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< 0,  -1 >() ],
            u[ neighborEntities.template getEntityIndex< 0,   1 >() ] );
  }
  if( fabs( a ) == std::numeric_limits< RealType >::max() && 
          fabs( b ) == std::numeric_limits< RealType >::max() )
    return;
  /*if( fabs( a ) == TypeInfo< Real >::getMaxValue() ||
   fabs( b ) == TypeInfo< Real >::getMaxValue() ||
   fabs( a - b ) >= TNL::sqrt( (hx * hx + hy * hy)/v ) )
   {
   tmp = 
   fabs( a ) >= fabs( b ) ? b + TNL::sign( value ) * hy :
   a + TNL::sign( value ) * hx;
   }*/
  /*if( fabs( a ) != TypeInfo< Real >::getMaxValue() &&
   fabs( b ) != TypeInfo< Real >::getMaxValue() &&
   fabs( a - b ) < TNL::sqrt( (hx * hx + hy * hy)/v ) )
   {
   tmp = ( hx * hx * b + hy * hy * a + 
   sign( value ) * hx * hy * TNL::sqrt( ( hx * hx + hy * hy )/v - 
   ( a - b ) * ( a - b ) ) )/( hx * hx + hy * hy );
   u[ cell.getIndex() ] =  tmp;
   }
   else
   {
   tmp = 
   fabs( a ) > fabs( b ) ? b + TNL::sign( value ) * hy/v :
   a + TNL::sign( value ) * hx/v;
   u[ cell.getIndex() ] = argAbsMin( value, tmp );
   //tmp = TypeInfo< RealType >::getMaxValue();
   }*/
  RealType pom[6] = { a, b, std::numeric_limits< RealType >::max(), (RealType)hx, (RealType)hy, 0.0 };
  sortMinims( pom );
  tmp = pom[ 0 ] + TNL::sign( value ) * pom[ 3 ]/v;
  
  
  if( fabs( tmp ) < fabs( pom[ 1 ] ) ) 
    u[ cell.getIndex() ] = argAbsMin( value, tmp );
  else
  {
    tmp = ( pom[ 3 ] * pom[ 3 ] * pom[ 1 ] + pom[ 4 ] * pom[ 4 ] * pom[ 0 ] + 
            TNL::sign( value ) * pom[ 3 ] * pom[ 4 ] * TNL::sqrt( ( pom[ 3 ] * pom[ 3 ] +  pom[ 4 ] *  pom[ 4 ] )/( v * v ) - 
            ( pom[ 1 ] - pom[ 0 ] ) * ( pom[ 1 ] - pom[ 0 ] ) ) )/( pom[ 3 ] * pom[ 3 ] + pom[ 4 ] * pom[ 4 ] );
    u[ cell.getIndex() ] = argAbsMin( value, tmp );
  }
}

template< typename Real,
        typename Device,
        typename Index >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >::
initInterface( const MeshFunctionPointer& _input,
        MeshFunctionPointer& _output,
        InterfaceMapPointer& _interfaceMap  )
{
  if( std::is_same< Device, Devices::Cuda >::value )
  {
#ifdef HAVE_CUDA
    const MeshType& mesh = _input->getMesh();
    
    const int cudaBlockSize( 8 );
    int numBlocksX = Devices::Cuda::getNumberOfBlocks( mesh.getDimensions().x(), cudaBlockSize );
    int numBlocksY = Devices::Cuda::getNumberOfBlocks( mesh.getDimensions().y(), cudaBlockSize );
    int numBlocksZ = Devices::Cuda::getNumberOfBlocks( mesh.getDimensions().z(), cudaBlockSize );
    if( cudaBlockSize * cudaBlockSize * cudaBlockSize > 1024 || numBlocksX > 1024 || numBlocksY > 1024 || numBlocksZ > 64 )
      std::cout << "Invalid kernel call. Dimensions of grid are max: [1024,1024,64], and maximum threads per block are 1024!" << std::endl;
    dim3 blockSize( cudaBlockSize, cudaBlockSize, cudaBlockSize );
    dim3 gridSize( numBlocksX, numBlocksY, numBlocksZ );
    Devices::Cuda::synchronizeDevice();
    CudaInitCaller3d<<< gridSize, blockSize >>>( _input.template getData< Device >(),
            _output.template modifyData< Device >(),
            _interfaceMap.template modifyData< Device >() );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
#endif
  }
  if( std::is_same< Device, Devices::Host >::value )
  {
    const MeshFunctionType& input =  _input.getData();
    MeshFunctionType& output =  _output.modifyData();
    InterfaceMapType& interfaceMap =  _interfaceMap.modifyData();
    const MeshType& mesh = input.getMesh();
    typedef typename MeshType::Cell Cell;
    Cell cell( mesh );
    for( cell.getCoordinates().z() = 0;
            cell.getCoordinates().z() < mesh.getDimensions().z();
            cell.getCoordinates().z() ++ )
      for( cell.getCoordinates().y() = 0;
              cell.getCoordinates().y() < mesh.getDimensions().y();
              cell.getCoordinates().y() ++ )
        for( cell.getCoordinates().x() = 0;
                cell.getCoordinates().x() < mesh.getDimensions().x();
                cell.getCoordinates().x() ++ )
        {
          cell.refresh();
          output[ cell.getIndex() ] =
                  input( cell ) > 0 ? 10://std::numeric_limits< RealType >::max() :
                    -10;//- std::numeric_limits< RealType >::max();
          interfaceMap[ cell.getIndex() ] = false;
        }
    
    const RealType& hx = mesh.getSpaceSteps().x();
    const RealType& hy = mesh.getSpaceSteps().y();
    const RealType& hz = mesh.getSpaceSteps().z();
    for( cell.getCoordinates().z() = 0;
            cell.getCoordinates().z() < mesh.getDimensions().z();
            cell.getCoordinates().z() ++ )   
      for( cell.getCoordinates().y() = 0;
              cell.getCoordinates().y() < mesh.getDimensions().y();
              cell.getCoordinates().y() ++ )
        for( cell.getCoordinates().x() = 0;
                cell.getCoordinates().x() < mesh.getDimensions().x();
                cell.getCoordinates().x() ++ )
        {
          cell.refresh();
          const RealType& c = input( cell );
          if( ! cell.isBoundaryEntity() )
          {
            auto neighbors = cell.getNeighborEntities();
            Real pom = 0;
            const IndexType e = neighbors.template getEntityIndex<  1,  0,  0 >();
            const IndexType n = neighbors.template getEntityIndex<  0,  1,  0 >();
            const IndexType t = neighbors.template getEntityIndex<  0,  0,  1 >();
            //Try exact initiation
            /*const IndexType w = neighbors.template getEntityIndex< -1,  0,  0 >();
             const IndexType s = neighbors.template getEntityIndex<  0, -1,  0 >();
             const IndexType b = neighbors.template getEntityIndex<  0,  0, -1 >();
             if( c * input[ e ] <= 0 )
             {
             output[ cell.getIndex() ] = c;
             output[ e ] = input[ e ];
             interfaceMap[ e ] = true;   
             interfaceMap[ cell.getIndex() ] = true;
             }
             else if( c * input[ n ] <= 0 )
             {
             output[ cell.getIndex() ] = c;
             output[ n ] = input[ n ];
             interfaceMap[ n ] = true;   
             interfaceMap[ cell.getIndex() ] = true;
             }
             else if( c * input[ t ] <= 0 )
             {
             output[ cell.getIndex() ] = c;
             output[ t ] = input[ t ];
             interfaceMap[ t ] = true;   
             interfaceMap[ cell.getIndex() ] = true;
             }*/
            if( c * input[ n ] <= 0 )
            {
              if( c >= 0 )
              {
                pom = ( hy * c )/( c - input[ n ]);
                if( output[ cell.getIndex() ] > pom ) 
                  output[ cell.getIndex() ] = pom;
                
                if ( output[ n ] < pom - hy)
                  output[ n ] = pom - hy; // ( hy * c )/( c - input[ n ]) - hy;
                
              }else
              {
                pom = - ( hy * c )/( c - input[ n ]);
                if( output[ cell.getIndex() ] < pom )
                  output[ cell.getIndex() ] = pom;
                if( output[ n ] > hy + pom )
                  output[ n ] = hy + pom; //hy - ( hy * c )/( c - input[ n ]);
                
              }
              interfaceMap[ cell.getIndex() ] = true;
              interfaceMap[ n ] = true;
            }
            if( c * input[ e ] <= 0 )
            {
              if( c >= 0 )
              {
                pom = ( hx * c )/( c - input[ e ]);
                if( output[ cell.getIndex() ] > pom )
                  output[ cell.getIndex() ] = pom;
                
                pom = pom - hx; //output[ e ] = (hx * c)/( c - input[ e ]) - hx;
                if( output[ e ] < pom )
                  output[ e ] = pom;      
                
              }else
              {
                pom = - (hx * c)/( c - input[ e ]);
                if( output[ cell.getIndex() ] < pom )
                  output[ cell.getIndex() ] = pom;
                
                pom = pom + hx; //output[ e ] = hx - (hx * c)/( c - input[ e ]);
                if( output[ e ] > pom )
                  output[ e ] = pom;
              }
              interfaceMap[ cell.getIndex() ] = true;
              interfaceMap[ e ] = true;
            }
            if( c * input[ t ] <= 0 )
            {
              if( c >= 0 )
              {
                pom = ( hz * c )/( c - input[ t ]);
                if( output[ cell.getIndex() ] > pom )
                  output[ cell.getIndex() ] = pom;
                
                pom = pom - hz; //output[ e ] = (hx * c)/( c - input[ e ]) - hx;
                if( output[ t ] < pom )
                  output[ t ] = pom; 
                
              }else
              {
                pom = - (hz * c)/( c - input[ t ]);
                if( output[ cell.getIndex() ] < pom )
                  output[ cell.getIndex() ] = pom;
                
                pom = pom + hz; //output[ e ] = hx - (hx * c)/( c - input[ e ]);
                if( output[ t ] > pom )
                  output[ t ] = pom;
                
              }
              interfaceMap[ cell.getIndex() ] = true;
              interfaceMap[ t ] = true;
            }    
          }
          /*output[ cell.getIndex() ] =
           c > 0 ? TypeInfo< RealType >::getMaxValue() :
           -TypeInfo< RealType >::getMaxValue();
           interfaceMap[ cell.getIndex() ] = false;*/ //is on line 245
        }
  }
}

template< typename Real,
        typename Device,
        typename Index >
template< typename MeshEntity >
__cuda_callable__
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
        const MeshEntity& cell, 
        const RealType v )
{
  const auto& neighborEntities = cell.template getNeighborEntities< 3 >();
  const MeshType& mesh = cell.getMesh();
  
  const RealType& hx = mesh.getSpaceSteps().x();
  const RealType& hy = mesh.getSpaceSteps().y();
  const RealType& hz = mesh.getSpaceSteps().z();
  const RealType value = u( cell );
  //std::cout << value << std::endl;
  RealType a, b, c, tmp = std::numeric_limits< RealType >::max();
  
  
  if( cell.getCoordinates().x() == 0 )
    a = u[ neighborEntities.template getEntityIndex< 1, 0, 0 >() ];
  else if( cell.getCoordinates().x() == mesh.getDimensions().x() - 1 )
    a = u[ neighborEntities.template getEntityIndex< -1, 0, 0 >() ];
  else
  {
    a = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< -1, 0, 0 >() ],
            u[ neighborEntities.template getEntityIndex< 1, 0, 0 >() ] );
  }
  if( cell.getCoordinates().y() == 0 )
    b = u[ neighborEntities.template getEntityIndex< 0, 1, 0 >() ];
  else if( cell.getCoordinates().y() == mesh.getDimensions().y() - 1 )
    b = u[ neighborEntities.template getEntityIndex< 0, -1, 0 >() ];
  else
  {
    b = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< 0, -1, 0 >() ],
            u[ neighborEntities.template getEntityIndex< 0, 1, 0 >() ] );
  }if( cell.getCoordinates().z() == 0 )
    c = u[ neighborEntities.template getEntityIndex< 0, 0, 1 >() ];
  else if( cell.getCoordinates().z() == mesh.getDimensions().z() - 1 )
    c = u[ neighborEntities.template getEntityIndex< 0, 0, -1 >() ];
  else
  {
    c = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< 0, 0, -1 >() ],
            u[ neighborEntities.template getEntityIndex< 0, 0, 1 >() ] );
  }
  if( fabs( a ) == std::numeric_limits< RealType >::max() && 
          fabs( b ) == std::numeric_limits< RealType >::max() &&
          fabs( c ) == std::numeric_limits< RealType >::max() )
    return;
  
  
  /*if( fabs( a ) != TypeInfo< Real >::getMaxValue() &&
   fabs( b ) != TypeInfo< Real >::getMaxValue() &&
   fabs( a - b ) >= TNL::sqrt( (hx * hx + hy * hy)/v ) )
   {
   tmp = ( hx * hx * a + hy * hy * b + 
   sign( value ) * hx * hy * sqrt( ( hx * hx + hy * hy )/v - 
   ( a - b ) * ( a - b ) ) )/( hx * hx + hy * hy );
   }
   if( fabs( a ) != TypeInfo< Real >::getMaxValue() &&
   fabs( c ) != TypeInfo< Real >::getMaxValue() &&
   fabs( a - c ) >= TNL::sqrt( (hx * hx + hz * hz)/v ) )
   {
   tmp = ( hx * hx * a + hz * hz * c + 
   sign( value ) * hx * hz * sqrt( ( hx * hx + hz * hz )/v - 
   ( a - c ) * ( a - c ) ) )/( hx * hx + hz * hz );
   }
   if( fabs( b ) != TypeInfo< Real >::getMaxValue() &&
   fabs( c ) != TypeInfo< Real >::getMaxValue() &&
   fabs( b - c ) >= TNL::sqrt( (hy * hy + hz * hz)/v ) )
   {
   tmp = ( hy * hy * b + hz * hz * c + 
   sign( value ) * hy * hz * sqrt( ( hy * hy + hz * hz )/v - 
   ( b - c ) * ( b - c ) ) )/( hy * hy + hz * hz );
   }*/
  RealType pom[6] = { a, b, c, (RealType)hx, (RealType)hy, (RealType)hz};
  sortMinims( pom );   
  tmp = pom[ 0 ] + TNL::sign( value ) * pom[ 3 ];
  if( fabs( tmp ) < fabs( pom[ 1 ] ) )
  {
    u[ cell.getIndex() ] = argAbsMin( value, tmp ); 
  }
  else
  {
    tmp = ( pom[ 3 ] * pom[ 3 ] * pom[ 1 ] + pom[ 4 ] * pom[ 4 ] * pom[ 0 ] + 
            TNL::sign( value ) * pom[ 3 ] * pom[ 4 ] * TNL::sqrt( ( pom[ 3 ] * pom[ 3 ] +  pom[ 4 ] *  pom[ 4 ] )/( v * v ) - 
            ( pom[ 1 ] - pom[ 0 ] ) * ( pom[ 1 ] - pom[ 0 ] ) ) )/( pom[ 3 ] * pom[ 3 ] + pom[ 4 ] * pom[ 4 ] );
    if( fabs( tmp ) < fabs( pom[ 2 ]) ) 
    {
      u[ cell.getIndex() ] = argAbsMin( value, tmp );
    }
    else
    {
      tmp = ( hy * hy * hz * hz * a + hx * hx * hz * hz * b + hx * hx * hy * hy * c +
              TNL::sign( value ) * hx * hy * hz * TNL::sqrt( ( hx * hx * hz * hz + hy * hy * hz * hz + hx * hx * hy * hy)/( v * v ) - 
              hz * hz * ( a - b ) * ( a - b ) - hy * hy * ( a - c ) * ( a - c ) -
              hx * hx * ( b - c ) * ( b - c ) ) )/( hx * hx * hy * hy + hy * hy * hz * hz + hz * hz * hx *hx );
      u[ cell.getIndex() ] = argAbsMin( value, tmp );
    }
  }
}

template < typename T1 >
__cuda_callable__ void sortMinims( T1 pom[] )
{
  T1 tmp[6] = {0.0,0.0,0.0,0.0,0.0,0.0}; 
  if( fabs(pom[0]) <= fabs(pom[1]) && fabs(pom[1]) <= fabs(pom[2])){
    tmp[0] = pom[0]; tmp[1] = pom[1]; tmp[2] = pom[2];
    tmp[3] = pom[3]; tmp[4] = pom[4]; tmp[5] = pom[5];
    
  }
  else if( fabs(pom[0]) <= fabs(pom[2]) && fabs(pom[2]) <= fabs(pom[1]) ){
    tmp[0] = pom[0]; tmp[1] = pom[2]; tmp[2] = pom[1];
    tmp[3] = pom[3]; tmp[4] = pom[5]; tmp[5] = pom[4];
  }
  else if( fabs(pom[1]) <= fabs(pom[0]) && fabs(pom[0]) <= fabs(pom[2]) ){
    tmp[0] = pom[1]; tmp[1] = pom[0]; tmp[2] = pom[2];
    tmp[3] = pom[4]; tmp[4] = pom[3]; tmp[5] = pom[5];
  }
  else if( fabs(pom[1]) <= fabs(pom[2]) && fabs(pom[2]) <= fabs(pom[0]) ){
    tmp[0] = pom[1]; tmp[1] = pom[2]; tmp[2] = pom[0];
    tmp[3] = pom[4]; tmp[4] = pom[5]; tmp[5] = pom[3];
  }
  else if( fabs(pom[2]) <= fabs(pom[0]) && fabs(pom[0]) <= fabs(pom[1]) ){
    tmp[0] = pom[2]; tmp[1] = pom[0]; tmp[2] = pom[1];
    tmp[3] = pom[5]; tmp[4] = pom[3]; tmp[5] = pom[4];
  }
  else if( fabs(pom[2]) <= fabs(pom[1]) && fabs(pom[1]) <= fabs(pom[0]) ){
    tmp[0] = pom[2]; tmp[1] = pom[1]; tmp[2] = pom[0];
    tmp[3] = pom[5]; tmp[4] = pom[4]; tmp[5] = pom[3];
  }
  
  for( unsigned int i = 0; i < 6; i++ )
  {
    pom[ i ] = tmp[ i ];
  }   
}

template< typename Real,
        typename Device,
        typename Index >
template< int sizeSArray >
__cuda_callable__
bool
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
updateCell( volatile Real *sArray, int thri, int thrj, const Real hx, const Real hy,
        const Real v )
{
  const RealType value = sArray[ thrj * sizeSArray + thri ];
  RealType a, b, tmp = std::numeric_limits< RealType >::max();
  
  b = TNL::argAbsMin( sArray[ (thrj+1) * sizeSArray + thri ],
          sArray[ (thrj-1) * sizeSArray + thri ] );
  
  a = TNL::argAbsMin( sArray[ thrj * sizeSArray + thri+1 ],
          sArray[ thrj * sizeSArray + thri-1 ] );
  
  if( fabs( a ) == std::numeric_limits< RealType >::max() && 
          fabs( b ) == std::numeric_limits< RealType >::max() )
    return false;
  
  RealType pom[6] = { a, b, std::numeric_limits< RealType >::max(), (RealType)hx, (RealType)hy, 0.0 };
  sortMinims( pom );
  tmp = pom[ 0 ] + TNL::sign( value ) * pom[ 3 ]/v;
  
  
  if( fabs( tmp ) < fabs( pom[ 1 ] ) ) 
  {
    sArray[ thrj * sizeSArray + thri ] = argAbsMin( value, tmp );
    tmp = value - sArray[ thrj * sizeSArray + thri ];
    if ( fabs( tmp ) >  0.001*hx )
      return true;
    else
      return false;
  }
  else
  {
    tmp = ( pom[ 3 ] * pom[ 3 ] * pom[ 1 ] + pom[ 4 ] * pom[ 4 ] * pom[ 0 ] + 
            TNL::sign( value ) * pom[ 3 ] * pom[ 4 ] * TNL::sqrt( ( pom[ 3 ] * pom[ 3 ] +  pom[ 4 ] *  pom[ 4 ] )/( v * v ) - 
            ( pom[ 1 ] - pom[ 0 ] ) * ( pom[ 1 ] - pom[ 0 ] ) ) )/( pom[ 3 ] * pom[ 3 ] + pom[ 4 ] * pom[ 4 ] );
    sArray[ thrj * sizeSArray + thri ] = argAbsMin( value, tmp );
    tmp = value - sArray[ thrj * sizeSArray + thri ];
    if ( fabs( tmp ) > 0.001*hx )
      return true;
    else
      return false;
  }
  
  return false;
}
template< typename Real,
        typename Device,
        typename Index >
template< int sizeSArray >
__cuda_callable__ 
bool 
tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >::
updateCell3D( volatile Real *sArray, int thri, int thrj, int thrk,
        const Real hx, const Real hy, const Real hz, const Real v )
{
  const RealType value = sArray[thrk *sizeSArray * sizeSArray + thrj * sizeSArray + thri];
  
  RealType a, b, c, tmp = std::numeric_limits< RealType >::max();
  
  c = TNL::argAbsMin( sArray[ (thrk+1)* sizeSArray*sizeSArray + thrj * sizeSArray + thri ],
          sArray[ (thrk-1) * sizeSArray *sizeSArray + thrj* sizeSArray + thri ] );
  
  b = TNL::argAbsMin( sArray[ thrk* sizeSArray*sizeSArray + (thrj+1) * sizeSArray + thri ],
          sArray[ thrk* sizeSArray * sizeSArray + (thrj-1)* sizeSArray +thri ] );
  
  a = TNL::argAbsMin( sArray[ thrk* sizeSArray* sizeSArray  + thrj* sizeSArray + thri+1 ],
          sArray[ thrk* sizeSArray * sizeSArray + thrj* sizeSArray +thri-1 ] );
  
  /*if( thrk == 8 )
    printf("Calculating a = %f, b = %f, c = %f\n" , a, b, c );*/
  
  if( fabs( a ) == 10&& //std::numeric_limits< RealType >::max() && 
          fabs( b ) == 10&&//std::numeric_limits< RealType >::max() &&
          fabs( c ) == 10)//std::numeric_limits< RealType >::max() )
    return false;
  
  RealType pom[6] = { a, b, c, (RealType)hx, (RealType)hy, (RealType)hz};
  
  sortMinims( pom );
  
  tmp = pom[ 0 ] + TNL::sign( value ) * pom[ 3 ];
  if( fabs( tmp ) < fabs( pom[ 1 ] ) ) 
  {
    sArray[ thrk* sizeSArray* sizeSArray + thrj* sizeSArray + thri ] = argAbsMin( value, tmp );
    tmp = value - sArray[ thrk* sizeSArray* sizeSArray  + thrj* sizeSArray + thri ];
    if ( fabs( tmp ) >  0.001*hx )
      return true;
    else
      return false;
  }
  else
  {
    tmp = ( pom[ 3 ] * pom[ 3 ] * pom[ 1 ] + pom[ 4 ] * pom[ 4 ] * pom[ 0 ] + 
            TNL::sign( value ) * pom[ 3 ] * pom[ 4 ] * TNL::sqrt( ( pom[ 3 ] * pom[ 3 ] +  pom[ 4 ] *  pom[ 4 ] )/( v * v ) - 
            ( pom[ 1 ] - pom[ 0 ] ) * ( pom[ 1 ] - pom[ 0 ] ) ) )/( pom[ 3 ] * pom[ 3 ] + pom[ 4 ] * pom[ 4 ] );
    if( fabs( tmp ) < fabs( pom[ 2 ]) ) 
    {
      sArray[ thrk* sizeSArray* sizeSArray  + thrj* sizeSArray + thri ] = argAbsMin( value, tmp );
      tmp = value - sArray[ thrk* sizeSArray* sizeSArray  + thrj* sizeSArray + thri ];
      if ( fabs( tmp ) > 0.001*hx )
        return true;
      else
        return false;
    }
    else
    {
      tmp = ( hy * hy * hz * hz * a + hx * hx * hz * hz * b + hx * hx * hy * hy * c +
              TNL::sign( value ) * hx * hy * hz * TNL::sqrt( ( hx * hx * hz * hz + hy * hy * hz * hz + hx * hx * hy * hy)/( v * v ) - 
              hz * hz * ( a - b ) * ( a - b ) - hy * hy * ( a - c ) * ( a - c ) -
              hx * hx * ( b - c ) * ( b - c ) ) )/( hx * hx * hy * hy + hy * hy * hz * hz + hz * hz * hx *hx );
      sArray[ thrk* sizeSArray* sizeSArray  + thrj* sizeSArray + thri ] = argAbsMin( value, tmp );
      tmp = value - sArray[ thrk* sizeSArray* sizeSArray  + thrj* sizeSArray + thri ];
      if ( fabs( tmp ) > 0.001*hx )
        return true;
      else
        return false;
    }
  }
  
  return false;
}

#ifdef HAVE_CUDA
template < typename Real, typename Device, typename Index >
__global__ void CudaInitCaller( const Functions::MeshFunction< Meshes::Grid< 1, Real, Device, Index > >& input, 
        Functions::MeshFunction< Meshes::Grid< 1, Real, Device, Index > >& output,
        Functions::MeshFunction< Meshes::Grid< 1, Real, Device, Index >, 1, bool >& interfaceMap  )
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  const Meshes::Grid< 1, Real, Device, Index >& mesh = input.template getMesh< Devices::Cuda >();
  
  if( i < mesh.getDimensions().x()  )
  {
    typedef typename Meshes::Grid< 1, Real, Device, Index >::Cell Cell;
    Cell cell( mesh );
    cell.getCoordinates().x() = i;
    cell.refresh();
    const Index cind = cell.getIndex();
    
    
    output[ cind ] =
            input( cell ) >= 0 ? std::numeric_limits< Real >::max() :
              - std::numeric_limits< Real >::max();
    interfaceMap[ cind ] = false; 
    
    const Real& h = mesh.getSpaceSteps().x();
    cell.refresh();
    const Real& c = input( cell );
    if( ! cell.isBoundaryEntity()  )
    {
      auto neighbors = cell.getNeighborEntities();
      Real pom = 0;
      const Index e = neighbors.template getEntityIndex< 1 >();
      const Index w = neighbors.template getEntityIndex< -1 >();
      if( c * input[ e ] <= 0 )
      {
        pom = TNL::sign( c )*( h * c )/( c - input[ e ]);
        if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) )
          output[ cind ] = pom;                       
        
        interfaceMap[ cind ] = true;
      }
      if( c * input[ w ] <= 0 )
      {
        pom = TNL::sign( c )*( h * c )/( c - input[ w ]);
        if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
          output[ cind ] = pom;
        
        interfaceMap[ cind ] = true;
      }
    }
  }
  
}
template < typename Real, typename Device, typename Index >
__global__ void CudaInitCaller( const Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& input, 
        Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& output,
        Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index >, 2, bool >& interfaceMap ) 
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = blockDim.y*blockIdx.y + threadIdx.y;
  const Meshes::Grid< 2, Real, Device, Index >& mesh = input.template getMesh< Devices::Cuda >();
  
  if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() )
  {
    typedef typename Meshes::Grid< 2, Real, Device, Index >::Cell Cell;
    Cell cell( mesh );
    cell.getCoordinates().x() = i; cell.getCoordinates().y() = j;
    cell.refresh();
    const Index cind = cell.getIndex();
    
    
    output[ cind ] =
            input( cell ) >= 0 ? std::numeric_limits< Real >::max() :
              - std::numeric_limits< Real >::max();
    interfaceMap[ cind ] = false; 
    
    const Real& hx = mesh.getSpaceSteps().x();
    const Real& hy = mesh.getSpaceSteps().y();
    cell.refresh();
    const Real& c = input( cell );
    if( ! cell.isBoundaryEntity()  )
    {
      auto neighbors = cell.getNeighborEntities();
      Real pom = 0;
      const Index e = neighbors.template getEntityIndex<  1,  0 >();
      const Index w = neighbors.template getEntityIndex<  -1,  0 >();
      const Index n = neighbors.template getEntityIndex<  0,  1 >();
      const Index s = neighbors.template getEntityIndex<  0,  -1 >();
      
      if( c * input[ n ] <= 0 )
      {
        pom = TNL::sign( c )*( hy * c )/( c - input[ n ]);
        if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
          output[ cind ] = pom;
        
        interfaceMap[ cell.getIndex() ] = true;
      }
      if( c * input[ e ] <= 0 )
      {
        pom = TNL::sign( c )*( hx * c )/( c - input[ e ]);
        if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) )
          output[ cind ] = pom;                       
        
        interfaceMap[ cind ] = true;
      }
      if( c * input[ w ] <= 0 )
      {
        pom = TNL::sign( c )*( hx * c )/( c - input[ w ]);
        if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
          output[ cind ] = pom;
        
        interfaceMap[ cind ] = true;
      }
      if( c * input[ s ] <= 0 )
      {
        pom = TNL::sign( c )*( hy * c )/( c - input[ s ]);
        if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
          output[ cind ] = pom;
        
        interfaceMap[ cind ] = true;
      }
    }
  }
}

template < typename Real, typename Device, typename Index >
__global__ void CudaInitCaller3d( const Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index > >& input, 
        Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index > >& output,
        Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index >, 3, bool >& interfaceMap )
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = blockDim.y*blockIdx.y + threadIdx.y;
  int k = blockDim.z*blockIdx.z + threadIdx.z;
  const Meshes::Grid< 3, Real, Device, Index >& mesh = input.template getMesh< Devices::Cuda >();
  
  if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() && k < mesh.getDimensions().z() )
  {
    typedef typename Meshes::Grid< 3, Real, Device, Index >::Cell Cell;
    Cell cell( mesh );
    cell.getCoordinates().x() = i; cell.getCoordinates().y() = j; cell.getCoordinates().z() = k;
    cell.refresh();
    const Index cind = cell.getIndex();
    
    
    output[ cind ] =
            input( cell ) >= 0 ? std::numeric_limits< Real >::max() :
              - std::numeric_limits< Real >::max();
    interfaceMap[ cind ] = false; 
    cell.refresh();
    
    const Real& hx = mesh.getSpaceSteps().x();
    const Real& hy = mesh.getSpaceSteps().y();
    const Real& hz = mesh.getSpaceSteps().z();
    const Real& c = input( cell );
    if( ! cell.isBoundaryEntity()  )
    {
      auto neighbors = cell.getNeighborEntities();
      Real pom = 0;
      const Index e = neighbors.template getEntityIndex<  1, 0, 0 >();
      const Index w = neighbors.template getEntityIndex<  -1, 0, 0 >();
      const Index n = neighbors.template getEntityIndex<  0, 1, 0 >();
      const Index s = neighbors.template getEntityIndex<  0, -1, 0 >();
      const Index t = neighbors.template getEntityIndex<  0, 0, 1 >();
      const Index b = neighbors.template getEntityIndex<  0, 0, -1 >();
      
      if( c * input[ n ] <= 0 )
      {
        pom = TNL::sign( c )*( hy * c )/( c - input[ n ]);
        if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
          output[ cind ] = pom;
        
        interfaceMap[ cind ] = true;
      }
      if( c * input[ e ] <= 0 )
      {
        pom = TNL::sign( c )*( hx * c )/( c - input[ e ]);
        if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) )
          output[ cind ] = pom;                       
        
        interfaceMap[ cind ] = true;
      }
      if( c * input[ w ] <= 0 )
      {
        pom = TNL::sign( c )*( hx * c )/( c - input[ w ]);
        if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
          output[ cind ] = pom;
        
        interfaceMap[ cind ] = true;
      }
      if( c * input[ s ] <= 0 )
      {
        pom = TNL::sign( c )*( hy * c )/( c - input[ s ]);
        if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
          output[ cind ] = pom;
        
        interfaceMap[ cind ] = true;
      }
      if( c * input[ b ] <= 0 )
      {
        pom = TNL::sign( c )*( hz * c )/( c - input[ b ]);
        if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
          output[ cind ] = pom;
        
        interfaceMap[ cind ] = true;
      }
      if( c * input[ t ] <= 0 )
      {
        pom = TNL::sign( c )*( hz * c )/( c - input[ t ]);
        if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
          output[ cind ] = pom;
        
        interfaceMap[ cind ] = true;
      }
    }
  }
}




template< typename Real,
        typename Device,
        typename Index >
__cuda_callable__
bool
tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > >::
updateCell( volatile Real sArray[18], int thri, const Real h, const Real v )
{
  const RealType value = sArray[ thri ];
  RealType a, tmp = std::numeric_limits< RealType >::max();
  
  a = TNL::argAbsMin( sArray[ thri+1 ],
          sArray[ thri-1 ] );
  
  if( fabs( a ) == std::numeric_limits< RealType >::max() )
    return false;
  
  tmp = a + TNL::sign( value ) * h/v;
  
  
  sArray[ thri ] = argAbsMin( value, tmp );
  
  tmp = value - sArray[ thri ];
  if ( fabs( tmp ) >  0.001*h )
    return true;
  else
    return false;
}
#endif
