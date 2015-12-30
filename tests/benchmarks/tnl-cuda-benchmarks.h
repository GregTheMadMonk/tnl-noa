/***************************************************************************
                          tnl-benchmarks.h  -  description
                             -------------------
    begin                : Jan 27, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLCUDABENCHMARKS_H_
#define TNLCUDBENCHMARKS_H_

#include <tnlConfig.h>
#include <core/vectors/tnlVector.h>
#include <core/tnlTimerRT.h>
#include <matrices/tnlSlicedEllpackMatrix.h>
#include <matrices/tnlEllpackMatrix.h>

#ifdef HAVE_CUBLAS
//#include <cublas.h>
#endif    

template< typename Matrix >
__global__ void setCudaTestMatrixKernel( Matrix* matrix,
                                         const int elementsPerRow,
                                         const int gridIdx )
{
   const int rowIdx = ( gridIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( rowIdx >= matrix->getRows() )
      return;
   int col = rowIdx - elementsPerRow / 2;   
   for( int element = 0; element < elementsPerRow; element++ )
   {
      if( col + element >= 0 &&
          col + element < matrix->getColumns() )
         matrix->setElementFast( rowIdx, col + element, element + 1 );
   }      
}

template< typename Matrix >
void setCudaTestMatrix( Matrix& matrix,
                        const int elementsPerRow )
{
   typedef typename Matrix::IndexType IndexType;
   typedef typename Matrix::RealType RealType;
   Matrix* kernel_matrix = tnlCuda::passToDevice( matrix );
   dim3 cudaBlockSize( 256 ), cudaGridSize( tnlCuda::getMaxGridSize() );
   const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
   const IndexType cudaGrids = roundUpDivision( cudaBlocks, tnlCuda::getMaxGridSize() );
   for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
   {
      if( gridIdx == cudaGrids - 1 )
         cudaGridSize.x = cudaBlocks % tnlCuda::getMaxGridSize();
      setCudaTestMatrixKernel< Matrix >
       <<< cudaGridSize, cudaBlockSize >>>
       ( kernel_matrix, elementsPerRow, gridIdx );
      checkCudaDevice;
   }
   tnlCuda::freeFromDevice( kernel_matrix );
}

int main( int argc, char* argv[] )
{
#ifdef HAVE_CUDA
   
   typedef double Real;
   typedef tnlVector< Real, tnlHost > HostVector;
   typedef tnlVector< Real, tnlCuda > CudaVector;

   
   /****
    * The first argument of this program is the size od data set to be reduced.
    * If no argument is given we use hardcoded default value.
    */
   int size = 1 << 22;
   if( argc > 1 )
      size = atoi( argv[ 1 ] );
   int loops = 10;
   if( argc > 2 )
      loops = atoi( argv[ 2 ] );
   
   
   
   const double oneGB = 1024.0 * 1024.0 * 1024.0;
   double datasetSize = ( double ) ( loops * size ) * sizeof( Real ) / oneGB;
   
   HostVector hostVector, hostVector2;
   CudaVector deviceVector, deviceVector2;
   hostVector.setSize( size );
   if( ! deviceVector.setSize( size ) )
      return EXIT_FAILURE;
   hostVector2.setLike( hostVector );
   if( ! deviceVector2.setLike( deviceVector ) )
      return EXIT_FAILURE;

   hostVector.setValue( 1.0 );
   deviceVector.setValue( 1.0 );
   hostVector2.setValue( 1.0 );
   deviceVector2.setValue( 1.0 );

   tnlTimerRT timer;
   double bandwidth( 0.0 );

   /*   
   cout << "Benchmarking CPU-GPU memory bandwidth: ";
   timer.reset();
   timer.start();
   for( int i = 0; i < loops; i++ )
     deviceVector = hostVector;
   timer.stop();    
   bandwidth = datasetSize / timer.getTime();
   cout << bandwidth << " GB/sec." << endl;
    
   cout << "Benchmarking vector addition on CPU: ";
   timer.reset();
   timer.start();
   for( int i = 0; i < loops; i++ )
     hostVector.addVector( hostVector2 );
   timer.stop();
   bandwidth = 2 * datasetSize / timer.getTime();
   cout << bandwidth << " GB/sec." << endl;
    
    cout << "Benchmarking vector addition on GPU: ";
    timer.reset();
    timer.start();
    for( int i = 0; i < loops; i++ )
      deviceVector.addVector( deviceVector2 );
    cudaThreadSynchronize();
    timer.stop();
    bandwidth = 3 * datasetSize / timer.getTime();
    cout << bandwidth << " GB/sec." << endl;
    */

   Real resultHost, resultDevice, timeHost, timeDevice;

   cout << "Benchmarking scalar product on CPU: ";
   timer.reset();
   timer.start();
   for( int i = 0; i < loops; i++ )
     resultHost = hostVector.scalarProduct( hostVector2 );
   timer.stop();
   timeHost = timer.getTime();
   bandwidth = 2 * datasetSize / timer.getTime();
   cout << "bandwidth: " << bandwidth << " GB/sec, time: " << timer.getTime() << " sec." << endl;
    
   cout << "Benchmarking scalar product on GPU: ";
   timer.reset();
   timer.start();
   for( int i = 0; i < loops; i++ )
      resultDevice = deviceVector.scalarProduct( deviceVector2 );
   timer.stop();
   timeDevice = timer.getTime();
   bandwidth = 2 * datasetSize / timer.getTime();
   cout << "bandwidth: " << bandwidth << " GB/sec, time: " << timer.getTime() << " sec." << endl;
   cout << "CPU/GPU speedup: " << timeHost / timeDevice << endl;

   if( resultHost != resultDevice )
   {
      cerr << "Error. " << resultHost << " != " << resultDevice << endl;
      //return EXIT_FAILURE;
   }

#ifdef HAVE_CUBLAS
   cout << "Benchmarking scalar product on GPU with Cublas: " << endl;
   cublasHandle_t handle;
   cublasCreate( &handle );
   timer.reset();
   timer.start();   
   for( int i = 0; i < loops; i++ )
      cublasDdot( handle,
                  size,
                  deviceVector.getData(), 1,
                  deviceVector.getData(), 1,
                  &resultDevice );
   cudaThreadSynchronize();
   timer.stop();
   bandwidth = 2 * datasetSize / timer.getTime();
   cout << "Time: " << timer.getTime() << " bandwidth: " << bandwidth << " GB/sec." << endl;
#endif    

   cout << "Benchmarking L2 norm on CPU: ";
   timer.reset();
   timer.start();
   for( int i = 0; i < loops; i++ )
     resultHost = hostVector.lpNorm( 2.0 );
   timer.stop();
   bandwidth = datasetSize / timer.getTime();
   cout << bandwidth << " GB/sec." << endl;
    
   cout << "Benchmarking L2 norm on GPU: " << endl;
   timer.reset();
   timer.start();
   for( int i = 0; i < loops; i++ )
      resultDevice = deviceVector.lpNorm( 2.0 );

   timer.stop();
   bandwidth = datasetSize / timer.getTime();
   cout << "Time: " << timer.getTime() << " bandwidth: " << bandwidth << " GB/sec." << endl;
   if( resultHost != resultDevice )
   {
      cerr << "Error. " << resultHost << " != " << resultDevice << endl;
      //return EXIT_FAILURE;
   }

   /*
   cout << "Benchmarking prefix-sum on CPU ..." << endl;
   timer.reset();
   timer.start();
   hostVector.computePrefixSum();
   timer.stop();
   timeHost = timer.getTime();
   bandwidth = 2 * datasetSize / loops / timer.getTime();
   cout << "bandwidth: " << bandwidth << " GB/sec, time: " << timer.getTime() << " sec." << endl;
   
   cout << "Benchmarking prefix-sum on GPU: ";
   timer.reset();
   timer.start();
   deviceVector.computePrefixSum();
   timer.stop();
   timeDevice = timer.getTime();
   bandwidth = 2 * datasetSize / loops / timer.getTime();
   cout << "bandwidth: " << bandwidth << " GB/sec, time: " << timer.getTime() << " sec." << endl;
   cout << "CPU/GPU speedup: " << timeHost / timeDevice << endl;

   HostVector auxHostVector;
   auxHostVector.setLike( deviceVector );
   auxHostVector = deviceVector;
   for( int i = 0; i < size; i++ )
      if( hostVector.getElement( i ) != auxHostVector.getElement( i ) )
      {
         cerr << "Error in prefix sum at position " << i << ":  " << hostVector.getElement( i ) << " != " << auxHostVector.getElement( i ) << endl;
      }
*/
   /****
    * Sliced Ellpack test
    */
   const int elementsPerRow( 5 );
   typedef tnlEllpackMatrix< double, tnlCuda > DeviceMatrix;
   tnlEllpackMatrix< double, tnlHost > hostMatrix;
   DeviceMatrix deviceMatrix;
   tnlVector< int, tnlHost, int > hostRowLengths;
   tnlVector< int, tnlCuda, int > deviceRowLengths;
   hostRowLengths.setSize( size );
   deviceRowLengths.setSize( size );
   hostMatrix.setDimensions( size, size );
   deviceMatrix.setDimensions( size, size );
   hostRowLengths.setValue( elementsPerRow );
   deviceRowLengths.setValue( elementsPerRow );
   hostMatrix.setCompressedRowsLengths( hostRowLengths );
   if( ! deviceMatrix.setCompressedRowsLengths( deviceRowLengths ) )
   {
      cerr << "Unable to allocate matrix elements." << endl;
      return false;
   }
   int elements( 0 );
   for( int row = 0; row < size; row++ )
   {
      if( row % 100 == 0 )
         cout << "Row " << row << "/" << size << "     \r" << flush;
      int col = row - elementsPerRow / 2;   
      for( int element = 0; element < elementsPerRow; element++ )
      {
         if( col + element >= 0 && col + element < size )
         {
            hostMatrix.setElement( row, col + element, element + 1 );
            //deviceMatrix.setElement( row, col + element, 1.0 );
            elements++;
         }
      }      
   }
   cout << endl;
   setCudaTestMatrix< DeviceMatrix >( deviceMatrix, elementsPerRow );
   datasetSize = loops * elements * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;
   hostVector.setValue( 1.0 );
   deviceVector.setValue( 1.0 );
   cout << "Benchmarking SpMV on CPU: ";
   timer.reset();
   for( int i = 0; i < loops; i++ )
      hostMatrix.vectorProduct( hostVector, hostVector2 );
   timer.stop();
   double hostTime = timer.getTime();
   bandwidth = datasetSize / timer.getTime();
   cout << timer.getTime() << " => " << bandwidth << " GB/s" << endl;
   
   cout << "Benchmarking SpMV on GPU: ";
   deviceVector2.setValue( 0.0 );
   timer.reset();
   for( int i = 0; i < loops; i++ )
      deviceMatrix.vectorProduct( deviceVector, deviceVector2 );
   timer.stop();
      
   if( hostVector2 != deviceVector2 )
      cerr << "Error in Spmv kernel" << endl;
   
   bandwidth = datasetSize / timer.getTime();
   cout << timer.getTime() << " => " << bandwidth << " GB/s" << " speedup " << hostTime / timer.getTime() << endl;
   
   return EXIT_SUCCESS;
#endif
}

#endif /* TNLCUDABENCHMARKS_H_ */
