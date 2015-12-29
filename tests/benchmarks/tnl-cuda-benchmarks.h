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
#include <core/tnlList.h>
#include <matrices/tnlSlicedEllpackMatrix.h>
#include <matrices/tnlEllpackMatrix.h>
#include <matrices/tnlCSRMatrix.h>

#ifdef HAVE_CUBLAS
//#include <cublas.h>
#endif    

// silly alias to match the number of template parameters with other formats
template< typename Real, typename Device, typename Index >
using SlicedEllpackMatrix = tnlSlicedEllpackMatrix< Real, Device, Index >;

const double oneGB = 1024.0 * 1024.0 * 1024.0;


// TODO:
// check operations with the timer:
//   - reset() clears the timer and starts it again
//   - getTime() stops the timer and starts it again !!!


template< typename Matrix >
int setHostTestMatrix( Matrix& matrix,
                       const int elementsPerRow )
{
   const int size = matrix.getRows();
   int elements( 0 );
   for( int row = 0; row < size; row++ )
   {
      if( row % 100 == 0 )
         cout << "Filling row " << row << "/" << size << "     \r" << flush;
      int col = row - elementsPerRow / 2;
      for( int element = 0; element < elementsPerRow; element++ )
      {
         if( col + element >= 0 &&
             col + element < size )
         {
            matrix.setElement( row, col + element, element + 1 );
            elements++;
         }
      }      
   }
   cout << endl;
   return elements;
}

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

template<typename Function, typename... Args>
double time_void_function(int loops, Function & f, Args & ...args)
{
    tnlTimerRT timer;
    timer.reset();

    for(int i = 0; i < loops; ++i) {
        timer.start();
        f(args...);
        timer.stop();
    }

    return timer.getTime();
}

template< typename Real,
          template< typename, typename, typename > class Matrix,
          template< typename, typename, typename > class Vector = tnlVector >
bool
benchmarkSpMV( const int & loops,
               const int & size,
               const int elementsPerRow = 5 )
{
   typedef Matrix< Real, tnlHost, int > HostMatrix;
   typedef Matrix< Real, tnlCuda, int > DeviceMatrix;
   typedef tnlVector< Real, tnlHost, int > HostVector;
   typedef tnlVector< Real, tnlCuda, int > CudaVector;

   HostMatrix hostMatrix;
   DeviceMatrix deviceMatrix;
   tnlVector< int, tnlHost, int > hostRowLengths;
   tnlVector< int, tnlCuda, int > deviceRowLengths;
   HostVector hostVector, hostVector2;
   CudaVector deviceVector, deviceVector2;

   if( ! hostRowLengths.setSize( size ) ||
       ! deviceRowLengths.setSize( size ) ||
       ! hostMatrix.setDimensions( size, size ) ||
       ! deviceMatrix.setDimensions( size, size ) ||
       ! hostVector.setSize( size ) ||
       ! hostVector2.setSize( size ) ||
       ! deviceVector.setSize( size ) ||
       ! deviceVector2.setSize( size ) )
   {
      cerr << "Unable to allocate all matrices and vectors for the SpMV benchmark." << endl;
      return false;
   }

   hostRowLengths.setValue( elementsPerRow );
   deviceRowLengths.setValue( elementsPerRow );

   if( ! hostMatrix.setCompressedRowsLengths( hostRowLengths ) )
   {
      cerr << "Unable to allocate host matrix elements." << endl;
      return false;
   }
   if( ! deviceMatrix.setCompressedRowsLengths( deviceRowLengths ) )
   {
      cerr << "Unable to allocate device matrix elements." << endl;
      return false;
   }

   double bandwidth( 0.0 ), datasetSize( 0.0 ), timeHost( 0.0 ), timeDevice( 0.0 );

   tnlList< tnlString > parsedType;
   parseObjectType( HostMatrix::getType(), parsedType );
   cout << "Benchmarking SpMV (matrix type: " << parsedType[ 0 ] << ", rows: " << size << ", elements per row: " << elementsPerRow << "):" << endl;

   const int elements = setHostTestMatrix< HostMatrix >( hostMatrix, elementsPerRow );
   setCudaTestMatrix< DeviceMatrix >( deviceMatrix, elementsPerRow );
   datasetSize = loops * elements * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;
   hostVector.setValue( 1.0 );
   deviceVector.setValue( 1.0 );

   auto spmvHost = []( const HostMatrix & m, const HostVector & x, HostVector & y ) {
      m.vectorProduct( x, y );
   };
   timeHost = time_void_function( loops, spmvHost, hostMatrix, hostVector, hostVector2 );
   bandwidth = datasetSize / timeHost;
   cout << "  CPU: bandwidth: " << bandwidth << " GB/sec, time: " << timeHost << " sec." << endl;
   
   auto spmvCuda = []( const DeviceMatrix & m, const CudaVector & x, CudaVector & y ) {
      m.vectorProduct( x, y );
   };
   timeDevice = time_void_function( loops, spmvCuda, deviceMatrix, deviceVector, deviceVector2 );
   bandwidth = datasetSize / timeDevice;
   cout << "  GPU: bandwidth: " << bandwidth << " GB/sec, time: " << timeDevice << " sec." << endl;
   cout << "  CPU/GPU speedup: " << timeHost / timeDevice << endl;

   //cout << hostVector2 << endl << deviceVector2 << endl;
      
   if( hostVector2 != deviceVector2 )
   {      
      cerr << "Error in Spmv kernel" << endl;
      //for( int i = 0; i < size; i++ )
      //   if( hostVector2.getElement( i ) != deviceVector2.getElement( i ) )
      //      cerr << " " << i;
         
   }

   return true;
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
   int elementsPerRow = 5;
   if( argc > 3 )
      elementsPerRow = atoi( argv[ 3 ] );
   
   
   
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

   double bandwidth( 0.0 );
   Real resultHost, resultDevice, timeHost, timeDevice;


   cout << "Benchmarking CPU-GPU memory bandwidth: ";
   auto copyAssign = []( CudaVector & v1, const HostVector & v2 ) {
      v1 = v2;
   };
   timeHost = time_void_function( loops, copyAssign, deviceVector, hostVector );
   bandwidth = datasetSize / timeHost;
   cout << bandwidth << " GB/sec." << endl;
    

   cout << "Benchmarking vector addition:" << endl;
   auto addVectorHost = []( HostVector & v1, const HostVector & v2 ) {
      v1.addVector( v2 );
   };
   timeHost = time_void_function( loops, addVectorHost, hostVector, hostVector2 );
   bandwidth = 3 * datasetSize / timeHost;
   cout << "  CPU: bandwidth: " << bandwidth << " GB/sec, time: " << timeHost << " sec." << endl;
    
   auto addVectorCuda = []( CudaVector & v1, const CudaVector & v2 ) {
      v1.addVector( v2 );
      // TODO: synchronization should be part of addVector
      cudaThreadSynchronize();
   };
   timeDevice = time_void_function( loops, addVectorCuda, deviceVector, deviceVector2 );
   bandwidth = 3 * datasetSize / timeDevice;
   cout << "  GPU: bandwidth: " << bandwidth << " GB/sec, time: " << timeDevice << " sec." << endl;
   cout << "  CPU/GPU speedup: " << timeHost / timeDevice << endl;


   hostVector.setValue( 1.0 );
   deviceVector.setValue( 1.0 );
   hostVector2.setValue( 1.0 );
   deviceVector2.setValue( 1.0 );


   cout << "Benchmarking scalar product:" << endl;
   // FIXME: scalarProduct is not const method
//   auto scalarProductHost = []( const HostVector & v1, const HostVector & v2 ) {
   auto scalarProductHost = []( HostVector & v1, const HostVector & v2 ) {
      return v1.scalarProduct( v2 );
   };
   timeHost = time_void_function( loops, scalarProductHost, hostVector, hostVector2 );
   bandwidth = 2 * datasetSize / timeHost;
   cout << "  CPU: bandwidth: " << bandwidth << " GB/sec, time: " << timeHost << " sec." << endl;
    
   // FIXME: scalarProduct is not const method
//   auto scalarProductCuda = []( const CudaVector & v1, const CudaVector & v2 ) {
   auto scalarProductCuda = []( CudaVector & v1, const CudaVector & v2 ) {
      return v1.scalarProduct( v2 );
   };
   timeDevice = time_void_function( loops, scalarProductCuda, deviceVector, deviceVector2 );
   bandwidth = 2 * datasetSize / timeDevice;
   cout << "  GPU: bandwidth: " << bandwidth << " GB/sec, time: " << timeDevice << " sec." << endl;
   cout << "  CPU/GPU speedup: " << timeHost / timeDevice << endl;

   // TODO: devise a way to check the result of the timed function
//   if( resultHost != resultDevice )
//   {
//      cerr << "Error. " << resultHost << " != " << resultDevice << endl;
      //return EXIT_FAILURE;
//   }

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
   cout << "bandwidth: " << bandwidth << " GB/sec, time: " << timer.getTime() << " sec." << endl;
#endif    

   cout << "Benchmarking L2 norm: " << endl;
   auto l2normHost = []( const HostVector & v ) {
      return v.lpNorm( 2.0 );
   };
   timeHost = time_void_function( loops, l2normHost, hostVector );
   bandwidth = datasetSize / timeHost;
   cout << "  CPU: bandwidth: " << bandwidth << " GB/sec, time: " << timeHost << " sec." << endl;
    
   auto l2normCuda = []( const CudaVector & v ) {
      return v.lpNorm( 2.0 );
   };
   timeDevice = time_void_function( loops, l2normCuda, deviceVector );
   bandwidth = datasetSize / timeDevice;
   cout << "  GPU: bandwidth: " << bandwidth << " GB/sec, time: " << timeDevice << " sec." << endl;
   cout << "  CPU/GPU speedup: " << timeHost / timeDevice << endl;

   // TODO: devise a way to check the result of the timed function
//   if( resultHost != resultDevice )
//   {
//      cerr << "Error. " << resultHost << " != " << resultDevice << endl;
      //return EXIT_FAILURE;
//   }


   /*
   cout << "Benchmarking prefix-sum:" << endl;
   timer.reset();
   timer.start();
   hostVector.computePrefixSum();
   timer.stop();
   timeHost = timer.getTime();
   bandwidth = 2 * datasetSize / loops / timer.getTime();
   cout << "  CPU: bandwidth: " << bandwidth << " GB/sec, time: " << timer.getTime() << " sec." << endl;
   
   timer.reset();
   timer.start();
   deviceVector.computePrefixSum();
   timer.stop();
   timeDevice = timer.getTime();
   bandwidth = 2 * datasetSize / loops / timer.getTime();
   cout << "  GPU: bandwidth: " << bandwidth << " GB/sec, time: " << timer.getTime() << " sec." << endl;
   cout << "  CPU/GPU speedup: " << timeHost / timeDevice << endl;

   HostVector auxHostVector;
   auxHostVector.setLike( deviceVector );
   auxHostVector = deviceVector;
   for( int i = 0; i < size; i++ )
      if( hostVector.getElement( i ) != auxHostVector.getElement( i ) )
      {
         cerr << "Error in prefix sum at position " << i << ":  " << hostVector.getElement( i ) << " != " << auxHostVector.getElement( i ) << endl;
      }
*/

   benchmarkSpMV< Real, tnlEllpackMatrix >( loops, size, elementsPerRow );
   benchmarkSpMV< Real, SlicedEllpackMatrix >( loops, size, elementsPerRow );
   benchmarkSpMV< Real, tnlCSRMatrix >( loops, size, elementsPerRow );
   
   return EXIT_SUCCESS;
#else
   tnlCudaSupportMissingMessage;
   return EXIT_FAILURE;
#endif
}

#endif /* TNLCUDABENCHMARKS_H_ */
