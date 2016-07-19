/***************************************************************************
                          tnl-benchmark-spmv.h  -  description
                             -------------------
    begin                : Jun 5, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNL_BENCHMARK_SPMV_H_
#define TNL_BENCHMARK_SPMV_H_

#include <fstream>
#include <iomanip>
#include <unistd.h>
#ifdef HAVE_CUDA
#include <cusparse.h>
#endif

#include <TNL/config/tnlConfigDescription.h>
#include <TNL/config/tnlParameterContainer.h>
#include <TNL/matrices/tnlCSRMatrix.h>
#include <TNL/matrices/tnlEllpackMatrix.h>
#include <TNL/matrices/tnlSlicedEllpackMatrix.h>
#include <TNL/matrices/tnlChunkedEllpackMatrix.h>
#include <TNL/matrices/tnlMatrixReader.h>
#include <TNL/core/tnlTimerRT.h>
#include "tnlCusparseCSRMatrix.h"

using namespace std;
using namespace TNL;

void setupConfig( tnlConfigDescription& config )
{
   config.addDelimiter                            ( "General settings:" );
   config.addRequiredEntry< tnlString >( "test" , "Test to be performed." );
      config.addEntryEnum< tnlString >( "mtx" );
      config.addEntryEnum< tnlString >( "tnl" );
   config.addRequiredEntry< tnlString >( "input-file" , "Input file name." );
   config.addEntry< tnlString >( "log-file", "Log file name.", "tnl-benchmark-spmv.log");
   config.addEntry< tnlString >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntry< double >( "stop-time", "Seconds to iterate the SpMV operation.", 1.0 );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
}

bool initLogFile( std::fstream& logFile, const tnlString& fileName )
{
   if( access( fileName.getString(), F_OK ) == -1 )
   {
      logFile.open( fileName.getString(), std::ios::out );
      if( ! logFile )
         return false;
      const tnlString fillingColoring = " : COLORING 0 #FFF8DC 20 #FFFF00 40 #FFD700 60 #FF8C0 80 #FF0000 100";
      const tnlString speedupColoring = " : COLORING #0099FF 1 #FFFFFF 2 #00FF99 4 #33FF99 8 #33FF22 16 #FF9900";
      const tnlString paddingColoring = " : COLORING #FFFFFF 1 #FFFFCC 10 #FFFF99 100 #FFFF66 1000 #FFFF33 10000 #FFFF00";
      logFile << "#Matrix file " << std::endl;
      logFile << "#Rows" << std::endl;
      logFile << "#Columns" << std::endl;
      logFile << "#Non-zero elements" << std::endl;
      logFile << "#Filling (in %)" << fillingColoring << std::endl;
      logFile << "#CSR Format" << std::endl;
      logFile << "# CPU" << std::endl;
      logFile << "#  Gflops" << std::endl;
      logFile << "#  Throughput" << std::endl;
      logFile << "#  Speedup" << speedupColoring << std::endl;
#ifdef HAVE_CUDA
      logFile << "# Cusparse CSR" << std::endl;
      logFile << "#  Gflops" << std::endl;
      logFile << "#  Throughput" << std::endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - cusparse-csr-speedup.txt" << std::endl;
      logFile << "# CUDA" << std::endl;
      logFile << "#  Scalar" << std::endl;
      logFile << "#   Gflops" << std::endl;
      logFile << "#   Throughput" << std::endl;
      logFile << "#   Speedup" << speedupColoring << " SORT - csr-scalar-cuda-speedup.txt" << std::endl;
      logFile << "#  Vector" << std::endl;
      logFile << "#   Warp Size 1" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-vector-1-cuda-speedup.txt" << std::endl;
      logFile << "#   Warp Size 2" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-vector-2-cuda-speedup.txt" << std::endl;
      logFile << "#   Warp Size 4" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-vector-4-cuda-speedup.txt" << std::endl;
      logFile << "#   Warp Size 8" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-vector-8-cuda-speedup.txt" << std::endl;
      logFile << "#   Warp Size 16" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-vector-16-cuda-speedup.txt" << std::endl;
      logFile << "#   Warp Size 32" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-vector-32-cuda-speedup.txt" << std::endl;
      logFile << "#  Hybrid" << std::endl;
      logFile << "#   Split 2" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-hybrid-2-cuda-speedup.txt" << std::endl;
      logFile << "#   Split 4" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-hybrid-4-cuda-speedup.txt" << std::endl;
      logFile << "#   Split 8" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-hybrid-8-cuda-speedup.txt" << std::endl;
      logFile << "#   Split 16" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-hybrid-16-cuda-speedup.txt" << std::endl;
      logFile << "#   Split 32" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-hybrid-32-cuda-speedup.txt" << std::endl;
      logFile << "#   Split 64" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-hybrid-64-cuda-speedup.txt" << std::endl;
#endif
      logFile << "#Ellpack Format" << std::endl;
      logFile << "# Padding (in %)" << paddingColoring << std::endl;
      logFile << "# CPU" << std::endl;
      logFile << "#  Gflops" << std::endl;
      logFile << "#  Throughput" << std::endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - ellpack-host-speedup.txt" << std::endl;
#ifdef HAVE_CUDA
      logFile << "# CUDA" << std::endl;
      logFile << "#  Gflops" << std::endl;
      logFile << "#  Throughput" << std::endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - ellpack-cuda-speedup.txt" << std::endl;
#endif
      logFile << "#SlicedEllpack Format" << std::endl;
      logFile << "# Padding (in %)" << paddingColoring << std::endl;
      logFile << "# CPU" << std::endl;
      logFile << "#  Gflops" << std::endl;
      logFile << "#  Throughput" << std::endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - sliced-ellpack-host-speedup.txt" << std::endl;
#ifdef HAVE_CUDA
      logFile << "# CUDA" << std::endl;
      logFile << "#  Gflops" << std::endl;
      logFile << "#  Throughput" << std::endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - sliced-ellpack-cuda-speedup.txt" << std::endl;
#endif
      logFile << "#ChunkedEllpack Format" << std::endl;
      logFile << "# Padding (in %)" << paddingColoring << std::endl;
      logFile << "# CPU" << std::endl;
      logFile << "#  Gflops" << std::endl;
      logFile << "#  Throughput" << std::endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - chunked-ellpack-host-speedup.txt" << std::endl;
#ifdef HAVE_CUDA
      logFile << "# CUDA" << std::endl;
      logFile << "#  Gflops" << std::endl;
      logFile << "#  Throughput" << std::endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - chunked-ellpack-cuda-speedup.txt" << std::endl;
#endif
      return true;
   }
   logFile.open( fileName.getString(), std::ios::out | std::ios::app );
   //logFile << std::setprecision( 2 );
   if( ! logFile )
      return false;
   return true;
}

template< typename Matrix >
void printMatrixInfo( const tnlString& inputFileName,
                      const Matrix& matrix,
                      std::ostream& str )
{
   str << " Rows: " << std::setw( 8 ) << matrix.getRows();
   str << " Columns: " << std::setw( 8 ) << matrix.getColumns();
   str << " Nonzero Elements: " << std::setw( 10 ) << matrix.getNumberOfNonzeroMatrixElements();
   const double fillingRatio = ( double ) matrix.getNumberOfNonzeroMatrixElements() / ( double ) matrix.getNumberOfMatrixElements();
   str << " Filling: " << std::setw( 5 ) << 100.0 * fillingRatio << "%" << std::endl;
   str << std::setw( 25 ) << "Format"
       << std::setw( 15 ) << "Padding"
       << std::setw( 15 ) << "Time"
       << std::setw( 15 ) << "GFLOPS"
       << std::setw( 15 ) << "Throughput"
       << std::setw( 15 ) << "Speedup" << std::endl;
}

template< typename Matrix >
bool writeMatrixInfo( const tnlString& inputFileName,
                      const Matrix& matrix,
                      std::ostream& logFile )
{
   logFile << std::endl;
   logFile << inputFileName << std::endl;
   logFile << " " << matrix.getRows() << std::endl;
   logFile << " " << matrix.getColumns() << std::endl;
   logFile << " " << matrix.getNumberOfNonzeroMatrixElements() << std::endl;
   const double fillingRatio = ( double ) matrix.getNumberOfNonzeroMatrixElements() / ( double ) matrix.getNumberOfMatrixElements();
   logFile << " " << 100.0 * fillingRatio << std::endl;
   logFile << std::flush;
   if( ! logFile.good() )
      return false;
   return true;
}

double computeGflops( const long int nonzeroElements,
                      const int iterations,
                      const double& time )
{
   return ( double ) ( 2 * iterations * nonzeroElements ) / time * 1.0e-9;
}

template< typename Real >
double computeThroughput( const long int nonzeroElements,
                          const int iterations,
                          const int rows,
                          const double& time )
{
   return ( double ) ( ( 2 * nonzeroElements + rows ) * iterations ) * sizeof( Real ) / time * 1.0e-9;
}

template< typename Matrix,
          typename Vector >
double benchmarkMatrix( const Matrix& matrix,
                        const Vector& x,
                        Vector& b,
                        const long int nonzeroElements,
                        const char* format,
                        const double& stopTime,
                        const double& baseline,
                        int verbose,
                        std::fstream& logFile )
{
   tnlTimerRT timer;
   timer.reset();
   double time( 0.0 );
   int iterations( 0 );
   while( time < stopTime )
   {
      matrix.vectorProduct( x, b );
#ifdef HAVE_CUDA
      if( ( tnlDeviceEnum ) Matrix::DeviceType::DeviceType == tnlCudaDevice )
         cudaThreadSynchronize();
#endif
      time = timer.getTime();
      iterations++;
   }
   const double gflops = computeGflops( nonzeroElements, iterations, time );
   const double throughput = computeThroughput< typename Matrix::RealType >( nonzeroElements, iterations, matrix.getRows(), time );
   const long int allocatedElements = matrix.getNumberOfMatrixElements();
   const double padding = ( double ) allocatedElements / ( double ) nonzeroElements * 100.0 - 100.0;
   if( verbose )
   {
     std::cout << std::setw( 25 ) << format
           << std::setw( 15 ) << padding
           << std::setw( 15 ) << time
           << std::setw( 15 ) << gflops
           << std::setw( 15 ) << throughput;
      if( baseline )
        std::cout << std::setw( 15 ) << gflops / baseline << std::endl;
      else
        std::cout << std::setw( 15 ) << "N/A" << std::endl;
   }
   logFile << "  " << gflops << std::endl;
   logFile << "  " << throughput << std::endl;
   if( baseline )
      logFile << gflops / baseline << std::endl;
   else
      logFile << "N/A" << std::endl;
   return gflops;
}

void writeTestFailed( std::fstream& logFile,
                      int repeat )
{
   for( int i = 0; i < repeat; i++ )
      logFile << "N/A" << std::endl;
}

template< typename Real >
bool setupBenchmark( const tnlParameterContainer& parameters )
{
   const tnlString& test = parameters.getParameter< tnlString >( "test" );
   const tnlString& inputFileName = parameters.getParameter< tnlString >( "input-file" );
   const tnlString& logFileName = parameters.getParameter< tnlString >( "log-file" );
   const int verbose = parameters.getParameter< int >( "verbose" );
   const double stopTime = parameters.getParameter< double >( "stop-time" );
   std::fstream logFile;
   if( ! initLogFile( logFile, logFileName ) )
   {
      std::cerr << "I am not able to open the file " << logFileName << "." << std::endl;
      return false;
   }
   if( test == "mtx" )
   {
      typedef tnlCSRMatrix< Real, tnlHost, int > CSRMatrixType;
      CSRMatrixType csrMatrix;
      try
      {
         if( ! tnlMatrixReader< CSRMatrixType >::readMtxFile( inputFileName, csrMatrix ) )
         {
            std::cerr << "I am not able to read the matrix file " << inputFileName << "." << std::endl;
            logFile << std::endl;
            logFile << inputFileName << std::endl;
            logFile << "Benchmark failed: Unable to read the matrix." << std::endl;
            return false;
         }
      }
      catch( std::bad_alloc )
      {
         std::cerr << "Not enough memory to read the matrix." << std::endl;
         logFile << std::endl;
         logFile << inputFileName << std::endl;
         logFile << "Benchmark failed: Not enough memory." << std::endl;
         return false;
      }
      if( verbose )
         printMatrixInfo( inputFileName, csrMatrix,std::cout );
      if( ! writeMatrixInfo( inputFileName, csrMatrix, logFile ) )
      {
         std::cerr << "I am not able to write new matrix to the log file." << std::endl;
         return false;
      }
      const int rows = csrMatrix.getRows();
      const int columns = csrMatrix.getColumns();
      const long int nonzeroElements = csrMatrix.getNumberOfMatrixElements();
      tnlVector< int, tnlHost, int > rowLengthsHost;
      rowLengthsHost.setSize( rows );
      for( int row = 0; row < rows; row++ )
         rowLengthsHost[ row ] = csrMatrix.getRowLength( row );

      typedef tnlVector< Real, tnlHost, int > HostVector;
      HostVector hostX, hostB;
      hostX.setSize( csrMatrix.getColumns() );
      hostX.setValue( 1.0 );
      hostB.setSize( csrMatrix.getRows() );
#ifdef HAVE_CUDA
      typedef tnlVector< Real, tnlCuda, int > CudaVector;
      CudaVector cudaX, cudaB;
      tnlVector< int, tnlCuda, int > rowLengthsCuda;
      cudaX.setSize( csrMatrix.getColumns() );
      cudaX.setValue( 1.0 );
      cudaB.setSize( csrMatrix.getRows() );
      rowLengthsCuda.setSize( csrMatrix.getRows() );
      rowLengthsCuda = rowLengthsHost;
      cusparseHandle_t cusparseHandle;
      cusparseCreate( &cusparseHandle );
#endif
      const double baseline = benchmarkMatrix( csrMatrix,
                                               hostX,
                                               hostB,
                                               nonzeroElements,
                                               "CSR Host",
                                               stopTime,
                                               0.0,
                                               verbose,
                                               logFile );
#ifdef HAVE_CUDA
      typedef tnlCSRMatrix< Real, tnlCuda, int > CSRMatrixCudaType;
      CSRMatrixCudaType cudaCSRMatrix;
      //cout << "Copying matrix to GPU... ";
      if( ! cudaCSRMatrix.copyFrom( csrMatrix, rowLengthsCuda ) )
      {
         std::cerr << "I am not able to transfer the matrix on GPU." << std::endl;
         writeTestFailed( logFile, 21 );
      }
      else
      {
         tnlCusparseCSRMatrix< Real > cusparseCSRMatrix;
         cusparseCSRMatrix.init( cudaCSRMatrix, &cusparseHandle );
         benchmarkMatrix( cusparseCSRMatrix,
                          cudaX,
                          cudaB,
                          nonzeroElements,
                          "Cusparse CSR",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
         cusparseDestroy( cusparseHandle );

        std::cout << " done.   \r";
         /*cudaCSRMatrix.setCudaKernelType( CSRMatrixCudaType::scalar );
         benchmarkMatrix( cudaCSRMatrix,
                          cudaX,
                          cudaB,
                          nonzeroElements,
                          "CSR Cuda Scalar",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
         cudaCSRMatrix.setCudaKernelType( CSRMatrixCudaType::vector );
         cudaCSRMatrix.setCudaWarpSize( 1 );
         benchmarkMatrix( cudaCSRMatrix,
                          cudaX,
                          cudaB,
                          nonzeroElements,
                          "CSR Cuda Vector 1",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
         cudaCSRMatrix.setCudaWarpSize( 2 );
         benchmarkMatrix( cudaCSRMatrix,
                          cudaX,
                          cudaB,
                          nonzeroElements,
                          "CSR Cuda Vector 2",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
         cudaCSRMatrix.setCudaWarpSize( 4 );
         benchmarkMatrix( cudaCSRMatrix,
                          cudaX,
                          cudaB,
                          nonzeroElements,
                          "CSR Cuda Vector 4",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
         cudaCSRMatrix.setCudaWarpSize( 8 );
         benchmarkMatrix( cudaCSRMatrix,
                          cudaX,
                          cudaB,
                          nonzeroElements,
                          "CSR Cuda Vector 8",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
         cudaCSRMatrix.setCudaWarpSize( 16 );
         benchmarkMatrix( cudaCSRMatrix,
                          cudaX,
                          cudaB,
                          nonzeroElements,
                          "CSR Cuda Vector 16",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
         cudaCSRMatrix.setCudaWarpSize( 32 );
         benchmarkMatrix( cudaCSRMatrix,
                          cudaX,
                          cudaB,
                          nonzeroElements,
                          "CSR Cuda Vector 32",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
         cudaCSRMatrix.setCudaKernelType( CSRMatrixCudaType::hybrid );
         cudaCSRMatrix.setHybridModeSplit( 2 );
         benchmarkMatrix( cudaCSRMatrix,
                          cudaX,
                          cudaB,
                          nonzeroElements,
                          "CSR Cuda Hyrbid 2",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
         cudaCSRMatrix.setHybridModeSplit( 4 );
         benchmarkMatrix( cudaCSRMatrix,
                          cudaX,
                          cudaB,
                          nonzeroElements,
                          "CSR Cuda Hyrbid 4",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
         cudaCSRMatrix.setHybridModeSplit( 8 );
         benchmarkMatrix( cudaCSRMatrix,
                          cudaX,
                          cudaB,
                          nonzeroElements,
                          "CSR Cuda Hyrbid 8",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
         cudaCSRMatrix.setHybridModeSplit( 16 );
         benchmarkMatrix( cudaCSRMatrix,
                          cudaX,
                          cudaB,
                          nonzeroElements,
                          "CSR Cuda Hyrbid 16",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
         cudaCSRMatrix.setHybridModeSplit( 32 );
         benchmarkMatrix( cudaCSRMatrix,
                          cudaX,
                          cudaB,
                          nonzeroElements,
                          "CSR Cuda Hyrbid 32",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
         cudaCSRMatrix.setHybridModeSplit( 64 );
         benchmarkMatrix( cudaCSRMatrix,
                          cudaX,
                          cudaB,
                          nonzeroElements,
                          "CSR Cuda Hyrbid 64",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );*/
      }
      cudaCSRMatrix.reset();
#endif

      long int allocatedElements;
      double padding;
      typedef tnlEllpackMatrix< Real, tnlHost, int > EllpackMatrixType;
      EllpackMatrixType ellpackMatrix;
      if( ! ellpackMatrix.copyFrom( csrMatrix, rowLengthsHost ) )
         writeTestFailed( logFile, 7 );
      else
      {
         allocatedElements = ellpackMatrix.getNumberOfMatrixElements();
         padding = ( double ) allocatedElements / ( double ) nonzeroElements * 100.0 - 100.0;
         logFile << "    " << padding << std::endl;
         benchmarkMatrix( ellpackMatrix,
                          hostX,
                          hostB,
                          nonzeroElements,
                          "Ellpack Host",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
#ifdef HAVE_CUDA
         typedef tnlEllpackMatrix< Real, tnlCuda, int > EllpackMatrixCudaType;
         EllpackMatrixCudaType cudaEllpackMatrix;
        std::cout << "Copying matrix to GPU... ";
         if( ! cudaEllpackMatrix.copyFrom( ellpackMatrix, rowLengthsCuda ) )
         {
            std::cerr << "I am not able to transfer the matrix on GPU." << std::endl;
            writeTestFailed( logFile, 3 );
         }
         else
         {
           std::cout << " done.   \r";
            benchmarkMatrix( cudaEllpackMatrix,
                             cudaX,
                             cudaB,
                             nonzeroElements,
                             "Ellpack Cuda",
                             stopTime,
                             baseline,
                             verbose,
                             logFile );
         }
         cudaEllpackMatrix.reset();
#endif
         ellpackMatrix.reset();
      }

      typedef tnlSlicedEllpackMatrix< Real, tnlHost, int > SlicedEllpackMatrixType;
      SlicedEllpackMatrixType slicedEllpackMatrix;
      if( ! slicedEllpackMatrix.copyFrom( csrMatrix, rowLengthsHost ) )
         writeTestFailed( logFile, 7 );
      else
      {
         allocatedElements = slicedEllpackMatrix.getNumberOfMatrixElements();
         padding = ( double ) allocatedElements / ( double ) nonzeroElements * 100.0 - 100.0;
         logFile << "    " << padding << std::endl;
         benchmarkMatrix( slicedEllpackMatrix,
                          hostX,
                          hostB,
                          nonzeroElements,
                          "SlicedEllpack Host",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
#ifdef HAVE_CUDA
         typedef tnlSlicedEllpackMatrix< Real, tnlCuda, int > SlicedEllpackMatrixCudaType;
         SlicedEllpackMatrixCudaType cudaSlicedEllpackMatrix;
        std::cout << "Copying matrix to GPU... ";
         if( ! cudaSlicedEllpackMatrix.copyFrom( slicedEllpackMatrix, rowLengthsCuda ) )
         {
            std::cerr << "I am not able to transfer the matrix on GPU." << std::endl;
            writeTestFailed( logFile, 3 );
         }
         else
         {
           std::cout << " done.   \r";
            benchmarkMatrix( cudaSlicedEllpackMatrix,
                             cudaX,
                             cudaB,
                             nonzeroElements,
                             "SlicedEllpack Cuda",
                             stopTime,
                             baseline,
                             verbose,
                             logFile );
         }
         cudaSlicedEllpackMatrix.reset();
#endif
         slicedEllpackMatrix.reset();
      }

      typedef tnlChunkedEllpackMatrix< Real, tnlHost, int > ChunkedEllpackMatrixType;
      ChunkedEllpackMatrixType chunkedEllpackMatrix;
      if( ! chunkedEllpackMatrix.copyFrom( csrMatrix, rowLengthsHost ) )
         writeTestFailed( logFile, 7 );
      else
      {
         allocatedElements = chunkedEllpackMatrix.getNumberOfMatrixElements();
         padding = ( double ) allocatedElements / ( double ) nonzeroElements * 100.0 - 100.0;
         logFile << "    " << padding << std::endl;
         benchmarkMatrix( chunkedEllpackMatrix,
                          hostX,
                          hostB,
                          nonzeroElements,
                          "ChunkedEllpack Host",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
#ifdef HAVE_CUDA
         typedef tnlChunkedEllpackMatrix< Real, tnlCuda, int > ChunkedEllpackMatrixCudaType;
         ChunkedEllpackMatrixCudaType cudaChunkedEllpackMatrix;
        std::cout << "Copying matrix to GPU... ";
         if( ! cudaChunkedEllpackMatrix.copyFrom( chunkedEllpackMatrix, rowLengthsCuda ) )
         {
            std::cerr << "I am not able to transfer the matrix on GPU." << std::endl;
            writeTestFailed( logFile, 3 );
         }
         else
         {
           std::cout << " done.    \r";
            benchmarkMatrix( cudaChunkedEllpackMatrix,
                             cudaX,
                             cudaB,
                             nonzeroElements,
                             "ChunkedEllpack Cuda",
                             stopTime,
                             baseline,
                             verbose,
                             logFile );
         }
         cudaChunkedEllpackMatrix.reset();
#endif
         chunkedEllpackMatrix.reset();
      }
   }
   return true;
}

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;

   setupConfig( conf_desc );
 
   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc.printUsage( argv[ 0 ] );
      return 1;
   }
   const tnlString& precision = parameters.getParameter< tnlString >( "precision" );
   if( precision == "float" )
      if( ! setupBenchmark< float >( parameters ) )
         return EXIT_FAILURE;
   if( precision == "double" )
      if( ! setupBenchmark< double >( parameters ) )
         return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

#endif /* TNL_BENCHMARK_SPMV_H_ */
