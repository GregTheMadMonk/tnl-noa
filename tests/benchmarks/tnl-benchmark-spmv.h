/***************************************************************************
                          tnl-benchmark-spmv.h  -  description
                             -------------------
    begin                : Jun 5, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNL_BENCHMARK_SPMV_H_
#define TNL_BENCHMARK_SPMV_H_

#include <fstream>
#include <iomanip>
#include <unistd.h>

#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <matrices/tnlCSRMatrix.h>
#include <matrices/tnlEllpackMatrix.h>
#include <matrices/tnlSlicedEllpackMatrix.h>
#include <matrices/tnlChunkedEllpackMatrix.h>
#include <matrices/tnlMatrixReader.h>
#include <core/tnlTimerRT.h>

using namespace std;

#include "tnlConfig.h"
const char configFile[] = TNL_CONFIG_DIRECTORY "tnl-benchmark-spmv.cfg.desc";

bool initLogFile( fstream& logFile, const tnlString& fileName )
{
   if( access( fileName.getString(), F_OK ) == -1 )
   {
      logFile.open( fileName.getString(), ios::out );
      if( ! logFile )
         return false;
      const tnlString fillingColoring = " : COLORING 0 #FFF8DC 20 #FFFF00 40 #FFD700 60 #FF8C0 80 #FF0000 100";
      const tnlString speedupColoring = " : COLORING #0099FF 1 #FFFFFF 2 #00FF99 4 #33FF99 8 #33FF22 16 #FF9900";
      const tnlString paddingColoring = " : COLORING #FFFFFF 1 #FFFFCC 10 #FFFF99 100 #FFFF66 1000 #FFFF33 10000 #FFFF00";
      logFile << "#Matrix file " << endl;
      logFile << "#Rows" << endl;
      logFile << "#Columns" << endl;
      logFile << "#Non-zero elements" << endl;
      logFile << "#Filling (in %)" << fillingColoring << endl;
      logFile << "#CSR Format" << endl;
      logFile << "# CPU" << endl;
      logFile << "#  Gflops" << endl;
      logFile << "#  Throughput" << endl;
      logFile << "#  Speedup" << speedupColoring << endl;
#ifdef HAVE_CUDA
      logFile << "# CUDA" << endl;
      logFile << "#  Gflops" << endl;
      logFile << "#  Throughput" << endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - csr-cuda-speedup.txt" << endl;
#endif
      logFile << "#Ellpack Format" << endl;
      logFile << "# Padding (in %)" << paddingColoring << endl;
      logFile << "# CPU" << endl;
      logFile << "#  Gflops" << endl;
      logFile << "#  Throughput" << endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - ellpack-host-speedup.txt" << endl;
#ifdef HAVE_CUDA
      logFile << "# CUDA" << endl;
      logFile << "#  Gflops" << endl;
      logFile << "#  Throughput" << endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - ellpack-cuda-speedup.txt" << endl;
#endif
      logFile << "#SlicedEllpack Format" << endl;
      logFile << "# Padding (in %)" << paddingColoring << endl;
      logFile << "# CPU" << endl;
      logFile << "#  Gflops" << endl;
      logFile << "#  Throughput" << endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - sliced-ellpack-host-speedup.txt" << endl;
#ifdef HAVE_CUDA
      logFile << "# CUDA" << endl;
      logFile << "#  Gflops" << endl;
      logFile << "#  Throughput" << endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - sliced-ellpack-cuda-speedup.txt" << endl;
#endif
      logFile << "#ChunkedEllpack Format" << endl;
      logFile << "# Padding (in %)" << paddingColoring << endl;
      logFile << "# CPU" << endl;
      logFile << "#  Gflops" << endl;
      logFile << "#  Throughput" << endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - chunked-ellpack-host-speedup.txt" << endl;
#ifdef HAVE_CUDA
      logFile << "# CUDA" << endl;
      logFile << "#  Gflops" << endl;
      logFile << "#  Throughput" << endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - chunked-ellpack-cuda-speedup.txt" << endl;
#endif
      return true;
   }
   logFile.open( fileName.getString(), ios::out | ios::app );
   //logFile << setprecision( 2 );
   if( ! logFile )
      return false;
   return true;
}

template< typename Matrix >
void printMatrixInfo( const tnlString& inputFileName,
                      const Matrix& matrix,
                      ostream& str )
{
   str << " Rows: " << setw( 8 ) << matrix.getRows();
   str << " Columns: " << setw( 8 ) << matrix.getColumns();
   str << " Nonzero Elements: " << setw( 10 ) << matrix.getNumberOfNonzeroMatrixElements();
   const double fillingRatio = ( double ) matrix.getNumberOfNonzeroMatrixElements() / ( double ) matrix.getNumberOfMatrixElements();
   str << " Filling: " << setw( 5 ) << 100.0 * fillingRatio << "%" << endl;
   str << setw( 25 ) << "Format"
       << setw( 15 ) << "Padding"
       << setw( 15 ) << "Time"
       << setw( 15 ) << "GFLOPS"
       << setw( 15 ) << "Throughput"
       << setw( 15 ) << "Speedup" << endl;
}

template< typename Matrix >
bool writeMatrixInfo( const tnlString& inputFileName,
                      const Matrix& matrix,
                      ostream& logFile )
{
   logFile << endl;
   logFile << inputFileName << endl;
   logFile << " " << matrix.getRows() << endl;
   logFile << " " << matrix.getColumns() << endl;
   logFile << " " << matrix.getNumberOfNonzeroMatrixElements() << endl;
   const double fillingRatio = ( double ) matrix.getNumberOfNonzeroMatrixElements() / ( double ) matrix.getNumberOfMatrixElements();
   logFile << " " << 100.0 * fillingRatio << endl;
   logFile << flush;
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
                        fstream& logFile )
{
   tnlTimerRT timer;
   timer.Reset();
   double time( 0.0 );
   int iterations( 0 );
   while( time < stopTime )
   {
      matrix.vectorProduct( x, b );
#ifdef HAVE_CUDA
      if( Matrix::DeviceType::DeviceType == tnlCudaDevice )
         cudaThreadSynchronize();
#endif
      time = timer.GetTime();
      iterations++;
   }
   const double gflops = computeGflops( nonzeroElements, iterations, time );
   const double throughput = computeThroughput< typename Matrix::RealType >( nonzeroElements, iterations, matrix.getRows(), time );
   const long int allocatedElements = matrix.getNumberOfMatrixElements();
   const double padding = ( double ) allocatedElements / ( double ) nonzeroElements * 100.0 - 100.0;
   if( verbose )
   {
      cout << setw( 25 ) << format
           << setw( 15 ) << padding
           << setw( 15 ) << time
           << setw( 15 ) << gflops
           << setw( 15 ) << throughput;
      if( baseline )
         cout << setw( 15 ) << gflops / baseline << endl;
      else
         cout << setw( 15 ) << "N/A" << endl;
   }
   logFile << "  " << gflops << endl;
   logFile << "  " << throughput << endl;
   if( baseline )
      logFile << gflops / baseline << endl;
   else
      logFile << "N/A" << endl;
   return gflops;
}

void writeTestFailed( fstream& logFile,
                      int repeat )
{
   for( int i = 0; i < repeat; i++ )
      logFile << "N/A" << endl;
}

template< typename Real >
bool setupBenchmark( const tnlParameterContainer& parameters )
{
   const tnlString& test = parameters.GetParameter< tnlString >( "test" );
   const tnlString& inputFileName = parameters.GetParameter< tnlString >( "input-file" );
   const tnlString& logFileName = parameters.GetParameter< tnlString >( "log-file" );
   const int verbose = parameters.GetParameter< int >( "verbose" );
   const double stopTime = parameters.GetParameter< double >( "stop-time" );
   fstream logFile;
   if( ! initLogFile( logFile, logFileName ) )
   {
      cerr << "I am not able to open the file " << logFileName << "." << endl;
      return false;
   }
   if( test == "mtx" )
   {
      typedef tnlCSRMatrix< Real, tnlHost, int > CSRMatrixType;
      CSRMatrixType csrMatrix;
      if( ! tnlMatrixReader< CSRMatrixType >::readMtxFile( inputFileName, csrMatrix ) )
      {
         cerr << "I am not able to read the matrix file " << inputFileName << "." << endl;
         logFile << endl;
         logFile << inputFileName << endl;
         logFile << "Benchmark failed." << endl;
         return false;
      }
      if( verbose )
         printMatrixInfo( inputFileName, csrMatrix, cout );
      if( ! writeMatrixInfo( inputFileName, csrMatrix, logFile ) )
      {
         cerr << "I am not able to write new matrix to the log file." << endl;
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
      cout << "Copying matrix to GPU... ";
      if( ! cudaCSRMatrix.copyFrom( csrMatrix, rowLengthsCuda ) )
      {
         cerr << "I am not able to transfer the matrix on GPU." << endl;
         writeTestFailed( logFile, 3 );
      }
      else
      {
         cout << " done.   \r";
         benchmarkMatrix( cudaCSRMatrix,
                          cudaX,
                          cudaB,
                          nonzeroElements,
                          "CSR Cuda",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
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
         logFile << "    " << padding << endl;
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
         cout << "Copying matrix to GPU... ";
         if( ! cudaEllpackMatrix.copyFrom( ellpackMatrix, rowLengthsCuda ) )
         {
            cerr << "I am not able to transfer the matrix on GPU." << endl;
            writeTestFailed( logFile, 3 );
         }
         else
         {
            cout << " done.   \r";
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
         logFile << "    " << padding << endl;
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
         cout << "Copying matrix to GPU... ";
         if( ! cudaSlicedEllpackMatrix.copyFrom( slicedEllpackMatrix, rowLengthsCuda ) )
         {
            cerr << "I am not able to transfer the matrix on GPU." << endl;
            writeTestFailed( logFile, 3 );
         }
         else
         {
            cout << " done.   \r";
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
         logFile << "    " << padding << endl;
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
         cout << "Copying matrix to GPU... ";
         if( ! cudaChunkedEllpackMatrix.copyFrom( chunkedEllpackMatrix, rowLengthsCuda ) )
         {
            cerr << "I am not able to transfer the matrix on GPU." << endl;
            writeTestFailed( logFile, 3 );
         }
         else
         {
            cout << " done.    \r";
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
}

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;

   if( conf_desc. ParseConfigDescription( configFile ) != 0 )
      return 1;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc. PrintUsage( argv[ 0 ] );
      return 1;
   }
   const tnlString& precision = parameters.GetParameter< tnlString >( "precision" );
   if( precision == "float" )
      if( ! setupBenchmark< float >( parameters ) )
         return EXIT_FAILURE;
   if( precision == "double" )
      if( ! setupBenchmark< double >( parameters ) )
         return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

#endif /* TNL_BENCHMARK_SPMV_H_ */
