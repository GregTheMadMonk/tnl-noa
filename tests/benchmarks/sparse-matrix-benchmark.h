/***************************************************************************
                          sparse-matrix-benchmark.h  -  description
                             -------------------
    begin                : Jul 27, 2010
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

#ifndef SPARSEMATRIXBENCHMARK_H_
#define SPARSEMATRIXBENCHMARK_H_


#include <fstream>
#include <iomanip>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <matrix/tnlFullMatrix.h>
#include <matrix/tnlFastCSRMatrix.h>
#include <matrix/tnlFastRgCSRMatrix.h>
#include <matrix/tnlFastRgCSRMatrixCUDA.h>
#include <matrix/tnlEllpackMatrix.h>
#include <matrix/tnlEllpackMatrixCUDA.h>
#include <core/mfuncs.h>
#include "tnlSpmvBenchmarkCSRMatrix.h"
#include "tnlSpmvBenchmarkCusparseCSRMatrix.h"
#include "tnlSpmvBenchmarkHybridMatrix.h"
#include "tnlSpmvBenchmarkRgCSRMatrix.h"
#include "tnlSpmvBenchmarkAdaptiveRgCSRMatrix.h"

#include "tnlConfig.h"
const char configFile[] = TNL_CONFIG_DIRECTORY "tnl-sparse-matrix-benchmark.cfg.desc";


using namespace std;

double bestCudaRgCSRGflops( 0 );

template< typename Real >
void benchmarkRgCSRFormat( const tnlCSRMatrix< Real, tnlHost, int >& csrMatrix,
                           const tnlLongVector< Real, tnlHost >& refX,
                           const tnlLongVector< Real, tnlCuda >& cudaX,
                           tnlLongVector< Real, tnlHost >& refB,
                           bool formatTest,
                           const int maxIterations,
                           const bool useAdaptiveGroupSize,
                           const tnlAdaptiveGroupSizeStrategy adaptiveGroupSizeStrategy,
                           const tnlSpmvBenchmarkCSRMatrix< Real, int >& csrMatrixBenchmark,
                           bool verbose,
                           const tnlString& inputMtxFile,
                           const tnlString& logFileName,
                           fstream& logFile )
{
   tnlSpmvBenchmarkRgCSRMatrix< Real, tnlHost, int > hostRgCsrMatrixBenchmark;
   for( int groupSize = 16; groupSize <= 64; groupSize *= 2 )
   {

      hostRgCsrMatrixBenchmark. setGroupSize( groupSize );
      hostRgCsrMatrixBenchmark. setUseAdaptiveGroupSize( useAdaptiveGroupSize );
      hostRgCsrMatrixBenchmark. setAdaptiveGroupSizeStrategy( adaptiveGroupSizeStrategy );
      hostRgCsrMatrixBenchmark. setup( csrMatrix );
      if( formatTest )
         hostRgCsrMatrixBenchmark. testMatrix( csrMatrix, verbose );
      hostRgCsrMatrixBenchmark. setMaxIterations( maxIterations );
      //hostRgCsrMatrixBenchmark. runBenchmark( refX, refB, verbose );
      hostRgCsrMatrixBenchmark. tearDown();

      if( logFileName )
         hostRgCsrMatrixBenchmark. writeToLogTable( logFile,
                                                    csrMatrixBenchmark. getGflops(),
                                                    inputMtxFile,
                                                    csrMatrix,
                                                    true );

      tnlSpmvBenchmarkRgCSRMatrix< Real, tnlCuda, int > cudaRgCsrMatrixBenchmark;
      cudaRgCsrMatrixBenchmark. setGroupSize( groupSize );
      cudaRgCsrMatrixBenchmark. setup( csrMatrix );
      cudaRgCsrMatrixBenchmark. setMaxIterations( maxIterations );
      for( int cudaBlockSize = 32; cudaBlockSize <= 256; cudaBlockSize *= 2 )
      {
         cudaRgCsrMatrixBenchmark. setCudaBlockSize( cudaBlockSize );
         if( formatTest )
            cudaRgCsrMatrixBenchmark. testMatrix( csrMatrix, verbose );
         cudaRgCsrMatrixBenchmark. runBenchmark( cudaX, refB, verbose );
         if( logFileName )
            cudaRgCsrMatrixBenchmark. writeToLogTable( logFile,
                                                       csrMatrixBenchmark. getGflops(),
                                                       inputMtxFile,
                                                       csrMatrix,
                                                       false );
         bestCudaRgCSRGflops = Max( bestCudaRgCSRGflops, cudaRgCsrMatrixBenchmark. getGflops() );
      }
      cudaRgCsrMatrixBenchmark. tearDown();
   }
}

template< class Real >
bool benchmarkMatrix( const tnlString& inputFile,
                      const tnlString& inputMtxFile,
                      const tnlString& pdfFile,
                      const tnlString& logFileName,
                      bool formatTest,
                      int maxIterations,
                      int verbose )
{
   /****
    * Read the CSR matrix ...
    */
   tnlCSRMatrix< Real > csrMatrix( "csr-matrix" );
   tnlString inputMtxSortedFile( inputMtxFile );
   inputMtxSortedFile += tnlString( ".sort" );
   tnlFile binaryFile;
   if( ! binaryFile. open( inputFile, tnlReadMode ) )
   {
      cerr << "I am not able to open the file " << inputFile << "." << endl;
      return 1;
   }
   if( verbose )
      cout << "Reading the CSR matrix ... " << flush;
   if( ! csrMatrix. load( binaryFile ) )
   {
      cerr << "Unable to restore the CSR matrix." << endl;
      return false;
   }
   if( verbose )
      cout << " OK." << endl;
   binaryFile. close();

   /****
    * Check the number of the non-zero elements
    */
   const long int nonzeroElements = csrMatrix. checkNonzeroElements();
   if( nonzeroElements != csrMatrix. getNonzeroElements() )
      cerr << "WARNING: The matrix reports " << csrMatrix. getNonzeroElements() << " but actually there are " << nonzeroElements << " non-zero elements." << endl;
   if( verbose )
      cout << "Matrix size: " << csrMatrix. getSize()
           << " Non-zero elements: " << nonzeroElements << endl;

   const long int size = csrMatrix. getSize();
   tnlLongVector< Real, tnlHost > refX( "ref-x", size ), refB( "ref-b", size);
   tnlLongVector< Real, tnlCuda > cudaX( "cudaX", size );
   refX. setValue( 0.0 );
   for( int i = 0; i < size; i ++ )
      refX[ i ] = 1.0; //( Real ) i * 1.0 / ( Real ) size;
   cudaX = refX;
   csrMatrix. vectorProduct( refX, refB );

   /****
    * CSR format benchmark
    */
   tnlSpmvBenchmarkCSRMatrix< Real, int > csrMatrixBenchmark;

   /****
    * Use the first instance of tnlSpmvBenchmark which we have
    * to write the progress-table header.
    */
   if( verbose )
      csrMatrixBenchmark. writeProgressTableHeader();

   csrMatrixBenchmark. setup( csrMatrix );
   if( formatTest )
   {
      if( verbose )
            cout << "Reading the FULL matrix ... " << endl;
      tnlFullMatrix< Real, tnlHost, int > fullMatrix( "full-matrix" );
      fstream mtxFile;
      mtxFile. open( inputMtxFile. getString(), ios :: in );
      if( ! fullMatrix. read( mtxFile, verbose ) )
         cerr << "Unable to get the FULL matrix." << endl;
      else
         csrMatrixBenchmark. testMatrix( fullMatrix, verbose );
      mtxFile. close();
   }
   csrMatrixBenchmark. setMaxIterations( maxIterations );
   csrMatrixBenchmark. runBenchmark( refX, refB, verbose );
   csrMatrixBenchmark. tearDown();

   /****
    * Open and write one line to the log file
    */
   fstream logFile;
   if( logFileName )
   {
      logFile. open( logFileName. getString(), ios :: out | ios :: app );
      if( ! logFile )
      {
         cerr << "Unable to open log file " << logFileName << " for appending logs." << endl;
         return false;
      }
      /****
       * Open new line of the table and write basic matrix information
       */
      long int allElements = csrMatrix. getSize() * csrMatrix. getSize();
      logFile << "          <tr>" << endl;
      logFile << "             <td> <a href=\"" << pdfFile << "\">" << inputFile << "</a> </td>" << endl;
      logFile << "             <td> " << csrMatrix. getSize() << "</td>" << endl;
      logFile << "             <td> " << nonzeroElements << "</td>" << endl;
      logFile << "             <td> " << ( double ) nonzeroElements / allElements * 100.0 << "</td>" << endl;
      csrMatrixBenchmark. writeToLogTable( logFile,
                                           csrMatrixBenchmark. getGflops(),
                                           inputMtxFile,
                                           csrMatrix,
                                           false );
   }

   /****
    * Cusparse CSR format benchmark
    */
   tnlSpmvBenchmarkCusparseCSRMatrix< Real, int > cusparseCSRMatrixBenchmark;
   cusparseCSRMatrixBenchmark. setup( csrMatrix );
   cusparseCSRMatrixBenchmark. setMaxIterations( maxIterations );
   cusparseCSRMatrixBenchmark. runBenchmark( cudaX, refB, verbose );
   cusparseCSRMatrixBenchmark. tearDown();

   if( logFileName )
       cusparseCSRMatrixBenchmark. writeToLogTable( logFile,
                                                    csrMatrixBenchmark. getGflops(),
                                                    inputMtxFile,
                                                    csrMatrix,
                                                    true );

   /****
    * Hybrid format benchmark
    */
   tnlSpmvBenchmarkHybridMatrix< Real, int > hybridMatrixBenchmark;
   hybridMatrixBenchmark. setFileName( inputMtxFile );
   hybridMatrixBenchmark. setup( csrMatrix );
   hybridMatrixBenchmark. setMaxIterations( maxIterations );
   hybridMatrixBenchmark. setNonzeroElements( nonzeroElements );
   hybridMatrixBenchmark. runBenchmark( refX, refB, verbose );
   hybridMatrixBenchmark. tearDown();

   if( logFileName )
   {
      hybridMatrixBenchmark. writeToLogTable( logFile,
                                              csrMatrixBenchmark. getGflops(),
                                              inputMtxFile,
                                              csrMatrix,
                                              false );
   }

   /****
    * Row-Grouped CSR format
    */
   bestCudaRgCSRGflops = 0.0;
   benchmarkRgCSRFormat( csrMatrix,
                         refX,
                         cudaX,
                         refB,
                         formatTest,
                         maxIterations,
                         false,
                         tnlAdaptiveGroupSizeStrategyByAverageRowSize,
                         csrMatrixBenchmark,
                         verbose,
                         inputMtxFile,
                         logFileName,
                         logFile );

   tnlSpmvBenchmarkRgCSRMatrix< Real, tnlHost, int > hostRgCsrMatrixBenchmark;
   hostRgCsrMatrixBenchmark. setGroupSize( 16 );
   hostRgCsrMatrixBenchmark. setUseAdaptiveGroupSize( true );
   hostRgCsrMatrixBenchmark. setAdaptiveGroupSizeStrategy( tnlAdaptiveGroupSizeStrategyByAverageRowSize );
   hostRgCsrMatrixBenchmark. setup( csrMatrix );
   if( formatTest )
      hostRgCsrMatrixBenchmark. testMatrix( csrMatrix, verbose );
   hostRgCsrMatrixBenchmark. setMaxIterations( maxIterations );
   //hostRgCsrMatrixBenchmark. runBenchmark( refX, refB, verbose );
   hostRgCsrMatrixBenchmark. tearDown();
   if( logFileName )
      hostRgCsrMatrixBenchmark. writeToLogTable( logFile,
                                                 csrMatrixBenchmark. getGflops(),
                                                 inputMtxFile,
                                                 csrMatrix,
                                                 true );
   tnlSpmvBenchmarkRgCSRMatrix< Real, tnlCuda, int > cudaRgCsrMatrixBenchmark;
   for( int cudaBlockSize = 32; cudaBlockSize <= 256; cudaBlockSize *= 2 )
   {
      cudaRgCsrMatrixBenchmark. setCudaBlockSize( cudaBlockSize );
      cudaRgCsrMatrixBenchmark. setGroupSize( 16 );
      cudaRgCsrMatrixBenchmark. setUseAdaptiveGroupSize( true );
      cudaRgCsrMatrixBenchmark. setAdaptiveGroupSizeStrategy( tnlAdaptiveGroupSizeStrategyByAverageRowSize );
      cudaRgCsrMatrixBenchmark. setMaxIterations( maxIterations );
      cudaRgCsrMatrixBenchmark. setup( csrMatrix );
      if( formatTest )
         cudaRgCsrMatrixBenchmark. testMatrix( csrMatrix, verbose );
      cudaRgCsrMatrixBenchmark. runBenchmark( cudaX, refB, verbose );
      if( logFileName )
         cudaRgCsrMatrixBenchmark. writeToLogTable( logFile,
                                                    csrMatrixBenchmark. getGflops(),
                                                    inputMtxFile,
                                                    csrMatrix,
                                                    false );
   }
   cudaRgCsrMatrixBenchmark. tearDown();

   /****
    * Row-Grouped CSR format with reordered rows
    * The rows are now sorted decreasingly by the number of the nonzero elements
    */
   if( verbose )
      cout << "          ------------------------------- Test with sorted matrix ----------------------------------          " << endl;

   tnlLongVector< int, tnlHost > rowPermutation( "rowPermutation" );
   {
      tnlCSRMatrix< Real, tnlHost > orderedCsrMatrix( "orderedCsrMatrix" );
      csrMatrix. sortRowsDecreasingly( rowPermutation );

      /****
       * Check if the ordering is OK.
       */
      int rowSize = csrMatrix. getNonzeroElementsInRow( rowPermutation[ 0 ] );
      for( int i = 1; i < csrMatrix. getSize(); i ++ )
      {
         if( rowSize < csrMatrix. getNonzeroElementsInRow( rowPermutation[ i ] ) )
         {
            cerr << "The rows are not sorted properly. Error is at row number " << i << endl;
         }
         rowSize = csrMatrix. getNonzeroElementsInRow( rowPermutation[ i ] );
      }
      orderedCsrMatrix. reorderRows( rowPermutation, csrMatrix );
      orderedCsrMatrix. vectorProduct( refX, refB );
      benchmarkRgCSRFormat( orderedCsrMatrix,
                            refX,
                            cudaX,
                            refB,
                            formatTest,
                            maxIterations,
                            false,
                            tnlAdaptiveGroupSizeStrategyByAverageRowSize,
                            csrMatrixBenchmark,
                            verbose,
                            inputMtxSortedFile,
                            logFileName,
                            logFile );

      tnlSpmvBenchmarkRgCSRMatrix< Real, tnlHost, int > hostRgCsrMatrixBenchmark;
      hostRgCsrMatrixBenchmark. setGroupSize( 16 );
      hostRgCsrMatrixBenchmark. setUseAdaptiveGroupSize( true ); // TODO: fix with true - not implemented yet
      hostRgCsrMatrixBenchmark. setAdaptiveGroupSizeStrategy( tnlAdaptiveGroupSizeStrategyByFirstGroup );
      hostRgCsrMatrixBenchmark. setMaxIterations( maxIterations );
      hostRgCsrMatrixBenchmark. setup( orderedCsrMatrix );
      if( formatTest )
         hostRgCsrMatrixBenchmark. testMatrix( orderedCsrMatrix, verbose );
      //hostRgCsrMatrixBenchmark. runBenchmark( refX, refB, verbose );
      hostRgCsrMatrixBenchmark. tearDown();
      if( logFileName )
         hostRgCsrMatrixBenchmark. writeToLogTable( logFile,
                                                    csrMatrixBenchmark. getGflops(),
                                                    inputMtxSortedFile,
                                                    csrMatrix,
                                                    true );
      for( int cudaBlockSize = 32; cudaBlockSize <= 256; cudaBlockSize *= 2 )
      {
         tnlSpmvBenchmarkRgCSRMatrix< Real, tnlCuda, int > cudaRgCsrMatrixBenchmark;
         cudaRgCsrMatrixBenchmark. setCudaBlockSize( cudaBlockSize );
         cudaRgCsrMatrixBenchmark. setGroupSize( 16 );
         cudaRgCsrMatrixBenchmark. setUseAdaptiveGroupSize( true );
         cudaRgCsrMatrixBenchmark. setAdaptiveGroupSizeStrategy( tnlAdaptiveGroupSizeStrategyByFirstGroup );
         cudaRgCsrMatrixBenchmark. setup( orderedCsrMatrix );
         cudaRgCsrMatrixBenchmark. setMaxIterations( maxIterations );

         if( formatTest )
            cudaRgCsrMatrixBenchmark. testMatrix( orderedCsrMatrix, verbose );
         cudaRgCsrMatrixBenchmark. runBenchmark( cudaX, refB, verbose );
         if( logFileName )
            cudaRgCsrMatrixBenchmark. writeToLogTable( logFile,
                                                       csrMatrixBenchmark. getGflops(),
                                                       inputMtxSortedFile,
                                                       csrMatrix,
                                                       false );
      }
      cudaRgCsrMatrixBenchmark. tearDown();
   }
   csrMatrix. vectorProduct( refX, refB );

   /****
    * Adaptive Row-Grouped CSR format
    */

   for( int desiredChunkSize = 1; desiredChunkSize <= 32; desiredChunkSize *= 2 )
   {
      tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, tnlCuda, int > cudaArgCsrMatrixBenchmark;
      cudaArgCsrMatrixBenchmark. setDesiredChunkSize( desiredChunkSize );
      for( int cudaBlockSize = 32; cudaBlockSize <= 256; cudaBlockSize *= 2 )
      {
         cudaArgCsrMatrixBenchmark. setCudaBlockSize( cudaBlockSize );
         cudaArgCsrMatrixBenchmark. setup( csrMatrix );
         if( formatTest )
            cudaArgCsrMatrixBenchmark. testMatrix( csrMatrix, verbose );
         cudaArgCsrMatrixBenchmark. setMaxIterations( maxIterations );
         cudaArgCsrMatrixBenchmark. runBenchmark( cudaX, refB, verbose );
         cudaArgCsrMatrixBenchmark. setBestRgCSRGflops( bestCudaRgCSRGflops );
         if( logFileName )
            cudaArgCsrMatrixBenchmark. writeToLogTable( logFile,
                                                        csrMatrixBenchmark. getGflops(),
                                                        inputMtxFile,
                                                        csrMatrix,
                                                        true );
      }
      cudaRgCsrMatrixBenchmark. tearDown();
   }



   if( logFileName )
   {
      logFile << "          </tr>" << endl;
      logFile. close();
   }
   return true;

}

int main( int argc, char* argv[] )
{
   dbgFunctionName( "", "main" );
   dbgInit( "" );

   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;

   if( conf_desc. ParseConfigDescription( configFile ) != 0 )
      return 1;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc. PrintUsage( argv[ 0 ] );
      return 1;
   }
   tnlString inputFile = parameters. GetParameter< tnlString >( "input-file" );
   tnlString inputMtxFile = parameters. GetParameter< tnlString >( "input-mtx-file" );
   tnlString pdfFile = parameters. GetParameter< tnlString >( "pdf-file" );
   tnlString logFileName = parameters. GetParameter< tnlString >( "log-file" );
   double stop_time = parameters. GetParameter< double >( "stop-time" );
   bool formatTest = parameters. GetParameter< bool >( "format-test" );
   int maxIterations = parameters. GetParameter< int >( "max-iterations" );
   int verbose = parameters. GetParameter< int >( "verbose");


   tnlFile binaryFile;
   if( ! binaryFile. open( inputFile, tnlReadMode ) )
   {
      cerr << "I am not able to open the file " << inputFile << "." << endl;
      return 1;
   }
   tnlString object_type;
   if( ! getObjectType( binaryFile, object_type ) )
   {
      cerr << "Unknown object ... SKIPPING!" << endl;
      return EXIT_FAILURE;
   }
   if( verbose )
      cout << object_type << " detected ... " << endl;
   binaryFile. close();

   if( object_type == "tnlCSRMatrix< float, tnlHost >")
      benchmarkMatrix< float >( inputFile,
                                inputMtxFile,
                                pdfFile,
                                logFileName,
                                formatTest,
                                maxIterations,
                                verbose );

   if( object_type == "tnlCSRMatrix< double, tnlHost >" )
   {
      benchmarkMatrix< double >( inputFile,
                                 inputMtxFile,
                                 pdfFile,
                                 logFileName,
                                 formatTest,
                                 maxIterations,
                                 verbose );
   }
   //binaryFile. close();



   return EXIT_SUCCESS;
}


#endif /* SPARSEMATRIXBENCHMARK_H_ */
