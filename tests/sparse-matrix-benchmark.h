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

//#define HAVE_CUDA

#include <fstream>
#include <iomanip>
#include <tnlSpmvBenchmarkCSRMatrix.h>
#include <tnlSpmvBenchmarkHybridMatrix.h>
#include <tnlSpmvBenchmarkRgCSRMatrix.h>
#include <tnlSpmvBenchmarkAdaptiveRgCSRMatrix.h>
#include <matrix/tnlFastCSRMatrix.h>
#include <matrix/tnlFastRgCSRMatrix.h>
#include <matrix/tnlFastRgCSRMatrixCUDA.h>
#include <matrix/tnlEllpackMatrix.h>
#include <matrix/tnlEllpackMatrixCUDA.h>
#include <core/mfuncs.h>
#include <config.h>

using namespace std;

tnlString getBgColorBySpeedUp( const double& speedUp )
{
   if( speedUp >= 30.0 )
      return tnlString( "#FF9900" );
   if( speedUp >= 25.0 )
      return tnlString( "#FFAA00" );
   if( speedUp >= 20.0 )
      return tnlString( "#FFBB00" );
   if( speedUp >= 15.0 )
      return tnlString( "#FFCC00" );
   if( speedUp >= 10.0 )
      return tnlString( "#FFDD00" );
   if( speedUp >= 5.0 )
      return tnlString( "#FFEE00" );
   if( speedUp >= 1.0 )
      return tnlString( "#FFFF00" );
   return tnlString( "#FFFFFF" );
}

template< typename Real >
void benchmarkRgCSRFormat( const tnlCSRMatrix< Real, tnlHost, int >& csrMatrix,
                           const tnlLongVector< Real, tnlHost >& refX,
                           const tnlLongVector< Real, tnlCuda >& cudaX,
                           tnlLongVector< Real, tnlHost >& refB,
                           const bool useAdaptiveGroupSize,
                           const tnlAdaptiveGroupSizeStrategy adaptiveGroupSizeStrategy,
                           const tnlSpmvBenchmarkCSRMatrix< Real, int >& csrMatrixBenchmark,
                           bool verbose,
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
      hostRgCsrMatrixBenchmark. runBenchmark( refX, refB, verbose );
      hostRgCsrMatrixBenchmark. tearDown();

      if( logFileName )
      {
         if( hostRgCsrMatrixBenchmark. getBenchmarkWasSuccesful() )
         {
            tnlString bgColor;
            switch( groupSize )
            {
               case 16: bgColor = "#55FF55"; break;
               case 32: bgColor = "#99FF99"; break;
               case 64: bgColor = "#CCFFCC"; break;
               default: bgColor = "#FFFFFF";
            }
            logFile << "             <td bgcolor=" << bgColor << "> " << hostRgCsrMatrixBenchmark. getArtificialZeroElements() << "</td>" << endl;
            logFile << "             <td bgcolor=" << bgColor << "> " << hostRgCsrMatrixBenchmark. getGflops() << "</td>" << endl;
            logFile << "             <td bgcolor=" << bgColor << "> " << hostRgCsrMatrixBenchmark. getGflops() / csrMatrixBenchmark. getGflops() << "</td>" << endl;
         }
         else
         {
            logFile << "             <td bgcolor=#FFFFFF> N/A </td>" << endl;
            logFile << "             <td bgcolor=#FFFFFF> N/A </td>" << endl;
            logFile << "             <td bgcolor=#FFFFFF> N/A </td>" << endl;
         }
      }
      tnlSpmvBenchmarkRgCSRMatrix< Real, tnlCuda, int > cudaRgCsrMatrixBenchmark;
      cudaRgCsrMatrixBenchmark. setGroupSize( groupSize );
      cudaRgCsrMatrixBenchmark. setup( csrMatrix );
      for( int cudaBlockSize = 32; cudaBlockSize <= 256; cudaBlockSize *= 2 )
      {
         cudaRgCsrMatrixBenchmark. setCudaBlockSize( cudaBlockSize );
         cudaRgCsrMatrixBenchmark. runBenchmark( cudaX, refB, verbose );
         if( logFileName )
         {
            if( cudaRgCsrMatrixBenchmark. getBenchmarkWasSuccesful() )
            {
               logFile << "             <td> " << cudaRgCsrMatrixBenchmark. getGflops() << "</td>" << endl;
               double speedUp = cudaRgCsrMatrixBenchmark. getGflops() / csrMatrixBenchmark. getGflops();
               tnlString bgColor = getBgColorBySpeedUp( speedUp );
               logFile << "             <td bgcolor=" << bgColor << "> " << speedUp << "</td>" << endl;
            }
            else
            {
               logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
               logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
            }
         }
      }
      cudaRgCsrMatrixBenchmark. tearDown();
   }
}

template< class Real >
bool benchmarkMatrix( const tnlString& input_file,
                      const tnlString& input_mtx_file,
                      const tnlString& logFileName,
                      int verbose )
{
   /****
    * Read the CSR matrix ...
    */
   tnlCSRMatrix< Real > csrMatrix( "csr-matrix" );
   tnlFile binaryFile;
   if( ! binaryFile. open( input_file, tnlReadMode ) )
   {
      cerr << "I am not able to open the file " << input_file << "." << endl;
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
      long int allElements = csrMatrix. getSize() * csrMatrix. getSize();
      logFile << "          <tr>" << endl;
      logFile << "             <td> " << input_file << "</td>" << endl;
      logFile << "             <td> " << csrMatrix. getSize() << "</td>" << endl;
      logFile << "             <td> " << nonzeroElements << "</td>" << endl;
      logFile << "             <td> " << ( double ) nonzeroElements / allElements * 100.0 << "</td>" << endl;
      if( csrMatrixBenchmark. getBenchmarkWasSuccesful() )
         logFile << "             <td> " << csrMatrixBenchmark. getGflops() << "</td>" << endl;
      else
         logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
   }

   /****
    * Hybrid format benchmark
    */
   tnlSpmvBenchmarkHybridMatrix< Real, int > hybridMatrixBenchmark;
   hybridMatrixBenchmark. setFileName( input_mtx_file );
   hybridMatrixBenchmark. setNonzeroElements( csrMatrix. getNonzeroElements() );
   hybridMatrixBenchmark. setup( csrMatrix );
   hybridMatrixBenchmark. runBenchmark( refX, refB, verbose );
   hybridMatrixBenchmark. tearDown();

   if( logFileName )
   {
      if( hybridMatrixBenchmark. getBenchmarkWasSuccesful() )
      {
         logFile << "             <td> " << hybridMatrixBenchmark. getGflops() << "</td>" << endl;
         double speedUp = hybridMatrixBenchmark. getGflops() / csrMatrixBenchmark. getGflops();;
         tnlString bgColor = getBgColorBySpeedUp( speedUp );
         logFile << "             <td bgcolor=" << bgColor << "> " << speedUp << "</td>" << endl;
      }
      else
      {
         logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
         logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;

      }
   }

   /****
    * Row-Grouped CSR format
    */
   benchmarkRgCSRFormat( csrMatrix,
                         refX,
                         cudaX,
                         refB,
                         false,
                         tnlAdaptiveGroupSizeStrategyByAverageRowSize,
                         csrMatrixBenchmark,
                         verbose,
                         logFileName,
                         logFile );

   tnlSpmvBenchmarkRgCSRMatrix< Real, tnlHost, int > hostRgCsrMatrixBenchmark;
   hostRgCsrMatrixBenchmark. setGroupSize( 16 );
   hostRgCsrMatrixBenchmark. setUseAdaptiveGroupSize( true );
   hostRgCsrMatrixBenchmark. setAdaptiveGroupSizeStrategy( tnlAdaptiveGroupSizeStrategyByAverageRowSize );
   hostRgCsrMatrixBenchmark. setup( csrMatrix );
   hostRgCsrMatrixBenchmark. runBenchmark( refX, refB, verbose );
   hostRgCsrMatrixBenchmark. tearDown();
   if( logFileName )
   {
      if( hostRgCsrMatrixBenchmark. getBenchmarkWasSuccesful() )
      {
         tnlString bgColor( "#55FF55" );
         logFile << "             <td bgcolor=" << bgColor << "> " << hostRgCsrMatrixBenchmark. getArtificialZeroElements() << "</td>" << endl;
         logFile << "             <td bgcolor=" << bgColor << "> " << hostRgCsrMatrixBenchmark. getGflops() << "</td>" << endl;
         logFile << "             <td bgcolor=" << bgColor << "> " << hostRgCsrMatrixBenchmark. getGflops() / csrMatrixBenchmark. getGflops() << "</td>" << endl;
      }
      else
      {
         logFile << "             <td bgcolor=#FFFFFF> N/A </td>" << endl;
         logFile << "             <td bgcolor=#FFFFFF> N/A </td>" << endl;
         logFile << "             <td bgcolor=#FFFFFF> N/A </td>" << endl;
      }
   }
   tnlSpmvBenchmarkRgCSRMatrix< Real, tnlCuda, int > cudaRgCsrMatrixBenchmark;
   cudaRgCsrMatrixBenchmark. setGroupSize( 16 );
   cudaRgCsrMatrixBenchmark. setUseAdaptiveGroupSize( true );
   cudaRgCsrMatrixBenchmark. setAdaptiveGroupSizeStrategy( tnlAdaptiveGroupSizeStrategyByAverageRowSize );
   cudaRgCsrMatrixBenchmark. setup( csrMatrix );
   for( int cudaBlockSize = 32; cudaBlockSize <= 256; cudaBlockSize *= 2 )
   {
      cudaRgCsrMatrixBenchmark. setCudaBlockSize( cudaBlockSize );
      cudaRgCsrMatrixBenchmark. runBenchmark( cudaX, refB, verbose );
      if( logFileName )
      {
         if( cudaRgCsrMatrixBenchmark. getBenchmarkWasSuccesful() )
         {
            logFile << "             <td> " << cudaRgCsrMatrixBenchmark. getGflops() << "</td>" << endl;
            double speedUp = cudaRgCsrMatrixBenchmark. getGflops() / csrMatrixBenchmark. getGflops();
            tnlString bgColor = getBgColorBySpeedUp( speedUp );
            logFile << "             <td bgcolor=" << bgColor << "> " << speedUp << "</td>" << endl;

         }
         else
         {
            logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
            logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
         }
      }
   }
   cudaRgCsrMatrixBenchmark. tearDown();


   /****
    * Row-Grouped CSR format with reordered rows
    * The rows are now sorted decreasingly by the number of the nonzero elements
    */
   tnlLongVector< int, tnlHost > rowPermutation( "rowPermutation" );
   {
      tnlCSRMatrix< Real, tnlHost > orderedCsrMatrix( "orderedCsrMatrix" );
      csrMatrix. sortRowsDecreasingly( rowPermutation );

      /****
       * Check if the ordering is OK.
       */
      int rowSize = csrMatrix. getNonzeroElementsInRow( rowPermutation[ 0 ] );
      bool rowSortingError = false;
      for( int i = 1; i < csrMatrix. getSize(); i ++ )
      {
         //cout << csrMatrix. getNonzeroElementsInRow( rowPermutation[ i ] ) << endl;
         if( rowSize < csrMatrix. getNonzeroElementsInRow( rowPermutation[ i ] ) )
         {
            cerr << "The rows are not sorted properly. Error is at row number " << i << endl;
            rowSortingError = true;
         }
         rowSize = csrMatrix. getNonzeroElementsInRow( rowPermutation[ i ] );
      }
      orderedCsrMatrix. reorderRows( rowPermutation, csrMatrix );
      orderedCsrMatrix. vectorProduct( refX, refB );
      benchmarkRgCSRFormat( orderedCsrMatrix,
                            refX,
                            cudaX,
                            refB,
                            false,
                            tnlAdaptiveGroupSizeStrategyByAverageRowSize,
                            csrMatrixBenchmark,
                            verbose,
                            logFileName,
                            logFile );

      tnlSpmvBenchmarkRgCSRMatrix< Real, tnlHost, int > hostRgCsrMatrixBenchmark;
      hostRgCsrMatrixBenchmark. setGroupSize( 16 );
      hostRgCsrMatrixBenchmark. setUseAdaptiveGroupSize( true );
      hostRgCsrMatrixBenchmark. setAdaptiveGroupSizeStrategy( tnlAdaptiveGroupSizeStrategyByFirstGroup );
      hostRgCsrMatrixBenchmark. setup( orderedCsrMatrix );
      hostRgCsrMatrixBenchmark. runBenchmark( refX, refB, verbose );
      hostRgCsrMatrixBenchmark. tearDown();
      if( logFileName )
      {
         if( hostRgCsrMatrixBenchmark. getBenchmarkWasSuccesful() )
         {
            tnlString bgColor( "#55FF55" );
            logFile << "             <td bgcolor=" << bgColor << "> " << hostRgCsrMatrixBenchmark. getArtificialZeroElements() << "</td>" << endl;
            logFile << "             <td bgcolor=" << bgColor << "> " << hostRgCsrMatrixBenchmark. getGflops() << "</td>" << endl;
            logFile << "             <td bgcolor=" << bgColor << "> " << hostRgCsrMatrixBenchmark. getGflops() / csrMatrixBenchmark. getGflops() << "</td>" << endl;
         }
         else
         {
            logFile << "             <td bgcolor=#FFFFFF> N/A </td>" << endl;
            logFile << "             <td bgcolor=#FFFFFF> N/A </td>" << endl;
            logFile << "             <td bgcolor=#FFFFFF> N/A </td>" << endl;
         }
      }
      tnlSpmvBenchmarkRgCSRMatrix< Real, tnlCuda, int > cudaRgCsrMatrixBenchmark;
      cudaRgCsrMatrixBenchmark. setGroupSize( 16 );
      cudaRgCsrMatrixBenchmark. setUseAdaptiveGroupSize( true );
      cudaRgCsrMatrixBenchmark. setAdaptiveGroupSizeStrategy( tnlAdaptiveGroupSizeStrategyByFirstGroup );
      cudaRgCsrMatrixBenchmark. setup( orderedCsrMatrix );
      for( int cudaBlockSize = 32; cudaBlockSize <= 256; cudaBlockSize *= 2 )
      {
         cudaRgCsrMatrixBenchmark. setCudaBlockSize( cudaBlockSize );
         cudaRgCsrMatrixBenchmark. runBenchmark( cudaX, refB, verbose );
         if( logFileName )
         {
            if( cudaRgCsrMatrixBenchmark. getBenchmarkWasSuccesful() )
            {
               logFile << "             <td> " << cudaRgCsrMatrixBenchmark. getGflops() << "</td>" << endl;
               double speedUp = cudaRgCsrMatrixBenchmark. getGflops() / csrMatrixBenchmark. getGflops();
               tnlString bgColor = getBgColorBySpeedUp( speedUp );
               logFile << "             <td bgcolor=" << bgColor << "> " << speedUp << "</td>" << endl;

            }
            else
            {
               logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
               logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
            }
         }
      }
      cudaRgCsrMatrixBenchmark. tearDown();
   }
   csrMatrix. vectorProduct( refX, refB );

   /****
    * Adaptive Row-Grouped CSR format
    */

   for( int desiredChunkSize = 1; desiredChunkSize <= 32; desiredChunkSize *= 2 )
   {

      tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, tnlHost, int > hostArgCsrMatrixBenchmark;
      hostArgCsrMatrixBenchmark. setDesiredChunkSize( desiredChunkSize );
      hostArgCsrMatrixBenchmark. setCudaBlockSize( 32 );
      hostArgCsrMatrixBenchmark. setup( csrMatrix );
      hostArgCsrMatrixBenchmark. runBenchmark( refX, refB, verbose );
      hostArgCsrMatrixBenchmark. tearDown();

      if( logFileName )
      {
         if( hostArgCsrMatrixBenchmark. getBenchmarkWasSuccesful() )
         {
            tnlString bgColor( "#55FF55" );
            switch( desiredChunkSize )
            {
               case 1: bgColor = "#5555FF"; break;
               case 2: bgColor = "#6666FF"; break;
               case 4: bgColor = "#7777FF"; break;
               case 8: bgColor = "#8888FF"; break;
               case 16: bgColor = "#9999FF"; break;
               case 32: bgColor = "#AAAAFF"; break;
               default: bgColor = "#FFFFFF";
            }
            logFile << "             <td bgcolor=" << bgColor << "> " << hostArgCsrMatrixBenchmark. getArtificialZeroElements() << "</td>" << endl;
            logFile << "             <td bgcolor=" << bgColor << "> " << hostArgCsrMatrixBenchmark. getGflops() << "</td>" << endl;
            logFile << "             <td bgcolor=" << bgColor << "> " << hostArgCsrMatrixBenchmark. getGflops() / csrMatrixBenchmark. getGflops() << "</td>" << endl;
         }
         else
         {
            logFile << "             <td bgcolor=#FFFFFF> N/A </td>" << endl;
            logFile << "             <td bgcolor=#FFFFFF> N/A </td>" << endl;
            logFile << "             <td bgcolor=#FFFFFF> N/A </td>" << endl;
         }
      }

      tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, tnlCuda, int > cudaArgCsrMatrixBenchmark;
      cudaArgCsrMatrixBenchmark. setDesiredChunkSize( desiredChunkSize );
      for( int cudaBlockSize = 32; cudaBlockSize <= 256; cudaBlockSize *= 2 )
      {
         cudaArgCsrMatrixBenchmark. setCudaBlockSize( cudaBlockSize );
         cudaArgCsrMatrixBenchmark. setup( csrMatrix );
         cudaArgCsrMatrixBenchmark. runBenchmark( cudaX, refB, verbose );
         if( logFileName )
         {
            if( cudaArgCsrMatrixBenchmark. getBenchmarkWasSuccesful() )
            {
               logFile << "             <td> " << cudaArgCsrMatrixBenchmark. getGflops() << "</td>" << endl;
               double speedUp = cudaArgCsrMatrixBenchmark. getGflops() / csrMatrixBenchmark. getGflops();
               tnlString bgColor = getBgColorBySpeedUp( speedUp );
               logFile << "             <td bgcolor=" << bgColor << "> " << speedUp << "</td>" << endl;

            }
            else
            {
               logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
               logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
            }
         }
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

#ifdef UNDEF
   /*
    * Benchmark of the Fast CSR format.
    */
   {
      if( verbose )
         cout << left << setw( 30 ) << "Fast CSR " << flush;

      tnlFastCSRMatrix< Real > fast_csrMatrix( "fast-csr-matrix" );

      if( fast_csrMatrix. copyFrom( csrMatrix ) )
      {
         benchmarkStatistics. fast_csr_compression = 100.0 * ( 1.0 -  ( double ) fast_csrMatrix. getColumnSequencesLength() / ( double ) fast_csrMatrix. getNonzeroElements() );

         time = stop_time;
         benchmarkSpMV< Real, tnlHost >( fast_csrMatrix,
                                         host_x,
                                         nonzero_elements,
                                         host_b,
                                         time,
                                         benchmarkStatistics. spmv_fast_csr_gflops,
                                         benchmarkStatistics. spmv_fast_csr_iter );
         if( verbose )
            cout << right << setw( 12 ) << setprecision( 2 ) << time
                 << right << setw( 15 ) << benchmarkStatistics. spmv_fast_csr_iter
                 << right << setw( 12 ) << setprecision( 2 ) << benchmarkStatistics. spmv_fast_csr_gflops << flush;

         if( refB != host_b )
         {
            if( verbose )
               cout << right << setw( 12 ) << "FAILED." << endl;
            Real max_err( 0.0 );
            for( int i = 0; i < size; i ++ )
               max_err = Max( max_err, ( Real ) fabs( host_b[ i ] - refB[ i ] ) );
            if( verbose )
               cout << left << setw( 12 ) <<  "  Max. err. is " << max_err << endl;
            benchmarkStatistics. spmv_fast_csr_gflops = -1.0;
            return false;
         }
         if( verbose )
            cout << left << setw( 12 ) << "  OK."
                 << right << setw( 14 ) << "Compression: " << benchmarkStatistics. fast_csr_compression << "%" << endl;
      }
      else
         if( verbose )
            cout << "Format transfer failed!!!" << endl;
      /*
       * Benchmark Coalesced Fast CSR format.
       */
      block_iter = 0;
      for( int block_size = 16; block_size < 64; block_size *= 2 )
      {
         if( verbose )
            cout << left << setw( 25 ) << "Colesced Fast CSR " << setw( 5 ) << block_size << flush;

         tnlFastRgCSRMatrix< Real > coa_fast_csrMatrix( "coa_fast-csr-matrix", block_size );

         if( coa_fast_csrMatrix. copyFrom( fast_csrMatrix ) )
         {
            //coa_fast_csr_compression = 100.0 * ( 1.0 -  ( double ) coa_fast_csrMatrix. getColumnSequencesLength() / ( double ) coa_fast_csrMatrix. getNonzeroElements() );
            benchmarkStatistics. coa_fast_csr_max_cs_dict_size[ block_iter ] = coa_fast_csrMatrix. getMaxColumnSequenceDictionarySize();
            time = stop_time;
            benchmarkSpMV< Real, tnlHost >( coa_fast_csrMatrix,
                                            host_x,
                                            nonzero_elements,
                                            host_b,
                                            time,
                                            benchmarkStatistics. spmv_coa_fast_csr_gflops[ block_iter ],
                                            benchmarkStatistics. spmv_coa_fast_csr_iter[ block_iter ] );
            if( verbose )
               cout << right << setw( 12 ) << setprecision( 2 ) << time
                    << right << setw( 15 ) << benchmarkStatistics. spmv_coa_fast_csr_iter[ block_iter ]
                    << right << setw( 12 ) << setprecision( 2 ) << benchmarkStatistics. spmv_coa_fast_csr_gflops[ block_iter ] << flush;

            if( refB != host_b )
            {
               if( verbose )
                  cout << right << setw( 12 ) << "FAILED." << endl;
               benchmarkStatistics. spmv_coa_fast_csr_gflops[ block_iter ] = -1.0;
               return false;
            }
            if( verbose )
               cout << left << setw( 12 ) << "  OK." << endl;
         }
         else
         {
            benchmarkStatistics. coa_fast_csr_max_cs_dict_size[ block_iter ] = coa_fast_csrMatrix. getMaxColumnSequenceDictionarySize();
            if( verbose )
               cout << "Format transfer failed!!!" << endl;
            continue;
         }

#ifdef HAVE_CUDA
         /*
          * Benchmark Coalesced Fast CSR format on the CUDA device.
          */

         if( verbose )
            cout << left << setw( 25 ) << "Coalesced Fast CSR CUDA" << setw( 5 ) << block_size << flush;

         tnlFastRgCSRMatrix< Real, tnlCuda > cuda_coa_fast_csrMatrix( "cuda-coa-fast-csr-matrix" );

         if( cuda_coa_fast_csrMatrix. copyFrom( coa_fast_csrMatrix ) )
         {
            time = stop_time;
            cuda_b. setValue( -1.0 );
            benchmarkSpMV< Real, tnlCuda >( cuda_coa_fast_csrMatrix,
                                            cuda_x,
                                            nonzero_elements,
                                            cuda_b,
                                            time,
                                            benchmarkStatistics. spmv_cuda_coa_fast_csr_gflops[ block_iter ],
                                            benchmarkStatistics. spmv_cuda_coa_fast_csr_iter[ block_iter ] );

            if( verbose )
               cout << right << setw( 12 ) << setprecision( 2 ) << time
                    << right << setw( 15 ) << benchmarkStatistics. spmv_cuda_coa_fast_csr_iter[ block_iter ]
                    << right << setw( 12 ) << setprecision( 2 ) << benchmarkStatistics. spmv_cuda_coa_fast_csr_gflops[ block_iter ] << endl;

            //if( refB != cuda_b )
				// {
				//	if( verbose )
				//	   cout << right << setw( 12 ) << "FAILED." << endl;
				//	//spmv_cuda_coa_fast_csr_gflops[ block_iter ] = -1.0;
				//	return false;
				// }
				// else
				// 	if( verbose )
				//		cout << right << setw( 12 ) << "OK." << endl;
         }
         else
            if( verbose )
               cout << "Format transfer failed!!!" << endl;
#endif
         block_iter ++;
      }
   }



   /*
	 * Benchmarks of the ELLPACK format.
	 */
	/*{
      if( verbose )
         cout << "Benchmarking ELLPACK format ... " << flush;

      int max_row_length, min_row_length, average_row_length;
      csrMatrix. getRowStatistics( min_row_length,
                                    max_row_length,
                                    average_row_length );
      double alpha= 1.0;
      int ellpack_row_length = ( 1.0 - alpha ) * average_row_length +
                               alpha * max_row_length;
      tnlEllpackMatrix< Real, tnlHost > ellpack_matrix( "ellpack-matrix", ellpack_row_length );
      ellpack_artificial_zeros = 100.0 * ( double ) ellpack_matrix. getArtificialZeroElements() / ( double ) ellpack_matrix. getNonzeroElements();
      ellpack_matrix. copyFrom( csrMatrix );
      if( verbose )
           cout << "Min row length = " << min_row_length << endl
                << "Max row length = " << max_row_length << endl
                << "Average row length = " << average_row_length << endl
                << "Ellpack row length = " << ellpack_row_length << endl
                << "COO elements = " << ellpack_matrix. getCOONonzeroElements() << endl;
      time = stop_time;
      host_x. setValue( 1.0 );
      host_b. setValue( 0.0 );
      benchmarkSpMV< Real, tnlHost >( ellpack_matrix,
                                      host_x,
                                      host_b,
                                      time,
                                      spmv_ellpack_gflops,
                                      spmv_ellpack_iter );

      if( verbose )
         cout << time << " sec. " << spmv_ellpack_iter << " iterations " << spmv_ellpack_gflops << " GFLOPS." << endl;
      if( verbose )
         cout << "Comparing results ... ";
      if( refB != host_b )
      {
         if( verbose )
            cout << "FAILED." << endl;
         return false;
      }
      if( verbose )
         cout << "OK." << endl;

	}*/

   if( verbose )
      cout << setfill( '-' ) << setw( 95 ) << "--" << endl
           << setfill( ' ');
   return true;           

}
#endif

#endif /* SPARSEMATRIXBENCHMARK_H_ */
