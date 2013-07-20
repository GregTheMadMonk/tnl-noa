/***************************************************************************
                          tnlSpmvBenchmarkAdaptiveRgCSRMatrix.h  -  description
                             -------------------
    begin                : May 15, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
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

#ifndef TNLSPMVBENCHMARKADAPTIVERGCSRMATRIX_H_
#define TNLSPMVBENCHMARKADAPTIVERGCSRMATRIX_H_

#include "tnlSpmvBenchmark.h"
#include <matrix/tnlAdaptiveRgCSRMatrix.h>
#include <core/tnlAssert.h>

template< typename Real, typename Device, typename Index>
class tnlSpmvBenchmarkAdaptiveRgCSRMatrix : public tnlSpmvBenchmark< Real, Device, Index, tnlAdaptiveRgCSRMatrix >
{
   public:

   tnlSpmvBenchmarkAdaptiveRgCSRMatrix();

   bool setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix );

   void tearDown();

   void writeProgress() const;

   void writeToLogTable( ostream& logFile,
                         const double& csrGflops,
                         const tnlString& inputMtxFile,
                         const tnlCSRMatrix< Real, tnlHost, Index >& csrMatrix,
                         bool writeMatrixInfo  ) const;

   void setDesiredChunkSize( const Index desiredChunkSize );

   void setCudaBlockSize( const Index cudaBlockSize );

   Index getArtificialZeroElements() const;

   void setBestRgCSRGflops( const double& bestRgCSRGflops );

   protected:

   /****
    * This is helper method for generating HTML table with benchmark results
    */
    tnlString getBgColorByRgCSRSpeedUp( const double& speedUp ) const;

   Index desiredChunkSize;

   Index cudaBlockSize;

   bool useAdaptiveGroupSize;

   tnlAdaptiveGroupSizeStrategy adaptiveGroupSizeStrategy;

   double bestRgCSRGflops;
};

template< typename Real,
          typename Device,
          typename Index>
tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, Device, Index > :: tnlSpmvBenchmarkAdaptiveRgCSRMatrix()
 : desiredChunkSize( 4 ),
   cudaBlockSize( 32 ),
   useAdaptiveGroupSize( false ),
   adaptiveGroupSizeStrategy( tnlAdaptiveGroupSizeStrategyByAverageRowSize ),
   bestRgCSRGflops( 0.0 )

{
}

template< typename Real,
          typename Device,
          typename Index>
bool tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, Device, Index > :: setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix )
{
   //tnlAssert( this -> groupSize > 0, cerr << "groupSize = " << this -> groupSize );
   if( Device :: getDevice() == tnlHostDevice )
   {
      this -> matrix. tuneFormat( desiredChunkSize, cudaBlockSize );
      if( ! this -> matrix. copyFrom( matrix ) )
         return false;
      //matrix. printOut( cout, "text", 30 );
      //this -> matrix. printOut( cout, "text", 30 );
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      tnlAdaptiveRgCSRMatrix< Real, tnlHost, Index > hostMatrix( "tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, Device, Index > :: setup : hostMatrix" );
      hostMatrix. tuneFormat( desiredChunkSize, cudaBlockSize );
      hostMatrix. copyFrom( matrix );
      if( ! this -> matrix. copyFrom( hostMatrix ) )
         return false;
#else
      return false;
#endif
   }
   this -> setupOk = true;
   return true;
}

template< typename Real,
          typename Device,
          typename Index>
void tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, Device, Index > :: tearDown()
{
   //this -> matrix. setSize( 0 );
   //this -> matrix. setNonzeroElements( 0 );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, Device, Index > :: writeProgress() const
{
   cout << left << setw( this -> formatColumnWidth - 15 ) << "Adap. Row-grouped CSR ";
   if( Device :: getDevice() == tnlCudaDevice )
      cout << setw( 5 ) << this -> desiredChunkSize
           << setw( 10 ) << this -> cudaBlockSize;
   else
      cout << setw( 15 ) << this -> desiredChunkSize;
   cout << right << setw( this -> timeColumnWidth ) << setprecision( 2 ) << this -> getTime()
        << right << setw( this -> iterationsColumnWidth ) << this -> getIterations()
        << right << setw( this -> gflopsColumnWidth ) << setprecision( 2 ) << this -> getGflops();
   if( this -> getBenchmarkWasSuccesful() )
        cout << right << setw( this -> benchmarkStatusColumnWidth ) << " OK - maxError is " << this -> maxError << ". ";
   else
        cout << right << setw( this -> benchmarkStatusColumnWidth ) << "  FAILED";
#ifndef HAVE_CUDA
   if( Device :: getDevice() == tnlCudaDevice )
      tnlCudaSupportMissingMessage;;
#endif
      cout << endl;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, Device, Index > :: writeToLogTable( ostream& logFile,
                                                                                    const double& csrGflops,
                                                                                    const tnlString& inputMtxFile,
                                                                                    const tnlCSRMatrix< Real, tnlHost, Index >& csrMatrix,
                                                                                    bool writeMatrixInfo  ) const
{
   if( this -> getBenchmarkWasSuccesful() )
   {
      tnlString bgColor="#FFFFFF";
      double speedUp = this -> getGflops() / csrGflops;
      double rgCsrSpeedUp( 0.0 );
      if( this -> bestRgCSRGflops )
         rgCsrSpeedUp = this -> getGflops() / this -> bestRgCSRGflops;
      switch( desiredChunkSize )
      {
         case 1: bgColor = "#666666"; break;
         case 2: bgColor = "#777777"; break;
         case 4: bgColor = "#888888"; break;
         case 8: bgColor = "#999999"; break;
         case 16: bgColor = "#AAAAAA"; break;
         case 32: bgColor = "#BBBBBB"; break;
         default: bgColor = "#FFFFFF";
      }
      if( writeMatrixInfo )
      {
         tnlString baseFileName( inputMtxFile );
         baseFileName += tnlString( ".argcsr-");
         baseFileName += tnlString( desiredChunkSize );
         baseFileName += tnlString( "-" );
         baseFileName += tnlString( cudaBlockSize );
         tnlString matrixPdfFile = baseFileName + tnlString( ".pdf" );
         tnlString matrixHtmlFile = baseFileName + tnlString( ".html" );
         tnlAdaptiveRgCSRMatrix< Real > argCsrMatrix( inputMtxFile );
         argCsrMatrix. tuneFormat( this -> desiredChunkSize,
                                 this -> cudaBlockSize );
         argCsrMatrix. copyFrom( csrMatrix );
         this -> printMatrixInHtml( matrixHtmlFile, argCsrMatrix );
         if( rgCsrSpeedUp > 1.0 )
            bgColor=getBgColorByRgCSRSpeedUp( rgCsrSpeedUp );
         logFile << "             <td bgcolor=" << bgColor << "> <a href=\"" << matrixPdfFile << "\">PDF</a>, <a href=\"" << matrixHtmlFile << "\">HTML</a></td> " << endl;
         logFile << "             <td bgcolor=" << bgColor << "> " << this -> getArtificialZeroElements() << "</td>" << endl;
      }

      bgColor = this -> getBgColorBySpeedUp( speedUp );
      tnlString textColor = "#000000"; //getBgColorByRgCSRSpeedUp( rgCsrSpeedUp );
      logFile << "             <td bgcolor=" << bgColor << "><font size=3 color=\"" << textColor << "\"> " << this -> getTime() << "</font></td>" << endl;
      logFile << "             <td bgcolor=" << bgColor << "><font size=3 color=\"" << textColor << "\"> " << this -> getGflops() << "</font></td>" << endl;
      logFile << "             <td bgcolor=" << bgColor << "><font size=3 color=\"" << textColor << "\"> " << speedUp << "</font></td>" << endl;

   }
   else
   {
      if( writeMatrixInfo )
      {
         logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
         logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
      }
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
   }
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, Device, Index > :: setDesiredChunkSize( const Index desiredChunkSize )
{
   this -> desiredChunkSize = desiredChunkSize;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, Device, Index > :: setCudaBlockSize( const Index cudaBlockSize )
{
   this -> cudaBlockSize = cudaBlockSize;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, Device, Index > :: getArtificialZeroElements() const
{
   return this -> matrix. getArtificialZeroElements();
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, Device, Index > :: setBestRgCSRGflops( const double& bestRgCSRGflops )
{
   this -> bestRgCSRGflops = bestRgCSRGflops;
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, Device, Index > :: getBgColorByRgCSRSpeedUp( const double& speedUp ) const
{
   if( speedUp >= 30.0 )
      return tnlString( "#009900" );
   if( speedUp >= 25.0 )
      return tnlString( "#00AA00" );
   if( speedUp >= 20.0 )
      return tnlString( "#00BB00" );
   if( speedUp >= 15.0 )
      return tnlString( "#00CC00" );
   if( speedUp >= 10.0 )
      return tnlString( "#00DD00" );
   if( speedUp >= 5.0 )
      return tnlString( "#00EE00" );
   if( speedUp >= 1.0 )
      return tnlString( "#00FF00" );
   return tnlString( "#FFFFFF" );
}

#endif /* TNLSPMVBENCHMARKADAPTIVERGCSRMATRIX_H_ */
