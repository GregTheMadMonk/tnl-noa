 /***************************************************************************
                          tnlSpmvBenchmarkRgCSRMatrix.h  -  description
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

#ifndef TNLSPMVBENCHMARKRGCSRMATRIX_H_
#define TNLSPMVBENCHMARKRGCSRMATRIX_H_

#include "tnlSpmvBenchmark.h"
#include <matrix/tnlRgCSRMatrix.h>

template< typename Real, tnlDevice Device, typename Index>
class tnlSpmvBenchmarkRgCSRMatrix : public tnlSpmvBenchmark< Real, Device, Index, tnlRgCSRMatrix >
{
   public:

   tnlSpmvBenchmarkRgCSRMatrix();

   bool setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix );

   void tearDown();

   void writeProgress() const;

   void writeToLogTable( ostream& logFile,
                         const double& csrGflops,
                         const tnlString& inputMtxFile,
                         const tnlCSRMatrix< Real, tnlHost, Index >& csrMatrix,
                         bool writeMatrixInfo ) const;

   void setGroupSize( const Index groupSize );

   void setCudaBlockSize( const Index cudaBlockSize );

   void setUseAdaptiveGroupSize( bool useAdaptiveGroupSize );

   void setAdaptiveGroupSizeStrategy( tnlAdaptiveGroupSizeStrategy adaptiveGroupSizeStrategy );

   Index getArtificialZeroElements() const;

   protected:

   Index groupSize;

   Index cudaBlockSize;

   bool useAdaptiveGroupSize;

   tnlAdaptiveGroupSizeStrategy adaptiveGroupSizeStrategy;
};

template< typename Real,
          tnlDevice Device,
          typename Index>
tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: tnlSpmvBenchmarkRgCSRMatrix()
 : groupSize( 0 ),
   cudaBlockSize( 0 ),
   useAdaptiveGroupSize( false ),
   adaptiveGroupSizeStrategy( tnlAdaptiveGroupSizeStrategyByAverageRowSize )
{
}

template< typename Real,
          tnlDevice Device,
          typename Index>
bool tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: setup( const tnlCSRMatrix< Real, tnlHost, Index >& csrMatrix )
{
   tnlAssert( this -> groupSize > 0, cerr << "groupSize = " << this -> groupSize );
   if( Device == tnlHost )
   {
      this -> matrix. tuneFormat( groupSize,
                                  this -> useAdaptiveGroupSize,
                                  this -> adaptiveGroupSizeStrategy );
      if( ! this -> matrix. copyFrom( csrMatrix ) )
         return false;
   }
   if( Device == tnlCuda )
   {
#ifdef HAVE_CUDA
      tnlRgCSRMatrix< Real, tnlHost, Index > hostMatrix( "tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: setup : hostMatrix" );
      hostMatrix. tuneFormat( groupSize,
                              this -> useAdaptiveGroupSize,
                              this -> adaptiveGroupSizeStrategy );
      hostMatrix. copyFrom( csrMatrix );
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
          tnlDevice Device,
          typename Index>
void tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: tearDown()
{
   this -> matrix. reset();
}

template< typename Real,
          tnlDevice Device,
          typename Index >
void tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: writeProgress() const
{
   cout << left << setw( this -> formatColumnWidth - 15 ) << "Row-grouped CSR ";
   if( Device == tnlCuda )
   {
      if( useAdaptiveGroupSize )
         cout << setw( 5 ) << "Var.";
      else
         cout << setw( 5 ) << this -> groupSize;
      cout << setw( 10 ) << this -> cudaBlockSize;
   }
   else
   {
      if( useAdaptiveGroupSize )
         cout << setw( 15 ) << "Var.";
      else
         cout << setw( 15 ) << this -> groupSize;
   }
   cout << right << setw( this -> timeColumnWidth ) << setprecision( 2 ) << this -> getTime()
        << right << setw( this -> iterationsColumnWidth ) << this -> getIterations()
        << right << setw( this -> gflopsColumnWidth ) << setprecision( 2 ) << this -> getGflops();
   if( this -> getBenchmarkWasSuccesful() )
        cout << right << setw( this -> benchmarkStatusColumnWidth ) << "  OK  - maxError is " << this -> maxError << ". ";
   else
        cout << right << setw( this -> benchmarkStatusColumnWidth ) << "  FAILED - maxError is " << this -> maxError << ". ";
#ifndef HAVE_CUDA
   if( Device == tnlCuda )
      cout << "CUDA support is missing.";
#endif
      cout << endl;
}

template< typename Real,
          tnlDevice Device,
          typename Index >
void tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: setGroupSize( const Index groupSize )
{
   this -> groupSize = groupSize;
}

template< typename Real,
          tnlDevice Device,
          typename Index >
void tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: setCudaBlockSize( const Index cudaBlockSize )
{
   this -> matrix. setCUDABlockSize( cudaBlockSize );
   this -> cudaBlockSize = cudaBlockSize;
}

template< typename Real,
          tnlDevice Device,
          typename Index >
void tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: setUseAdaptiveGroupSize( bool useAdaptiveGroupSize )
{
   this -> useAdaptiveGroupSize = useAdaptiveGroupSize;
}

template< typename Real,
          tnlDevice Device,
          typename Index >
void tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: setAdaptiveGroupSizeStrategy( tnlAdaptiveGroupSizeStrategy adaptiveGroupSizeStrategy )
{
   this -> adaptiveGroupSizeStrategy = adaptiveGroupSizeStrategy;
}

template< typename Real,
          tnlDevice Device,
          typename Index >
Index tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: getArtificialZeroElements() const
{
   return this -> matrix. getArtificialZeroElements();
}

template< typename Real,
          tnlDevice Device,
          typename Index >
void tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: writeToLogTable( ostream& logFile,
                                                                            const double& csrGflops,
                                                                            const tnlString& inputMtxFile,
                                                                            const tnlCSRMatrix< Real, tnlHost, Index >& csrMatrix,
                                                                            bool writeMatrixInfo ) const
{
   tnlString bgColor;
   switch( groupSize )
   {
      case 16: bgColor = "#5555FF"; break;
      case 32: bgColor = "#9999FF"; break;
      case 64: bgColor = "#CCCCFF"; break;
      default: bgColor = "#FFFFFF";
   }
   if( writeMatrixInfo )
   {
      tnlString baseFileName( inputMtxFile );
      baseFileName += tnlString( ".rgcsr-");
      baseFileName += tnlString( groupSize );
      tnlString matrixPdfFile( baseFileName );
      matrixPdfFile += tnlString( ".pdf" );
      tnlString matrixHtmlFile( baseFileName );
      matrixHtmlFile += tnlString( ".html" );
      tnlRgCSRMatrix< Real > rgCsrMatrix( inputMtxFile );
      rgCsrMatrix. tuneFormat( this -> groupSize,
                               this -> useAdaptiveGroupSize,
                               this -> adaptiveGroupSizeStrategy );
      rgCsrMatrix. copyFrom( csrMatrix );
      this -> printMatrixInHtml( matrixHtmlFile, rgCsrMatrix );
      logFile << "             <td bgcolor=" << bgColor << "> <a href=\"" << matrixPdfFile << "\">PDF</a>,<a href=\"" << matrixHtmlFile << "\"> HTML</a></td>" << endl;
      logFile << "             <td bgcolor=" << bgColor << "> " << this -> getArtificialZeroElements() << "</td>" << endl;
   }
   if( this -> getBenchmarkWasSuccesful() )
   {
      const double speedUp = this -> getGflops() / csrGflops;
      bgColor =  this -> getBgColorBySpeedUp( speedUp );
      logFile << "             <td bgcolor=" << bgColor << ">" << this -> getTime() << "</td>" << endl;
      logFile << "             <td bgcolor=" << bgColor << "> " << this -> getGflops() << "</td>" << endl;
      logFile << "             <td bgcolor=" << bgColor << "> " << speedUp << "</td>" << endl;
   }
   else
   {
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
   }
}


#endif /* TNLSPMVBENCHMARKRGCSRMATRIX_H_ */
