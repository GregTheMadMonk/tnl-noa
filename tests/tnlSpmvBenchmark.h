/***************************************************************************
                          tnlSpmvBenchmark.h  -  description
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

#ifndef TNLSPMVBENCHMARK_H_
#define TNLSPMVBENCHMARK_H_

#include <matrix/tnlCSRMatrix.h>
#include <core/tnlTimerRT.h>
#include <core/mfuncs.h>


double tnlSpmvBenchmarkPrecision( const double& ) { return 1.0e-12; }
float tnlSpmvBenchmarkPrecision( const float& ) { return 1.0e-4; }


template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
class tnlSpmvBenchmark
{
   public:

   tnlSpmvBenchmark();

   virtual bool setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix ) = 0;

   virtual void tearDown() = 0;

   virtual void writeProgress() const = 0;

   /****
    * This is virtual only the purpose of testing external formats like
    * the Hybrid format from the CUSP library. This format is not wrapped
    * in tnlMatrix.
    */
   virtual void runBenchmark( const tnlLongVector< Real, Device, Index >& x,
                              const tnlLongVector< Real, tnlHost, Index >& refB,
                              bool verbose );

   bool getBenchmarkWasSuccesful() const;

   double getGflops() const;

   double getTime() const;

   void setMaxIterations( const int maxIterations );

   int getIterations() const;

   Index getArtificialZeros() const;

   Real getMaxError() const;

   void writeProgressTableHeader();

   virtual void writeToLogTable( ostream& logFile,
                                 const double& csrGflops,
                                 const tnlString& inputMtxFile,
                                 const tnlCSRMatrix< Real, tnlHost, Index >& csrMatrix,
                                 bool writeMatrixInfo  ) const = 0;


   protected:

   /****
    * This is helper method for generating HTML table with benchmark results
    */
   tnlString getBgColorBySpeedUp( const double& speedUp ) const;

   /****
    * Helper method for writing matrix statistics and information to HTML
    */
   bool printMatrixInHtml( const tnlString& fileName,
                           tnlMatrix< Real >& matrix ) const;


   bool benchmarkWasSuccesful;

   double gflops;

   double time;

   /****
    * Max number of SpMV repetitions.
    */
   int maxIterations;

   /****
    * Real number of repetitions.
    */
   int iterations;

   Index artificialZeros;

   Real maxError;

   Index firstErrorOccurence;

   Matrix< Real, Device, Index > matrix;

   /****
    * Parameters for the progress table columns
    */

   int formatColumnWidth;

   int timeColumnWidth;

   int iterationsColumnWidth;

   int gflopsColumnWidth;

   int benchmarkStatusColumnWidth;

   int infoColumnWidth;

};

template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
tnlSpmvBenchmark< Real, Device, Index, Matrix > :: tnlSpmvBenchmark()
   : benchmarkWasSuccesful( false ),
     gflops( 0.0 ),
     time( 0.0 ),
     maxIterations( 0 ),
     iterations( 0.0 ),
     artificialZeros( 0 ),
     maxError( 0.0 ),
     firstErrorOccurence( 0 ),
     matrix( "spmvBenchmark::matrix" ),
     formatColumnWidth( 40 ),
     timeColumnWidth( 12 ),
     iterationsColumnWidth( 15 ),
     gflopsColumnWidth( 12 ),
     benchmarkStatusColumnWidth( 12 ),
     infoColumnWidth( 20 )
{

}

template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
bool tnlSpmvBenchmark< Real, Device, Index, Matrix > :: getBenchmarkWasSuccesful() const
{
   return this -> benchmarkWasSuccesful;
}

template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
double tnlSpmvBenchmark< Real, Device, Index, Matrix > :: getGflops() const
{
   return this -> gflops;
}

template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
double tnlSpmvBenchmark< Real, Device, Index, Matrix > :: getTime() const
{
   return this -> time;
}

template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
void tnlSpmvBenchmark< Real, Device, Index, Matrix > :: setMaxIterations( const int maxIterations )
{
   this -> maxIterations = maxIterations;
}

template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
int tnlSpmvBenchmark< Real, Device, Index, Matrix > :: getIterations() const
{
   return this -> iterations;
}


template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
Index tnlSpmvBenchmark< Real, Device, Index, Matrix > :: getArtificialZeros() const
{
   return this -> artificialZeros;
}

template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
Real tnlSpmvBenchmark< Real, Device, Index, Matrix > :: getMaxError() const
{
   return this -> maxError;
}

template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
void tnlSpmvBenchmark< Real, Device, Index, Matrix > :: runBenchmark( const tnlLongVector< Real, Device, Index >& x,
                                                                      const tnlLongVector< Real, tnlHost, Index >& refB,
                                                                      bool verbose )
{
   benchmarkWasSuccesful = false;
#ifndef HAVE_CUDA
   if( Device == tnlCuda )
   {
      if( verbose )
         writeProgress();
      return;
   }
#endif

   tnlLongVector< Real, Device, Index > b( "tnlSpmvBenchmark< Real, Device, Index, Matrix > :: runBenchmark : b" );
   if( ! b. setSize( refB. getSize() ) )
      return;

   iterations = 0;

   tnlTimerRT rt_timer;
   maxIterations = 1;
   rt_timer. Reset();
   {
      for( int i = 0; i < maxIterations; i ++ )
      {
         matrix. vectorProduct( x, b );
         iterations ++;
      }
   }
   time = rt_timer. GetTime();

   firstErrorOccurence = 0;
   tnlLongVector< Real, tnlHost, Index > resB( "tnlSpmvBenchmark< Real, Device, Index, Matrix > :: runBenchmark : b" );
   if( ! resB. setSize( b. getSize() ) )
   {
      cerr << "I am not able to allocate copy of vector b on the host." << endl;
      return;
   }
   resB = b;
   benchmarkWasSuccesful = true;
   for( Index j = 0; j < refB. getSize(); j ++ )
   {
      //f << refB[ j ] << " - " << host_b[ j ] << " = "  << refB[ j ] - host_b[ j ] <<  endl;
      Real error( 0.0 );
      if( refB[ j ] != 0.0 )
         error = ( Real ) fabs( refB[ j ] - resB[ j ] ) /  ( Real ) fabs( refB[ j ] );
      else
         error = ( Real ) fabs( refB[ j ] );
      this -> maxError = Max( this -> maxError, error );

      if( error > tnlSpmvBenchmarkPrecision( error ) )
      {
         benchmarkWasSuccesful = false;
         firstErrorOccurence = j;
         //cerr << " b[ " << j << " ] = " << resB[ j ] << " while refB[ " << j << " ] = " << refB[ j ] << endl;
         //abort();
      }
   }
   cout << "First error was on " << firstErrorOccurence << endl;

   double flops = 2.0 * iterations * matrix. getNonzeroElements();
   gflops = flops / time * 1.0e-9;
   artificialZeros = matrix. getArtificialZeroElements();

   if( verbose )
      writeProgress();
}

template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
void tnlSpmvBenchmark< Real, Device, Index, Matrix > :: writeProgressTableHeader()
{
   int totalWidth = this -> formatColumnWidth +
                    this -> timeColumnWidth +
                    this -> iterationsColumnWidth +
                    this -> gflopsColumnWidth +
                    this -> benchmarkStatusColumnWidth +
                    this -> infoColumnWidth;

   cout << left << setw( this -> formatColumnWidth - 5 ) << "MATRIX FORMAT"
        << left << setw( 5 ) << "BLOCK"
        << right << setw( this -> timeColumnWidth ) << "TIME"
        << right << setw( this -> iterationsColumnWidth ) << "ITERATIONS"
        << right << setw( this -> gflopsColumnWidth ) << "GFLOPS"
        << right << setw( this -> benchmarkStatusColumnWidth ) << "CHECK"
        << left << setw(  this -> infoColumnWidth ) << " INFO" << endl
        << setfill( '-' ) << setw( totalWidth ) << "--" << endl
        << setfill( ' ');
}

template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
tnlString tnlSpmvBenchmark< Real, Device, Index, Matrix > :: getBgColorBySpeedUp( const double& speedUp ) const
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


template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
bool tnlSpmvBenchmark< Real, Device, Index, Matrix > :: printMatrixInHtml( const tnlString& fileName,
                                                                           tnlMatrix< Real >& matrix ) const
{
   //cout << "Writing to file " << fileName << endl;
   fstream file;
   file. open( fileName. getString(), ios :: out );
   if( ! file )
   {
      cerr << "I am not able to open the file " << fileName << endl;
      return false;
   }
   file << "<html>" << endl;
   file << "   <body>" << endl;
   matrix. printOut( file, "html" );
   file << "   </body>" << endl;
   file << "</html>" << endl;
   file. close();
   return true;
}

#endif /* TNLSPMVBENCHMARK_H_ */
