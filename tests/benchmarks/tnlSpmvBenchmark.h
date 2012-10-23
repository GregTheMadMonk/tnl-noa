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
          typename Device,
          typename Index,
          template< typename matrixReal, typename matrixDevice, typename matrixIndex > class Matrix >
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
   virtual void runBenchmark( const tnlVector< Real, Device, Index >& x,
                              const tnlVector< Real, tnlHost, Index >& refB,
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

   /*!***
    * This method test if the matrix is stored properly usually against full or CSR matrix.
    * It is useful test for more complicated formats. Matrices stored on CUDA device are
    * tested by SpMV with complete basis made of vectors e_0, \ldots e_{N-1}.
    */
   virtual bool testMatrix( const tnlMatrix< Real, tnlHost, Index >& testMatrix,
                            int verbose ) const;

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

   bool setupOk;

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
          typename Device,
          typename Index,
          template< typename matrixReal, typename matrixDevice, typename matrixIndex > class Matrix >
tnlSpmvBenchmark< Real, Device, Index, Matrix > :: tnlSpmvBenchmark()
   : benchmarkWasSuccesful( false ),
     setupOk( false ),
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
          typename Device,
          typename Index,
          template< typename matrixReal, typename matrixDevice, typename matrixIndex > class Matrix >
bool tnlSpmvBenchmark< Real, Device, Index, Matrix > :: getBenchmarkWasSuccesful() const
{
   return this -> benchmarkWasSuccesful;
}

template< typename Real,
          typename Device,
          typename Index,
          template< typename matrixReal, typename matrixDevice, typename matrixIndex > class Matrix >
double tnlSpmvBenchmark< Real, Device, Index, Matrix > :: getGflops() const
{
   return this -> gflops;
}

template< typename Real,
          typename Device,
          typename Index,
          template< typename matrixReal, typename matrixDevice, typename matrixIndex > class Matrix >
double tnlSpmvBenchmark< Real, Device, Index, Matrix > :: getTime() const
{
   return this -> time;
}

template< typename Real,
          typename Device,
          typename Index,
          template< typename matrixReal, typename matrixDevice, typename matrixIndex > class Matrix >
void tnlSpmvBenchmark< Real, Device, Index, Matrix > :: setMaxIterations( const int maxIterations )
{
   this -> maxIterations = maxIterations;
}

template< typename Real,
          typename Device,
          typename Index,
          template< typename matrixReal, typename matrixDevice, typename matrixIndex > class Matrix >
int tnlSpmvBenchmark< Real, Device, Index, Matrix > :: getIterations() const
{
   return this -> iterations;
}


template< typename Real,
          typename Device,
          typename Index,
          template< typename matrixReal, typename matrixDevice, typename matrixIndex > class Matrix >
Index tnlSpmvBenchmark< Real, Device, Index, Matrix > :: getArtificialZeros() const
{
   return this -> artificialZeros;
}

template< typename Real,
          typename Device,
          typename Index,
          template< typename matrixReal, typename matrixDevice, typename matrixIndex > class Matrix >
Real tnlSpmvBenchmark< Real, Device, Index, Matrix > :: getMaxError() const
{
   return this -> maxError;
}

template< typename Real,
          typename Device,
          typename Index,
          template< typename matrixReal, typename matrixDevice, typename matrixIndex > class Matrix >
void tnlSpmvBenchmark< Real, Device, Index, Matrix > :: runBenchmark( const tnlVector< Real, Device, Index >& x,
                                                                      const tnlVector< Real, tnlHost, Index >& refB,
                                                                      bool verbose )
{
   benchmarkWasSuccesful = false;
   if( ! setupOk )
      return;
#ifndef HAVE_CUDA
   if( Device :: getDevice() == tnlCudaDevice )
   {
      if( verbose )
         writeProgress();
      return;
   }
#endif

   tnlVector< Real, Device, Index > b( "tnlSpmvBenchmark< Real, Device, Index, Matrix > :: runBenchmark : b" );
   if( ! b. setSize( refB. getSize() ) )
      return;

   iterations = 0;

   tnlTimerRT rt_timer;
   rt_timer. Reset();
   //maxIterations = 1;

   for( int i = 0; i < maxIterations; i ++ )
   {
      matrix. vectorProduct( x, b );
      iterations ++;
   }

   this -> time = rt_timer. GetTime();

   firstErrorOccurence = 0;
   tnlVector< Real, tnlHost, Index > resB( "tnlSpmvBenchmark< Real, Device, Index, Matrix > :: runBenchmark : b" );
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
      if( error > maxError )
         firstErrorOccurence = j;
      this -> maxError = Max( this -> maxError, error );

      /*if( error > tnlSpmvBenchmarkPrecision( error ) )
         benchmarkWasSuccesful = false;*/

   }
   //cout << "First error was on " << firstErrorOccurence << endl;

   double flops = 2.0 * iterations * matrix. getNonzeroElements();
   this -> gflops = flops / time * 1.0e-9;
   artificialZeros = matrix. getArtificialZeroElements();

   if( verbose )
      writeProgress();
}

template< typename Real,
          typename Device,
          typename Index,
          template< typename matrixReal, typename matrixDevice, typename matrixIndex > class Matrix >
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
          typename Device,
          typename Index,
          template< typename matrixReal, typename matrixDevice, typename matrixIndex > class Matrix >
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
          typename Device,
          typename Index,
          template< typename matrixReal, typename matrixDevice, typename matrixIndex > class Matrix >
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

template< typename Real,
          typename Device,
          typename Index,
          template< typename matrixReal, typename matrixDevice, typename matrixIndex > class Matrix >
bool tnlSpmvBenchmark< Real, Device, Index, Matrix > :: testMatrix( const tnlMatrix< Real, tnlHost, Index >& testMatrix,
                                                                    int verbose ) const
{
   if( ! this -> setupOk )
      return false;

#ifndef HAVE_CUDA
   if( Device :: getDevice() == tnlCudaDevice )
      return false;
#endif

   const Index size = matrix. getSize();
   if( size != testMatrix. getSize() )
   {
      cerr << "Both matrices " << this -> matrix. getName() << " and " << testMatrix. getName()
           << " have different sizes: " << size << " and " << testMatrix. getSize() << "." << endl;
      return false;
   }
   if( Device :: getDevice() == tnlHostDevice )
   {
      for( Index i = 0; i < size; i ++ )
      {
         for( Index j = 0; j < size; j ++ )
            if( matrix. getElement( i, j ) != testMatrix. getElement( i, j ) )
            {
               if( verbose )
                  cout << "Comparing with testing matrix: " << i + 1 << " / " << size << " error at column " << j << "." << endl;
               return false;
            }
         if( verbose )
            cout << "Comparing with testing matrix: " << i + 1 << " / " << size << "           \r" << flush;
      }
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      tnlVector< Real, Device, Index > x( "x" ), b( "b" );
      if( ! x. setSize( size ) || ! b. setSize( size ) )
         return false;
      for( Index j = 0; j < size; j ++ )
      {
         x. setValue( 0.0 );
         x. setElement( j, 1.0 );
         this -> matrix. vectorProduct( x, b );
         for( Index i = 0; i < size; i ++ )
            if( b. getElement( i ) != testMatrix. getElement( i, j ) )
            {
               if( verbose )
                  cout << "Comparing with testing matrix: " << j + 1 << " / " << size << " error at line " << i << "." << endl;
               return false;
            }
         if( verbose )
            cout << "Comparing with testing matrix: " << j + 1 << " / " << size << "           \r" << flush;
      }
#endif
   }
   //if( verbose )
   //   cout << endl;
   return true;
}

#endif /* TNLSPMVBENCHMARK_H_ */
