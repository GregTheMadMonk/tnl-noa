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

#include <tnlSpmvBenchmark.h>
#include <matrix/tnlAdaptiveRgCSRMatrix.h>
#include <core/tnlAssert.h>

template< typename Real, tnlDevice Device, typename Index>
class tnlSpmvBenchmarkAdaptiveRgCSRMatrix : public tnlSpmvBenchmark< Real, Device, Index, tnlAdaptiveRgCSRMatrix >
{
   public:

   tnlSpmvBenchmarkAdaptiveRgCSRMatrix();

   bool setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix );

   void tearDown();

   void writeProgress() const;

   void setDesiredChunkSize( const Index desiredChunkSize );

   void setCudaBlockSize( const Index cudaBlockSize );

   Index getArtificialZeroElements() const;

   protected:

   Index desiredChunkSize;

   Index cudaBlockSize;

   bool useAdaptiveGroupSize;

   tnlAdaptiveGroupSizeStrategy adaptiveGroupSizeStrategy;
};

template< typename Real,
          tnlDevice Device,
          typename Index>
tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, Device, Index > :: tnlSpmvBenchmarkAdaptiveRgCSRMatrix()
 : desiredChunkSize( 4 ),
   cudaBlockSize( 32 ),
   useAdaptiveGroupSize( false ),
   adaptiveGroupSizeStrategy( tnlAdaptiveGroupSizeStrategyByAverageRowSize )

{
}

template< typename Real,
          tnlDevice Device,
          typename Index>
bool tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, Device, Index > :: setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix )
{
   //tnlAssert( this -> groupSize > 0, cerr << "groupSize = " << this -> groupSize );
   if( Device == tnlHost )
   {
      this -> matrix. tuneFormat( desiredChunkSize, cudaBlockSize );
      if( ! this -> matrix. copyFrom( matrix ) )
         return false;
   }
   if( Device == tnlCuda )
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
   return true;
}

template< typename Real,
          tnlDevice Device,
          typename Index>
void tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, Device, Index > :: tearDown()
{
   //this -> matrix. setSize( 0 );
   //this -> matrix. setNonzeroElements( 0 );
}

template< typename Real,
          tnlDevice Device,
          typename Index >
void tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, Device, Index > :: writeProgress() const
{
   cout << left << setw( this -> formatColumnWidth - 15 ) << "Adap. Row-grouped CSR ";
   if( Device == tnlCuda )
      cout << setw( 5 ) << this -> desiredChunkSize
           << setw( 10 ) << this -> cudaBlockSize;
   else
      cout << setw( 15 ) << this -> desiredChunkSize;
   cout << right << setw( this -> timeColumnWidth ) << setprecision( 2 ) << this -> getTime()
        << right << setw( this -> iterationsColumnWidth ) << this -> getIterations()
        << right << setw( this -> gflopsColumnWidth ) << setprecision( 2 ) << this -> getGflops();
   if( this -> getBenchmarkWasSuccesful() )
        cout << right << setw( this -> benchmarkStatusColumnWidth ) << "OK ";
   else
        cout << right << setw( this -> benchmarkStatusColumnWidth ) << "FAILED ";
#ifndef HAVE_CUDA
   if( Device == tnlCuda )
      cout << "CUDA support is missing.";
#endif
      cout << endl;
}

template< typename Real,
          tnlDevice Device,
          typename Index >
void tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, Device, Index > :: setDesiredChunkSize( const Index desiredChunkSize )
{
   this -> desiredChunkSize = desiredChunkSize;
}

template< typename Real,
          tnlDevice Device,
          typename Index >
void tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, Device, Index > :: setCudaBlockSize( const Index cudaBlockSize )
{
   this -> cudaBlockSize = cudaBlockSize;
}

template< typename Real,
          tnlDevice Device,
          typename Index >
Index tnlSpmvBenchmarkAdaptiveRgCSRMatrix< Real, Device, Index > :: getArtificialZeroElements() const
{
   return this -> matrix. getArtificialZeroElements();
}

#endif /* TNLSPMVBENCHMARKADAPTIVERGCSRMATRIX_H_ */
