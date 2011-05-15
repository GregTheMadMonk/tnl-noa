/***************************************************************************
                          tnlSpmvBenchmarkRgRgCSRMatrix.h  -  description
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

#include <tnlSpmvBenchmark.h>
#include <matrix/tnlRgCSRMatrix.h>

template< typename Real, tnlDevice Device, typename Index>
class tnlSpmvBenchmarkRgCSRMatrix : public tnlSpmvBenchmark< Real, Device, Index, tnlRgCSRMatrix >
{
   public:

   tnlSpmvBenchmarkRgCSRMatrix();

   bool setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix );

   void tearDown();

   void writeProgress() const;

   void setGroupSize( const Index groupSize );

   void setCudaBlockSize( const Index cudaBlockSize );

   Index getArtificialZeroElements() const;

   protected:

   Index groupSize;

   Index cudaBlockSize;
};

template< typename Real,
          tnlDevice Device,
          typename Index>
tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: tnlSpmvBenchmarkRgCSRMatrix()
 : groupSize( 0 ),
   cudaBlockSize( 0 )
{
}

template< typename Real,
          tnlDevice Device,
          typename Index>
bool tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix )
{
   tnlAssert( this -> groupSize > 0, cerr << "groupSize = " << this -> groupSize );
   if( Device == tnlHost )
   {
      this -> matrix. copyFrom( matrix, groupSize );
   }
   if( Device == tnlCuda )
   {
#ifdef HAVE_CUDA
      tnlRgCSRMatrix< Real, tnlHost, Index > hostMatrix( "tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: setup : hostMatrix" );
      hostMatrix. copyFrom( matrix, groupSize );
      this -> matrix. copyFrom( hostMatrix );
#endif
   }
}

template< typename Real,
          tnlDevice Device,
          typename Index>
void tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: tearDown()
{
   //this -> matrix. setSize( 0 );
   //this -> matrix. setNonzeroElements( 0 );
}

template< typename Real,
          tnlDevice Device,
          typename Index >
void tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: writeProgress() const
{
   cout << left << setw( this -> formatColumnWidth - 15 ) << "Row-grouped CSR ";
   if( Device == tnlCuda )
      cout << setw( 5 ) << this -> groupSize
           << setw( 10 ) << this -> cudaBlockSize;
   else
      cout << setw( 15 ) << this -> groupSize;
   cout << right << setw( this -> timeColumnWidth ) << setprecision( 2 ) << this -> getTime()
        << right << setw( this -> iterationsColumnWidth ) << this -> getIterations()
        << right << setw( this -> gflopsColumnWidth ) << setprecision( 2 ) << this -> getGflops();
   if( this -> getBenchmarkWasSuccesful() )
        cout << right << setw( this -> benchmarkStatusColumnWidth ) << "OK ";
   else
        cout << right << setw( this -> benchmarkStatusColumnWidth ) << "FAILED ";
   if( Device == tnlCuda )
#ifndef HAVE_CUDA
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
#ifdef HAVE_CUDA
   this -> matrix. setCUDABlockSize( cudaBlockSize );
#endif
   this -> cudaBlockSize = cudaBlockSize;
}

template< typename Real,
          tnlDevice Device,
          typename Index >
Index tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: getArtificialZeroElements() const
{
   return this -> matrix. getArtificialZeroElements();
}



#endif /* TNLSPMVBENCHMARKRGCSRMATRIX_H_ */
