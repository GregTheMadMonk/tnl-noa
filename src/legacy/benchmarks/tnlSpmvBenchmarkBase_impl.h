/***************************************************************************
                          tnlSpmBenchmarkBase_impl.h  -  description
                             -------------------
    begin                : Dec 29, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLSPMVBENCHMARKBASE_IMPL_H_
#define TNLSPMVBENCHMARKBASE_IMPL_H_

template< typename  Matrix >
tnlSpmvBenchmarkBase< Matrix >::tnlSpmvBenchmarkBase()
   : benchmarkWasSuccesful( false ),
     setupOk( false ),
     gflops( 0.0 ),
     time( 0.0 ),
     maxIterations( 0 ),
     iterations( 0.0 ),
     artificialZeros( 0 ),
     maxError( 0.0 ),
     firstErrorOccurence( 0 ),
     formatColumnWidth( 40 ),
     timeColumnWidth( 12 ),
     iterationsColumnWidth( 15 ),
     gflopsColumnWidth( 12 ),
     benchmarkStatusColumnWidth( 12 ),
     infoColumnWidth( 20 )
{
}

template< typename  Matrix >
bool tnlSpmvBenchmarkBase< Matrix >::getBenchmarkWasSuccesful() const
{
   return this->benchmarkWasSuccesful;
}

template< typename Matrix >
double tnlSpmvBenchmarkBase< Matrix >::getGflops() const
{
   return this->gflops;
}

template< typename Matrix >
double tnlSpmvBenchmarkBase< Matrix >::getTime() const
{
   return this->time;
}

template< typename Matrix >
void tnlSpmvBenchmarkBase< Matrix >::setMaxIterations( const int maxIterations )
{
   this->maxIterations = maxIterations;
}

template< typename Matrix >
int tnlSpmvBenchmarkBase< Matrix >::getIterations() const
{
   return this->iterations;
}


template< typename Matrix >
typename Matrix::IndexType tnlSpmvBenchmarkBase< Matrix >::getArtificialZeros() const
{
   return this->artificialZeros;
}

template< typename Matrix >
typename Matrix::RealType tnlSpmvBenchmarkBase< Matrix >::getMaxError() const
{
   return this->maxError;
}

template< typename Matrix >
void tnlSpmvBenchmarkBase< Matrix >::runBenchmark( const tnlVector< RealType, DeviceType, IndexType >& x,
                                                   const tnlVector< RealType, tnlHost, IndexType >& refB,
                                                   bool verbose )
{
   benchmarkWasSuccesful = false;
   if( ! setupOk )
      return;
#ifndef HAVE_CUDA
   if( DeviceType::getDevice() == tnlCudaDevice )
   {
      if( verbose )
         writeProgress();
      return;
   }
#endif

   tnlVector< RealType, DeviceType, IndexType > b( "tnlSpmvBenchmark< Real, Device, Index, Matrix > :: runBenchmark : b" );
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

   this->time = rt_timer. getTime();

   firstErrorOccurence = 0;
   tnlVector< RealType, tnlHost, IndexType > resB( "tnlSpmvBenchmark< Real, Device, Index, Matrix > :: runBenchmark : b" );
   if( ! resB. setSize( b. getSize() ) )
   {
      cerr << "I am not able to allocate copy of vector b on the host." << endl;
      return;
   }
   resB = b;
   benchmarkWasSuccesful = true;
   for( IndexType j = 0; j < refB. getSize(); j ++ )
   {
      //f << refB[ j ] << " - " << host_b[ j ] << " = "  << refB[ j ] - host_b[ j ] <<  endl;
      RealType error( 0.0 );
      if( refB[ j ] != 0.0 )
         error = ( RealType ) fabs( refB[ j ] - resB[ j ] ) /  ( RealType ) fabs( refB[ j ] );
      else
         error = ( RealType ) fabs( refB[ j ] );
      if( error > maxError )
         firstErrorOccurence = j;
      this->maxError = Max( this->maxError, error );

      /*if( error > tnlSpmvBenchmarkPrecision( error ) )
         benchmarkWasSuccesful = false;*/

   }
   //cout << "First error was on " << firstErrorOccurence << endl;

   double flops = 2.0 * iterations * matrix.getNumberOfNonzeroMatrixElements();
   this->gflops = flops / time * 1.0e-9;
   artificialZeros = matrix.getNumberOfMatrixElements() - matrix.getNumberOfNonzeroMatrixElements();

   if( verbose )
      writeProgress();
}

template< typename Matrix >
void tnlSpmvBenchmarkBase< Matrix >::writeProgressTableHeader()
{
   int totalWidth = this->formatColumnWidth +
                    this->timeColumnWidth +
                    this->iterationsColumnWidth +
                    this->gflopsColumnWidth +
                    this->benchmarkStatusColumnWidth +
                    this->infoColumnWidth;

   cout << left << setw( this->formatColumnWidth - 5 ) << "MATRIX FORMAT"
        << left << setw( 5 ) << "BLOCK"
        << right << setw( this->timeColumnWidth ) << "TIME"
        << right << setw( this->iterationsColumnWidth ) << "ITERATIONS"
        << right << setw( this->gflopsColumnWidth ) << "GFLOPS"
        << right << setw( this->benchmarkStatusColumnWidth ) << "CHECK"
        << left << setw(  this->infoColumnWidth ) << " INFO" << endl
        << setfill( '-' ) << setw( totalWidth ) << "--" << endl
        << setfill( ' ');
}

template< typename Matrix >
tnlString tnlSpmvBenchmarkBase< Matrix > :: getBgColorBySpeedUp( const double& speedUp ) const
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


template< typename Matrix >
bool tnlSpmvBenchmarkBase< Matrix > :: printMatrixInHtml( const tnlString& fileName,
                                                          tnlMatrix< RealType, tnlHost, IndexType >& matrix ) const
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

#endif /* TNLSPMVBENCHMARKBASE_IMPL_H_ */
