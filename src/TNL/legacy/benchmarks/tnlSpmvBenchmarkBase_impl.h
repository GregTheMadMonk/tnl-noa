/***************************************************************************
                          tnlSpmBenchmarkBase_impl.h  -  description
                             -------------------
    begin                : Dec 29, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

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
      std::cerr << "I am not able to allocate copy of vector b on the host." << std::endl;
      return;
   }
   resB = b;
   benchmarkWasSuccesful = true;
   for( IndexType j = 0; j < refB. getSize(); j ++ )
   {
      //f << refB[ j ] << " - " << host_b[ j ] << " = "  << refB[ j ] - host_b[ j ] <<  std::endl;
      RealType error( 0.0 );
      if( refB[ j ] != 0.0 )
         error = ( RealType ) fabs( refB[ j ] - resB[ j ] ) /  ( RealType ) fabs( refB[ j ] );
      else
         error = ( RealType ) fabs( refB[ j ] );
      if( error > maxError )
         firstErrorOccurence = j;
      this->maxError = max( this->maxError, error );

      /*if( error > tnlSpmvBenchmarkPrecision( error ) )
         benchmarkWasSuccesful = false;*/

   }
   //cout << "First error was on " << firstErrorOccurence << std::endl;

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

  std::cout << left << std::setw( this->formatColumnWidth - 5 ) << "MATRIX FORMAT"
        << left << std::setw( 5 ) << "BLOCK"
        << right << std::setw( this->timeColumnWidth ) << "TIME"
        << right << std::setw( this->iterationsColumnWidth ) << "ITERATIONS"
        << right << std::setw( this->gflopsColumnWidth ) << "GFLOPS"
        << right << std::setw( this->benchmarkStatusColumnWidth ) << "CHECK"
        << left << std::setw(  this->infoColumnWidth ) << " INFO" << std::endl
        << setfill( '-' ) << std::setw( totalWidth ) << "--" << std::endl
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
   //cout << "Writing to file " << fileName << std::endl;
   std::fstream file;
   file. open( fileName. getString(), std::ios::out );
   if( ! file )
   {
      std::cerr << "I am not able to open the file " << fileName << std::endl;
      return false;
   }
   file << "<html>" << std::endl;
   file << "   <body>" << std::endl;
   matrix. printOut( file, "html" );
   file << "   </body>" << std::endl;
   file << "</html>" << std::endl;
   file. close();
   return true;
}

#endif /* TNLSPMVBENCHMARKBASE_IMPL_H_ */
