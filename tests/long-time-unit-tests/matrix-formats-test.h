/***************************************************************************
                          matrix-formats-test.h  -  description
                             -------------------
    begin                : Dec 14, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef MATRIX_FORMATS_TEST_H_
#define MATRIX_FORMATS_TEST_H_

#include <TNL/matrices/tnlMatrixReader.h>

#include <cstdlib>
#include <TNL/File.h>
#include <TNL/debug/tnlDebug.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/matrices/tnlDenseMatrix.h>
#include <TNL/matrices/tnlEllpackMatrix.h>
#include <TNL/matrices/tnlSlicedEllpackMatrix.h>
#include <TNL/matrices/tnlChunkedEllpackMatrix.h>
#include <TNL/matrices/tnlCSRMatrix.h>

using namespace TNL;

void setupConfig( Config::ConfigDescription& config )
{
    config.addDelimiter                            ( "General settings:" );
    config.addEntry< String >( "input-file", "Input file name." );
    config.addEntry< String >( "matrix-format", "Matrix format." );
       config.addEntryEnum< String >( "dense" );
       config.addEntryEnum< String >( "ellpack" );
       config.addEntryEnum< String >( "sliced-ellpack" );
       config.addEntryEnum< String >( "chunked-ellpack" );
       config.addEntryEnum< String >( "csr" );
   config.addEntry< bool >( "hard-test", "Comparison against the dense matrix.", false );
   config.addEntry< bool >( "multiplication-test", "Matrix-vector multiplication test.", false );
   config.addEntry< bool >( "verbose", "Verbose mode." );
}


template< typename Matrix >
bool testMatrix( const Config::ParameterContainer& parameters )
{
   Matrix matrix;
   typedef typename Matrix::RealType RealType;
   typedef typename Matrix::DeviceType DeviceType;
   typedef typename Matrix::IndexType IndexType;

   const String& fileName = parameters.getParameter< String >( "input-file" );
   bool verbose = parameters.getParameter< bool >( "verbose" );
   std::fstream file;
   file.open( fileName.getString(), std::ios::in );
   if( ! file )
   {
      std::cerr << "Cannot open the file " << fileName << std::endl;
      return false;
   }
   if( ! tnlMatrixReader< Matrix >::readMtxFile( file, matrix, verbose ) )
      return false;
   if( ! tnlMatrixReader< Matrix >::verifyMtxFile( file, matrix, verbose ) )
      return false;
   if( parameters.getParameter< bool >( "hard-test" ) )
   {
      typedef tnlDenseMatrix< RealType, DeviceType, IndexType > DenseMatrix;
      DenseMatrix denseMatrix;
      if( ! tnlMatrixReader< DenseMatrix >::readMtxFile( file, denseMatrix, verbose ) )
         return false;
      if( ! tnlMatrixReader< DenseMatrix >::verifyMtxFile( file, denseMatrix, verbose ) )
         return false;
      //matrix.print(std::cout );
      //denseMatrix.print(std::cout );
      for( IndexType i = 0; i < matrix.getRows(); i++ )
      {
         for( IndexType j = 0; j < matrix.getColumns(); j++ )
            if( matrix.getElement( i, j ) != denseMatrix.getElement( i, j ) )
            {
               std::cerr << "The matrices differ at position " << i << ", " << j << "." << std::endl
                    << " The values are " << matrix.getElement( i, j ) << " (sparse) and "
                    << denseMatrix.getElement( i, j ) << " (dense)." << std::endl;
               String line;
               IndexType lineNumber;
               if( tnlMatrixReader< Matrix >::findLineByElement( file, i, j, line, lineNumber ) )
                  std::cerr << "The mtx file says ( line " << lineNumber << " ): " << line << std::endl;
               else
                  std::cerr << "The element is missing in the file. Should be zero therefore." << std::endl;
               return false;
            }
         if( verbose )
           std::cout << " Comparing the sparse matrix with the dense matrix ... " << i << " / " << matrix.getRows() << "             \r" << std::flush;
      }
      if( verbose )
        std::cout << " Comparing the sparse matrix with the dense matrix ... OK.           " << std::endl;
   }
   if( parameters.getParameter< bool >( "multiplication-test" ) )
   {
      Vectors::Vector< RealType, DeviceType, IndexType > x, b;
      x.setSize( matrix.getColumns() );
      b.setSize( matrix.getRows() );
      for( IndexType i = 0; i < x.getSize(); i++ )
      {
         x.setValue( 0 );
         x.setElement( i, 1.0 );
         matrix.vectorProduct( x, b );
         for( IndexType j = 0; j < b.getSize(); j++ )
            if( b.getElement( j ) != matrix.getElement( j, i ) )
            {
               std::cerr << "The matrix-vector multiplication gives wrong result at positions "
                    << j << ", " << i << ". The result is " << b.getElement( j ) << " and it should be "
                    << matrix.getElement( j, i ) << "." << std::endl;
               return false;
            }
         if( verbose )
            std::cerr << " Testing the matrix-vector multiplication ... " << i << " / " << matrix.getRows() << "            \r" << std::flush;
      }
      if( verbose )
         std::cerr << " Testing the matrix-vector multiplication ...  OK.                                       " << std::endl;
   }
   return true;
}

int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   setupConfig( conf_desc );
 
   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc.printUsage( argv[ 0 ] );
      return EXIT_FAILURE;
   }

   const String& matrixFormat = parameters.getParameter< String >( "matrix-format" );
   if( matrixFormat == "dense" )
   {
       if( !testMatrix< tnlDenseMatrix< double, tnlHost, int > >( parameters ) )
          return EXIT_FAILURE;
       return EXIT_SUCCESS;
   }
   if( matrixFormat == "ellpack" )
   {
       if( !testMatrix< tnlEllpackMatrix< double, tnlHost, int > >( parameters ) )
          return EXIT_FAILURE;
       return EXIT_SUCCESS;
   }
   if( matrixFormat == "sliced-ellpack" )
   {
       if( !testMatrix< tnlSlicedEllpackMatrix< double, tnlHost, int > >( parameters ) )
          return EXIT_FAILURE;
       return EXIT_SUCCESS;
   }
   if( matrixFormat == "chunked-ellpack" )
   {
       if( !testMatrix< tnlChunkedEllpackMatrix< double, tnlHost, int > >( parameters ) )
          return EXIT_FAILURE;
       return EXIT_SUCCESS;
   }
   if( matrixFormat == "csr" )
   {
       if( !testMatrix< tnlCSRMatrix< double, tnlHost, int > >( parameters ) )
          return EXIT_FAILURE;
       return EXIT_SUCCESS;
   }
   std::cerr << "Uknown matrix format " << matrixFormat << "." << std::endl;
   return EXIT_FAILURE;
}

#endif /* MATRIX_FORMATS_TEST_H_ */
