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

#include <matrices/tnlMatrixReader.h>

#include <cstdlib>
#include <core/tnlFile.h>
#include <debug/tnlDebug.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <matrices/tnlDenseMatrix.h>
#include <matrices/tnlEllpackMatrix.h>
#include <matrices/tnlSlicedEllpackMatrix.h>
#include <matrices/tnlChunkedEllpackMatrix.h>
#include <matrices/tnlCSRMatrix.h>

void setupConfig( tnlConfigDescription& config )
{
    config.addDelimiter                            ( "General settings:" );
    config.addEntry< tnlString >( "input-file", "Input file name." );
    config.addEntry< tnlString >( "matrix-format", "Matrix format." );
       config.addEntryEnum< tnlString >( "dense" );
       config.addEntryEnum< tnlString >( "ellpack" );
       config.addEntryEnum< tnlString >( "sliced-ellpack" );
       config.addEntryEnum< tnlString >( "chunked-ellpack" );
       config.addEntryEnum< tnlString >( "csr" );
   config.addEntry< bool >( "hard-test", "Comparison against the dense matrix.", false );
   config.addEntry< bool >( "multiplication-test", "Matrix-vector multiplication test.", false );
   config.addEntry< bool >( "verbose", "Verbose mode." );
}


template< typename Matrix >
bool testMatrix( const tnlParameterContainer& parameters )
{
   Matrix matrix;
   typedef typename Matrix::RealType RealType;
   typedef typename Matrix::DeviceType DeviceType;
   typedef typename Matrix::IndexType IndexType;

   const tnlString& fileName = parameters.getParameter< tnlString >( "input-file" );
   bool verbose = parameters.getParameter< bool >( "verbose" );
   fstream file;
   file.open( fileName.getString(), ios::in );
   if( ! file )
   {
      cerr << "Cannot open the file " << fileName << endl;
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
      //matrix.print( cout );
      //denseMatrix.print( cout );
      for( IndexType i = 0; i < matrix.getRows(); i++ )
      {
         for( IndexType j = 0; j < matrix.getColumns(); j++ )
            if( matrix.getElement( i, j ) != denseMatrix.getElement( i, j ) )
            {
               cerr << "The matrices differ at position " << i << ", " << j << "." << endl
                    << " The values are " << matrix.getElement( i, j ) << " (sparse) and "
                    << denseMatrix.getElement( i, j ) << " (dense)." << endl;
               tnlString line;
               IndexType lineNumber;
               if( tnlMatrixReader< Matrix >::findLineByElement( file, i, j, line, lineNumber ) )
                  cerr << "The mtx file says ( line " << lineNumber << " ): " << line << endl;
               else
                  cerr << "The element is missing in the file. Should be zero therefore." << endl;
               return false;
            }
         if( verbose )
            cout << " Comparing the sparse matrix with the dense matrix ... " << i << " / " << matrix.getRows() << "             \r" << flush;
      }
      if( verbose )
         cout << " Comparing the sparse matrix with the dense matrix ... OK.           " << endl;
   }
   if( parameters.getParameter< bool >( "multiplication-test" ) )
   {
      tnlVector< RealType, DeviceType, IndexType > x, b;
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
               cerr << "The matrix-vector multiplication gives wrong result at positions "
                    << j << ", " << i << ". The result is " << b.getElement( j ) << " and it should be "
                    << matrix.getElement( j, i ) << "." << endl;
               return false;
            }
         if( verbose )
            cerr << " Testing the matrix-vector multiplication ... " << i << " / " << matrix.getRows() << "            \r" << flush;
      }
      if( verbose )
         cerr << " Testing the matrix-vector multiplication ...  OK.                                       " << endl;
   }
   return true;
}

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;

   setupConfig( conf_desc );
 
   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc.printUsage( argv[ 0 ] );
      return EXIT_FAILURE;
   }

   const tnlString& matrixFormat = parameters.getParameter< tnlString >( "matrix-format" );
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
   cerr << "Uknown matrix format " << matrixFormat << "." << endl;
   return EXIT_FAILURE;
}

#endif /* MATRIX_FORMATS_TEST_H_ */
