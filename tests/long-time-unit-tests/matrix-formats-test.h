/***************************************************************************
                          matrix-formats-test.h  -  description
                             -------------------
    begin                : Dec 14, 2013
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

#include "tnlConfig.h"
const char configFile[] = TNL_CONFIG_DIRECTORY "tnl-test-matrix-formats.cfg.desc";



template< typename Matrix >
bool testMatrix( const tnlParameterContainer& parameters )
{
   Matrix matrix;
   const tnlString& fileName = parameters.GetParameter< tnlString >( "input-file" );
   bool verbose = parameters.GetParameter< bool >( "verbose" );
   fstream file;
   file.open( fileName.getString(), ios::in );
   if( ! file )
   {
      cerr << "Cannot open the file " << fileName << endl;
      return false;
   }
   if( ! tnlMatrixReader::readMtxFile( file, matrix, verbose ) )
   {
      file.close();
      return false;
   }
   file.close();
   return true;
}

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;
   if( conf_desc. ParseConfigDescription( configFile ) != 0 )
      return EXIT_FAILURE;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc. PrintUsage( argv[ 0 ] );
      return EXIT_FAILURE;
   }

   const tnlString& matrixFormat = parameters.GetParameter< tnlString >( "matrix-format" );
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
