/***************************************************************************
                          tnl-view.h  -  description
                             -------------------
    begin                : Jan 21, 2013
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

#ifndef TNL_VIEW_H_
#define TNL_VIEW_H_

#include <core/mfilename.h>
#include <config/tnlParameterContainer.h>
#include <core/tnlString.h>
#include <core/tnlVector.h>
#include <mesh/tnlGrid.h>


template< typename Mesh >
bool processMesh( const tnlParameterContainer& parameters )
{
   int verbose = parameters. GetParameter< int >( "verbose");
   tnlString meshFile = parameters. GetParameter< tnlString >( "mesh" );
   Mesh mesh;
   if( ! mesh. load( meshFile ) )
   {
      cerr << "I am not able to load mesh from the file " << meshFile << "." << endl;
      return false;
   }

   tnlList< tnlString > inputFiles = parameters. GetParameter< tnlList< tnlString > >( "input-files" );

   for( int i = 0; i < inputFiles. getSize(); i ++ )
   {
      if( verbose )
         cout << "Processing file " << inputFiles[ i ] << " ... " << flush;


      tnlString objectType;
      if( ! getObjectType( inputFiles[ i ], objectType ) )
          cerr << "unknown object ... SKIPPING!" << endl;
      else
      {
         if( verbose )
            cout << objectType << " detected ... ";

         tnlString outputFileName( inputFiles[ i ] );
         RemoveFileExtension( outputFileName );
         tnlString outputFormat = parameters. GetParameter< tnlString >( "output-format" );
         if( outputFormat == "gnuplot" )
            outputFileName += ".gplt";
         else
         {
            cerr << "Unknown file format " << outputFormat << "." << endl;
            continue;
         }
         if( verbose )
            cout << " writing to " << outputFileName << " ...                 \r" << flush;

         if( objectType == "tnlSharedVector< double, tnlHost, int >" ||
             objectType == "tnlVector< double, tnlHost, int >" )
         {
            tnlVector< double, tnlHost, int > v;
            if( ! v. load( inputFiles[ i ] ) )
               continue;
            mesh. write( v, outputFileName, outputFormat );
         }
      }
   }
   if( verbose )
      cout << endl;
}


#endif /* TNL_VIEW_H_ */
