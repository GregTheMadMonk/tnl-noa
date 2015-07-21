/***************************************************************************
                          tnl-image-converter.cpp  -  description
                             -------------------
    begin                : Jul 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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

#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <core/mfilename.h>
#include <core/io/tnlPGMImage.h>

void configSetup( tnlConfigDescription& config )
{
   config.addDelimiter( "General parameters" );
   config.addRequiredList < tnlString >( "input-files",   "Input files with images." );
   config.addEntry        < bool >     ( "one-mesh-file", "Generate only one mesh file. All the images dimensions must be the same.", true );
   config.addEntry        < int >      ( "verbose",       "Set the verbosity of the program.", 1 );
}

bool processImages( const tnlParameterContainer& parameters )
{
    const tnlList< tnlString >& inputFiles = parameters.getParameter< tnlList< tnlString > >( "input-files" );
    
    for( int i = 0; i < inputFiles.getSize(); i++ )
    {
        const tnlString& fileName = inputFiles[ i ];
        cout << "Processing image file " << fileName << "... ";
        tnlPGMImage< int > pgmImage;
        if( pgmImage.open( fileName ) )
        {
            cout << "PGM format detected ...";
        }
    }
}

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;
   configSetup( conf_desc );
   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;
   if( ! processImages( parameters ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}