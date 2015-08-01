/***************************************************************************
                          tnl-dicom-reader.cpp  -  description
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

#include <tnlConfig.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <core/io/DicomSeries.h>

void setupConfig( tnlConfigDescription& config )
{
   config.addDelimiter( "General parameters" );
   config.addList         < tnlString >( "dicom-files",   "Input DICOM files." );
   config.addList         < tnlString >( "dicom-series",   "Input DICOM series." );
   config.addEntry        < tnlString >( "mesh-file",     "Mesh file.", "mesh.tnl" );
   config.addEntry        < bool >     ( "one-mesh-file", "Generate only one mesh file. All the images dimensions must be the same.", true );
   config.addEntry        < int >      ( "roi-top",       "Top (smaller number) line of the region of interest.", -1 );
   config.addEntry        < int >      ( "roi-bottom",    "Bottom (larger number) line of the region of interest.", -1 );
   config.addEntry        < int >      ( "roi-left",      "Left (smaller number) column of the region of interest.", -1 );
   config.addEntry        < int >      ( "roi-right",     "Right (larger number) column of the region of interest.", -1 );
   config.addEntry        < bool >     ( "verbose",       "Set the verbosity of the program.", true );   
}

#ifdef HAVE_DCMTK_H
bool processDicomFiles( const tnlParameterContainer& parameters )
{
   
}

bool processDicomSeries( const tnlParameterContainer& parameters )
{
   const tnlList< tnlString >& dicomSeries = parameters.getParameter< tnlList< tnlString > >( "dicom-series" );
   
   for( int i = 0; i < dicomSeries.getSize(); i++ )
   {
      const tnlString& series = dicomSeries[ i ];
      DicomSeries( series.getString() );
   }
}
#endif

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
   if( ! parameters.checkParameter( "dicom-files" ) &&
       ! parameters.checkParameter( "dicom-series") )
   {
       cerr << "Neither DICOM series nor DICOM files are given." << endl;
       return EXIT_FAILURE;
   }
#ifdef HAVE_DCMTK_H   
   if( parameters.checkParameter( "dicom-files" ) && ! processDicomFiles( parameters ) )
      return EXIT_FAILURE;
   if( parameters.checkParameter( "dicom-series" ) && ! processDicomSeries( parameters ) )
      return EXIT_FAILURE;   
   return EXIT_SUCCESS;
#else
   cerr << "TNL was not compiled with DCMTK support." << endl;
#endif   
}