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
#include <core/images/tnlDicomSeries.h>
#include <core/mfilename.h>

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
   const tnlList< tnlString >& dicomSeriesNames = parameters.getParameter< tnlList< tnlString > >( "dicom-series" );
   tnlString meshFile = parameters.getParameter< tnlString >( "mesh-file" );    
   bool verbose = parameters.getParameter< bool >( "verbose" );

   tnlGrid< 2, double, tnlHost, int > grid;
   tnlVector< double, tnlHost, int > vector;
   tnlRegionOfInterest< int > roi;   
   for( int i = 0; i < dicomSeriesNames.getSize(); i++ )
   {
      const tnlString& seriesName = dicomSeriesNames[ i ];
      cout << "Reading a file " << seriesName << endl;   
      tnlDicomSeries dicomSeries( seriesName.getString() );
      if( !dicomSeries.isDicomSeriesLoaded() )
      {
         cerr << "Loading of the DICOM series " << seriesName << " failed." << endl;
      }
      if( i == 0 )
      {
         if( ! roi.setup( parameters, &dicomSeries ) )
            return false;
         roi.setGrid( grid, verbose );
         vector.setSize( grid.getNumberOfCells() );
         cout << "Writing grid to file " << meshFile << endl;
         grid.save( meshFile );
      }
      cout << "The series consists of " << dicomSeries.getImagesCount() << " images." << endl;
      for( int imageIdx = 0; imageIdx < dicomSeries.getImagesCount(); imageIdx++ )
      {
         dicomSeries.getImage( imageIdx, grid, roi, vector );
         tnlString fileName;
         FileNameBaseNumberEnding( seriesName.getString(), imageIdx, 2, ".tnl", fileName );
         cout << "Writing file " << fileName << " ... " << endl;
         vector.save( fileName );
      }      
   }
}
#endif

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription configDescription;
   setupConfig( configDescription );
   if( ! parseCommandLine( argc, argv, configDescription, parameters ) )
   {
      configDescription.printUsage( argv[ 0 ] );
      return EXIT_FAILURE;
   }   
   if( ! parameters.checkParameter( "dicom-files" ) &&
       ! parameters.checkParameter( "dicom-series") )
   {
       cerr << "Neither DICOM series nor DICOM files are given." << endl;
       configDescription.printUsage( argv[ 0 ] );
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