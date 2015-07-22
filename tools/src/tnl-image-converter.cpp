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
#include <mesh/tnlGrid.h>
#include <core/io/tnlPGMImage.h>
#include <core/io/tnlRegionOfInterest.h>

void configSetup( tnlConfigDescription& config )
{
   config.addDelimiter( "General parameters" );
   config.addRequiredList < tnlString >( "input-images",  "Input files with images." );
   config.addEntry        < tnlString >( "mesh-file",     "Mesh file.", "mesh.tnl" );
   config.addEntry        < bool >     ( "one-mesh-file", "Generate only one mesh file. All the images dimensions must be the same.", true );
   config.addEntry        < int >      ( "roi-top",       "Top (smaller number) line of the region of interest.", -1 );
   config.addEntry        < int >      ( "roi-bottom",    "Bottom (larger number) line of the region of interest.", -1 );
   config.addEntry        < int >      ( "roi-left",      "Left (smaller number) column of the region of interest.", -1 );
   config.addEntry        < int >      ( "roi-right",     "Right (larger number) column of the region of interest.", -1 );
   config.addEntry        < bool >      ( "verbose",       "Set the verbosity of the program.", true );
}

template< typename Index >
bool resolveRoi( const tnlParameterContainer& parameters,
                 const tnlImage< Index >* image,
                 int& top,
                 int& bottom,
                 int& right,
                 int& left )
{
}

template< typename Index,
          typename Grid >
bool setGrid( const tnlRegionOfInterest< Index >& roi,
              Grid& grid,
              bool verbose = false )
{
    grid.setDimensions( roi.getWidth(), roi.getHeight() );
    typename Grid::VertexType origin, proportions;
    origin.x() = 0.0;
    origin.y() = 0.0;
    proportions.x() = 1.0;
    proportions.y() = ( double ) grid.getDimensions().x() / ( double ) grid.getDimensions().y();
    grid.setDomain( origin, proportions );
    if( verbose )
    {
        cout << "Setting grid to dimensions " << grid.getDimensions() << 
                " and proportions " << grid.getProportions() << endl;
    }
    return true;
}

template< typename Image,
          typename Grid,
          typename Vector >
bool readImage( const Image& image,
                const Grid& grid,
                Vector& vector )
{
    
}

bool processImages( const tnlParameterContainer& parameters )
{
    const tnlList< tnlString >& inputImages = parameters.getParameter< tnlList< tnlString > >( "input-images" );
    tnlString meshFile = parameters.getParameter< tnlString >( "mesh-file" );
    
    bool verbose = parameters.getParameter< bool >( "verbose" );
    
    tnlGrid< 2, double, tnlHost, int > grid;
    tnlVector< double, tnlHost, int > vector;
    tnlRegionOfInterest< int > roi;
    for( int i = 0; i < inputImages.getSize(); i++ )
    {
        const tnlString& fileName = inputImages[ i ];
        cout << "Processing image file " << fileName << "... ";
        tnlPGMImage< int > pgmImage;
        if( pgmImage.openForRead( fileName ) )
        {
            cout << "PGM format detected ...";
            if( i == 0 )
            {
                if( ! roi.setup( parameters, &pgmImage ) )
                    return false;
                setGrid( roi, grid, verbose );
                vector.setSize( grid.getNumberOfCells() );
                cout << "Writing grid to file " << meshFile << endl;
                grid.save( meshFile );
            }
            else 
                if( ! roi.check( &pgmImage ) )
                    return false;
            if( ! pgmImage.read( roi, grid, vector ) )
                return false;
            tnlString outputFileName( fileName );
            RemoveFileExtension( outputFileName );
            outputFileName += ".tnl";
            cout << "Writing image data to " << outputFileName << endl;
            vector.save( outputFileName );
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