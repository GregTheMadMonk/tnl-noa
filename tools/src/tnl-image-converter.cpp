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

void configSetup( tnlConfigDescription& config )
{
   config.addDelimiter( "General parameters" );
   config.addRequiredList < tnlString >( "input-files",   "Input files with images." );
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
    const int roiTop    = parameters.getParameter< int >( "roi-top" );
    const int roiBottom = parameters.getParameter< int >( "roi-bottom" );
    const int roiRight  = parameters.getParameter< int >( "roi-right" );
    const int roiLeft   = parameters.getParameter< int >( "roi-left" );
    
    if( roiLeft == -1 )
        left = 0;
    else
    {
        if( roiLeft >= image->getWidth() )
        {
            cerr << "ROI left column is larger than image width ( " << image->getWidth() << ")." << cerr;
            return false;
        }
        left = roiLeft;
    }
    
    if( roiRight == -1 )
        right = image->getWidth();
    else
    {
        if( roiRight >= image->getWidth() )
        {
            cerr << "ROI right column is larger than image width ( " << image->getWidth() << ")." << cerr;
            return false;
        }
        right = roiRight;
    }
    
    if( roiTop == -1 )
        top = 0;
    else
    {
        if( roiTop >= image->getHeight() )
        {
            cerr << "ROI top line is larger than image height ( " << image->getHeight() << ")." << cerr;
            return false;
        }
        top = roiTop;
    }
    
    if( roiBottom == -1 )
        bottom = image->getHeight();
    else
    {
        if( roiBottom >= image->getHeight() )
        {
            cerr << "ROI bottom line is larger than image height ( " << image->getHeight() << ")." << cerr;
            return false;
        }
        bottom = roiBottom;
    }
    return true;
}

template< typename Index,
          typename Grid >
bool setGrid( const tnlParameterContainer& parameters,
              const tnlImage< Index >* image,
              Grid& grid,
              bool verbose = false )
{
    int top, bottom, right, left;
    if( ! resolveRoi( parameters, image, top, bottom, right, left ) )
        return false;
    
    grid.setDimensions( right - left, bottom - top );
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

template< typename Index,
          typename Grid >
bool checkGrid( const tnlParameterContainer& parameters,
                const tnlImage< Index >* image,
                Grid& grid )
{
    int top, bottom, right, left;
    if( ! resolveRoi( parameters, image, top, bottom, right, left ) )
        return false;
    
    const int width = right - left;
    const int height = bottom - top;
    if( grid.getDimensions().x() == width &&
        grid.getDimensions().y() == height )
        return true;
    else
        return false;
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
    const tnlList< tnlString >& inputFiles = parameters.getParameter< tnlList< tnlString > >( "input-files" );

    bool verbose = parameters.getParameter< bool >( "verbose" );
    
    tnlGrid< 2, double, tnlHost, int > grid;
    tnlVector< double, tnlHost, int > vector;
    for( int i = 0; i < inputFiles.getSize(); i++ )
    {
        const tnlString& fileName = inputFiles[ i ];
        cout << "Processing image file " << fileName << "... ";
        tnlPGMImage< int > pgmImage;
        if( pgmImage.open( fileName ) )
        {
            cout << "PGM format detected ...";
            if( i == 0 )
                if( ! setGrid( parameters, &pgmImage, grid, verbose ) )
                    return false;
                else
                    vector.setSize( grid.getNumberOfCells() );
            else 
                if( ! checkGrid( parameters, &pgmImage, grid ) )
                    return false;
            if( ! pgmImage.read( vector ) )
                return false;
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