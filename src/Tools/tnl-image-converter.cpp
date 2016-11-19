/***************************************************************************
                          tnl-image-converter.cpp  -  description
                             -------------------
    begin                : Jul 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/FileName.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Images//PGMImage.h>
#include <TNL/Images//PNGImage.h>
#include <TNL/Images//JPEGImage.h>
#include <TNL/Images//RegionOfInterest.h>

using namespace TNL;

void configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "General parameters" );
   config.addList < String >( "input-images",  "Input images for conversion to .tnl files." );
   config.addList < String >( "input-files",   "Input .tnl files for conversion to images." );
   config.addEntry        < String >( "image-format",  "Output images file format.", "pgm" );
   config.addEntry        < String >( "mesh-file",     "Mesh file.", "mesh.tnl" );
   config.addEntry        < bool >     ( "one-mesh-file", "Generate only one mesh file. All the images dimensions must be the same.", true );
   config.addEntry        < int >      ( "roi-top",       "Top (smaller number) line of the region of interest.", -1 );
   config.addEntry        < int >      ( "roi-bottom",    "Bottom (larger number) line of the region of interest.", -1 );
   config.addEntry        < int >      ( "roi-left",      "Left (smaller number) column of the region of interest.", -1 );
   config.addEntry        < int >      ( "roi-right",     "Right (larger number) column of the region of interest.", -1 );
   config.addEntry        < bool >      ( "verbose",       "Set the verbosity of the program.", true );
}


bool processImages( const Config::ParameterContainer& parameters )
{
    const Containers::List< String >& inputImages = parameters.getParameter< Containers::List< String > >( "input-images" );
    String meshFile = parameters.getParameter< String >( "mesh-file" );
    bool verbose = parameters.getParameter< bool >( "verbose" );
 
    typedef Meshes::Grid< 2, double, Devices::Host, int > GridType;
    GridType grid;
    Containers::Vector< double, Devices::Host, int > vector;
    Images::RegionOfInterest< int > roi;
    for( int i = 0; i < inputImages.getSize(); i++ )
    {
      const String& fileName = inputImages[ i ];
      std::cout << "Processing image file " << fileName << "... ";
      Images::tnlPGMImage< int > pgmImage;
      if( pgmImage.openForRead( fileName ) )
      {
         std::cout << "PGM format detected ...";
         if( i == 0 )
         {
            if( ! roi.setup( parameters, &pgmImage ) )
               return false;
            roi.setGrid( grid, verbose );
            vector.setSize( grid.template getEntitiesCount< typename GridType::Cell >() );
            std::cout << "Writing grid to file " << meshFile << std::endl;
            grid.save( meshFile );
         }
         else
            if( ! roi.check( &pgmImage ) )
               return false;
         if( ! pgmImage.read( roi, grid, vector ) )
            return false;
         String outputFileName( fileName );
         removeFileExtension( outputFileName );
         outputFileName += ".tnl";
         std::cout << "Writing image data to " << outputFileName << std::endl;
         vector.save( outputFileName );
         pgmImage.close();
         continue;
      }
      Images::PNGImage< int > pngImage;
      if( pngImage.openForRead( fileName ) )
      {
         std::cout << "PNG format detected ...";
         if( i == 0 )
         {
            if( ! roi.setup( parameters, &pngImage ) )
               return false;
            roi.setGrid( grid, verbose );
            vector.setSize( grid.template getEntitiesCount< typename GridType::Cell >() );
            std::cout << "Writing grid to file " << meshFile << std::endl;
            grid.save( meshFile );
         }
         else
            if( ! roi.check( &pgmImage ) )
               return false;
         if( ! pngImage.read( roi, grid, vector ) )
            return false;
         String outputFileName( fileName );
         removeFileExtension( outputFileName );
         outputFileName += ".tnl";
         std::cout << "Writing image data to " << outputFileName << std::endl;
         vector.save( outputFileName );
         pgmImage.close();
         continue;
      }
      Images::JPEGImage< int > jpegImage;
      if( jpegImage.openForRead( fileName ) )
      {
         std::cout << "JPEG format detected ...";
         if( i == 0 )
         {
            if( ! roi.setup( parameters, &jpegImage ) )
               return false;
            roi.setGrid( grid, verbose );
            vector.setSize( grid.template getEntitiesCount< typename GridType::Cell >() );
            std::cout << "Writing grid to file " << meshFile << std::endl;
            grid.save( meshFile );
         }
         else
            if( ! roi.check( &jpegImage ) )
               return false;
         if( ! jpegImage.read( roi, grid, vector ) )
            return false;
         String outputFileName( fileName );
         removeFileExtension( outputFileName );
         outputFileName += ".tnl";
         std::cout << "Writing image data to " << outputFileName << std::endl;
         vector.save( outputFileName );
         pgmImage.close();
         continue;
      }
   }
   return true;
}

bool processTNLFiles( const Config::ParameterContainer& parameters )
{
   const Containers::List< String >& inputFiles = parameters.getParameter< Containers::List< String > >( "input-files" );
   const String& imageFormat = parameters.getParameter< String >( "image-format" );
   String meshFile = parameters.getParameter< String >( "mesh-file" );
   bool verbose = parameters.getParameter< bool >( "verbose" );
 
   Meshes::Grid< 2, double, Devices::Host, int > grid;
   if( ! grid.load( meshFile ) )
   {
      std::cerr << "I am not able to load the mesh file " << meshFile << "." << std::endl;
      return false;
   }
   Containers::Vector< double, Devices::Host, int > vector;
   for( int i = 0; i < inputFiles.getSize(); i++ )
   {
      const String& fileName = inputFiles[ i ];
      std::cout << "Processing file " << fileName << "... ";
      if( ! vector.load( fileName ) )
      {
         std::cerr << "I am not able to load data from a file " << fileName << "." << std::endl;
         return false;
      }
      if( imageFormat == "pgm" || imageFormat == "pgm-binary" || imageFormat == "pgm-ascii" )
      {
         Images::tnlPGMImage< int > image;
         String outputFileName( fileName );
         removeFileExtension( outputFileName );
         outputFileName += ".pgm";
	 if ( imageFormat == "pgm" || imageFormat == "pgm-binary")
         	image.openForWrite( outputFileName, grid, true );
	 if ( imageFormat == "pgm-ascii" )
         	image.openForWrite( outputFileName, grid, false );
         image.write( grid, vector );
         image.close();
         continue;
      }
      if( imageFormat == "png" )
      {
         Images::PNGImage< int > image;
         String outputFileName( fileName );
         removeFileExtension( outputFileName );
         outputFileName += ".png";
         image.openForWrite( outputFileName, grid );
         image.write( grid, vector );
         image.close();
      }
      if( imageFormat == "jpg" )
      {
         Images::JPEGImage< int > image;
         String outputFileName( fileName );
         removeFileExtension( outputFileName );
         outputFileName += ".jpg";
         image.openForWrite( outputFileName, grid );
         image.write( grid, vector );
         image.close();
      }

   }
   return true;
}

int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription configDescription;
   configSetup( configDescription );
   if( ! parseCommandLine( argc, argv, configDescription, parameters ) )
   {
      configDescription.printUsage( argv[ 0 ] );
      return EXIT_FAILURE;
   }
   if( ! parameters.checkParameter( "input-images" ) &&
       ! parameters.checkParameter( "input-files") )
   {
       std::cerr << "Neither input images nor input .tnl files are given." << std::endl;
       configDescription.printUsage( argv[ 0 ] );
       return EXIT_FAILURE;
   }
   if( parameters.checkParameter( "input-images" ) && ! processImages( parameters ) )
      return EXIT_FAILURE;
   if( parameters.checkParameter( "input-files" ) && ! processTNLFiles( parameters ) )
      return EXIT_FAILURE;

   return EXIT_SUCCESS;
}
