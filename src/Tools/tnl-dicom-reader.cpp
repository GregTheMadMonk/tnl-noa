/***************************************************************************
                          tnl-dicom-reader.cpp  -  description
                             -------------------
    begin                : Jul 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Images//DicomSeries.h>
#include <TNL/FileName.h>

using namespace TNL;

void setupConfig( Config::ConfigDescription& config )
{
   config.addDelimiter( "General parameters" );
   config.addList         < String >( "dicom-files",   "Input DICOM files." );
   config.addList         < String >( "dicom-series",   "Input DICOM series." );
   config.addEntry        < String >( "mesh-file",     "Mesh file.", "mesh.tnl" );
   config.addEntry        < bool >     ( "one-mesh-file", "Generate only one mesh file. All the images dimensions must be the same.", true );
   config.addEntry        < int >      ( "roi-top",       "Top (smaller number) line of the region of interest.", -1 );
   config.addEntry        < int >      ( "roi-bottom",    "Bottom (larger number) line of the region of interest.", -1 );
   config.addEntry        < int >      ( "roi-left",      "Left (smaller number) column of the region of interest.", -1 );
   config.addEntry        < int >      ( "roi-right",     "Right (larger number) column of the region of interest.", -1 );
   config.addEntry        < bool >     ( "verbose",       "Set the verbosity of the program.", true );
}

#ifdef HAVE_DCMTK_H
bool processDicomFiles( const Config::ParameterContainer& parameters )
{
   return true;
}

bool processDicomSeries( const Config::ParameterContainer& parameters )
{
   const Containers::List< String >& dicomSeriesNames = parameters.getParameter< Containers::List< String > >( "dicom-series" );
   String meshFile = parameters.getParameter< String >( "mesh-file" );
   bool verbose = parameters.getParameter< bool >( "verbose" );

   typedef Meshes::Grid< 2, double, Devices::Host, int > GridType;
   GridType grid;
   Containers::Vector< double, Devices::Host, int > vector;
   Images::RegionOfInterest< int > roi;
   for( int i = 0; i < dicomSeriesNames.getSize(); i++ )
   {
      const String& seriesName = dicomSeriesNames[ i ];
      std::cout << "Reading a file " << seriesName << std::endl;
      Images::DicomSeries dicomSeries( seriesName.getString() );
      if( !dicomSeries.isDicomSeriesLoaded() )
      {
         std::cerr << "Loading of the DICOM series " << seriesName << " failed." << std::endl;
      }
      if( i == 0 )
      {
         if( ! roi.setup( parameters, &dicomSeries ) )
            return false;
         roi.setGrid( grid, verbose );
         vector.setSize( grid.template getEntitiesCount< typename GridType::Cell >() );
         std::cout << "Writing grid to file " << meshFile << std::endl;
         grid.save( meshFile );
      }
      std::cout << "The series consists of " << dicomSeries.getImagesCount() << " images." << std::endl;
      for( int imageIdx = 0; imageIdx < dicomSeries.getImagesCount(); imageIdx++ )
      {
         dicomSeries.getImage( imageIdx, grid, roi, vector );
         FileName fileName;
         fileName.setFileNameBase( seriesName.getString() );
         fileName.setExtension( "tnl" );
         fileName.setIndex( imageIdx );
         std::cout << "Writing file " << fileName.getFileName() << " ... " << std::endl;
         File( fileName.getFileName(), std::ios_base::out ) << vector;
      }
   }
   return true;
}
#endif

int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription configDescription;
   setupConfig( configDescription );
   if( ! parseCommandLine( argc, argv, configDescription, parameters ) )
   {
      configDescription.printUsage( argv[ 0 ] );
      return EXIT_FAILURE;
   }
   if( ! parameters.checkParameter( "dicom-files" ) &&
       ! parameters.checkParameter( "dicom-series") )
   {
       std::cerr << "Neither DICOM series nor DICOM files are given." << std::endl;
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
   std::cerr << "TNL was not compiled with DCMTK support." << std::endl;
   return EXIT_FAILURE;
#endif
}
