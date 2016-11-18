/***************************************************************************
                          tnl-view.cpp  -  description
                             -------------------
    begin                : Jan 21, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include "tnl-view.h"
#include <cstdlib>
#include <TNL/File.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Meshes/DummyMesh.h>
#include <TNL/Meshes/Grid.h>

void setupConfig( Config::ConfigDescription& config )
{
   config.addDelimiter                            ( "General settings:" );
   config.addEntry        < String >           ( "mesh", "Mesh file.", "mesh.tnl" );
   config.addRequiredList < String >           ( "input-files", "Input files." );
   config.addList         < String >           ( "output-files", "Output files." );
   config.addEntry        < bool >                ( "check-output-file", "If the output file already exists, do not recreate it.", false );

   config.addDelimiter( "Grid settings:");
   config.addList         < double >              ( "level-lines", "List of level sets which will be drawn." );
   config.addEntry        < int >                 ( "output-x-size", "X size of the output." );
   config.addEntry        < int >                 ( "output-y-size", "Y size of the output." );
   config.addEntry        < int >                 ( "output-z-size", "Z size of the output." );
   config.addEntry        < double >              ( "scale", "Multiply the function by given number.", 1.0 );
   config.addEntry        < String >           ( "output-format", "Output file format.", "gnuplot" );
      config.addEntryEnum  < String >             ( "gnuplot" );
      config.addEntryEnum  < String >             ( "vtk" );
   config.addEntry        < int >                 ( "verbose", "Set the verbosity of the program.", 1 );

   config.addDelimiter( "Matrix settings:" );
   config.addEntry        < String >           ( "matrix-format", "Matrix format to be drawn." );
      config.addEntryEnum  < String >             ( "csr" );
      config.addEntryEnum  < String >             ( "ellpack" );
      config.addEntryEnum  < String >             ( "sliced-ellpack" );
      config.addEntryEnum  < String >             ( "chunked-ellpack" );
   config.addEntry        < int >                 ( "matrix-slice-size", "Sets the slice size of the matrix.", 0 );
   config.addEntry        < int >                 ( "desired-matrix-chunk-size", "Sets desired chunk size for the Chunked Ellpack format.");
   config.addEntry        < int >                 ( "cuda-block-size", "Sets CUDA block size for the Chunked Ellpack format." );
   config.addEntry       < bool >                 ( "sort-matrix", "Sort the matrix rows decreasingly by the number of the non-zero elements.", false );
}

int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;
   setupConfig( conf_desc );
   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;


   int verbose = parameters. getParameter< int >( "verbose" );
   String meshFile = parameters. getParameter< String >( "mesh" );
   if( meshFile == "" )
   {
      //if( ! processFiles< DummyMesh< double, Devices::Host, int > >( parameters ) )
      //   return EXIT_FAILURE;
      //return EXIT_SUCCESS;
   }
   String meshType;
   if( ! getObjectType( meshFile, meshType ) )
   {
      std::cerr << "I am not able to detect the mesh type from the file " << meshFile << "." << std::endl;
      return EXIT_FAILURE;
   }
   std::cout << meshType << " detected in " << meshFile << " file." << std::endl;
   Containers::List< String > parsedMeshType;
   if( ! parseObjectType( meshType, parsedMeshType ) )
   {
      std::cerr << "Unable to parse the mesh type " << meshType << "." << std::endl;
      return false;
   }
   if( parsedMeshType[ 0 ] == "Meshes::Grid" ||
       parsedMeshType[ 0 ] == "tnlGrid" )   //  TODO: remove deprecated type name
   {
      int dimensions = atoi( parsedMeshType[ 1 ].getString() );
      if( dimensions == 1 )
      {
         typedef Meshes::Grid< 1, double, Devices::Host, int > MeshType;
         if( ! processFiles< MeshType >( parameters ) )
            return EXIT_FAILURE;
      }
      if( dimensions == 2 )
      {
         typedef Meshes::Grid< 2, double, Devices::Host, int > MeshType;
         if( ! processFiles< MeshType >( parameters ) )
            return EXIT_FAILURE;
      }
      if( dimensions == 3 )
      {
         typedef Meshes::Grid< 3, double, Devices::Host, int > MeshType;
         if( ! processFiles< MeshType >( parameters ) )
            return EXIT_FAILURE;
      }
      return EXIT_SUCCESS;
   }
   if( parsedMeshType[ 0 ] == "Meshes::Mesh" )
   {
      /*String meshFile = parameters. getParameter< String >( "mesh" );
      struct MeshConfig : public MeshConfigBase< 2 >
      {
         typedef MeshTriangleTopology CellType;
      };
      Mesh< MeshConfig > mesh;
      if( ! mesh.load( meshFile ) )
         return EXIT_FAILURE;
      if( ! MeshWriterNetgen::writeMesh( "tnl-mesh.ng", mesh, true ) )
         return EXIT_FAILURE;*/
   }
   return EXIT_FAILURE;
}
