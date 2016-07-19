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
#include <core/tnlFile.h>
#include <debug/tnlDebug.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <mesh/tnlDummyMesh.h>
#include <mesh/tnlGrid.h>

void setupConfig( tnlConfigDescription& config )
{
   config.addDelimiter                            ( "General settings:" );
   config.addEntry        < tnlString >           ( "mesh", "Mesh file.", "mesh.tnl" );
   config.addRequiredList < tnlString >           ( "input-files", "Input files." );
   config.addList         < tnlString >           ( "output-files", "Output files." );
   config.addEntry        < bool >                ( "check-output-file", "If the output file already exists, do not recreate it.", false );

   config.addDelimiter( "Grid settings:");
   config.addList         < double >              ( "level-lines", "List of level sets which will be drawn." );
   config.addEntry        < int >                 ( "output-x-size", "X size of the output." );
   config.addEntry        < int >                 ( "output-y-size", "Y size of the output." );
   config.addEntry        < int >                 ( "output-z-size", "Z size of the output." );
   config.addEntry        < double >              ( "scale", "Multiply the function by given number.", 1.0 );
   config.addEntry        < tnlString >           ( "output-format", "Output file format.", "gnuplot" );
      config.addEntryEnum  < tnlString >             ( "gnuplot" );
      config.addEntryEnum  < tnlString >             ( "vtk" );
   config.addEntry        < int >                 ( "verbose", "Set the verbosity of the program.", 1 );

   config.addDelimiter( "Matrix settings:" );
   config.addEntry        < tnlString >           ( "matrix-format", "Matrix format to be drawn." );
      config.addEntryEnum  < tnlString >             ( "csr" );
      config.addEntryEnum  < tnlString >             ( "ellpack" );
      config.addEntryEnum  < tnlString >             ( "sliced-ellpack" );
      config.addEntryEnum  < tnlString >             ( "chunked-ellpack" );
   config.addEntry        < int >                 ( "matrix-slice-size", "Sets the slice size of the matrix.", 0 );
   config.addEntry        < int >                 ( "desired-matrix-chunk-size", "Sets desired chunk size for the Chunked Ellpack format.");
   config.addEntry        < int >                 ( "cuda-block-size", "Sets CUDA block size for the Chunked Ellpack format." );
   config.addEntry       < bool >                 ( "sort-matrix", "Sort the matrix rows decreasingly by the number of the non-zero elements.", false );
}

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;
   setupConfig( conf_desc );
   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;


   int verbose = parameters. getParameter< int >( "verbose" );
   tnlString meshFile = parameters. getParameter< tnlString >( "mesh" );
   if( meshFile == "" )
   {
      //if( ! processFiles< tnlDummyMesh< double, tnlHost, int > >( parameters ) )
      //   return EXIT_FAILURE;
      //return EXIT_SUCCESS;
   }
   tnlString meshType;
   if( ! getObjectType( meshFile, meshType ) )
   {
      std::cerr << "I am not able to detect the mesh type from the file " << meshFile << "." << std::endl;
      return EXIT_FAILURE;
   }
   std::cout << meshType << " detected in " << meshFile << " file." << std::endl;
   tnlList< tnlString > parsedMeshType;
   if( ! parseObjectType( meshType, parsedMeshType ) )
   {
      std::cerr << "Unable to parse the mesh type " << meshType << "." << std::endl;
      return false;
   }
   if( parsedMeshType[ 0 ] == "tnlGrid" )
   {
      int dimensions = atoi( parsedMeshType[ 1 ].getString() );
      if( dimensions == 1 )
      {
         typedef tnlGrid< 1, double, tnlHost, int > MeshType;
         if( ! processFiles< MeshType >( parameters ) )
            return EXIT_FAILURE;
      }
      if( dimensions == 2 )
      {
         typedef tnlGrid< 2, double, tnlHost, int > MeshType;
         if( ! processFiles< MeshType >( parameters ) )
            return EXIT_FAILURE;
      }
      if( dimensions == 3 )
      {
         typedef tnlGrid< 3, double, tnlHost, int > MeshType;
         if( ! processFiles< MeshType >( parameters ) )
            return EXIT_FAILURE;
      }
      return EXIT_SUCCESS;
   }
   if( parsedMeshType[ 0 ] == "tnlMesh" )
   {
      /*tnlString meshFile = parameters. getParameter< tnlString >( "mesh" );
      struct MeshConfig : public tnlMeshConfigBase< 2 >
      {
         typedef tnlMeshTriangleTopology CellType;
      };
      tnlMesh< MeshConfig > mesh;
      if( ! mesh.load( meshFile ) )
         return EXIT_FAILURE;
      if( ! tnlMeshWriterNetgen::writeMesh( "tnl-mesh.ng", mesh, true ) )
         return EXIT_FAILURE;*/
   }
   return EXIT_FAILURE;
}
