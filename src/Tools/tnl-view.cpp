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
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

struct TNLViewBuildConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

/****
 * Turn off support for float and long double.
 */
//template<> struct GridRealTag< TNLViewBuildConfigTag, float > { enum { enabled = false }; };
template<> struct GridRealTag< TNLViewBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct GridIndexTag< TNLViewBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct GridIndexTag< TNLViewBuildConfigTag, long int >{ enum { enabled = false }; };

/****
 * Unstructured meshes.
 */
template<> struct MeshCellTopologyTag< TNLViewBuildConfigTag, Topologies::Edge > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< TNLViewBuildConfigTag, Topologies::Triangle > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< TNLViewBuildConfigTag, Topologies::Tetrahedron > { enum { enabled = true }; };

// Meshes are enabled only for the world dimension equal to the cell dimension.
template< typename CellTopology, int WorldDimension >
struct MeshWorldDimensionTag< TNLViewBuildConfigTag, CellTopology, WorldDimension >
{ enum { enabled = ( WorldDimension == CellTopology::dimension ) }; };

// Meshes are enabled only for types explicitly listed below.
template<> struct MeshRealTag< TNLViewBuildConfigTag, float > { enum { enabled = false }; };
template<> struct MeshRealTag< TNLViewBuildConfigTag, double > { enum { enabled = true }; };
template<> struct MeshGlobalIndexTag< TNLViewBuildConfigTag, int > { enum { enabled = true }; };
template<> struct MeshGlobalIndexTag< TNLViewBuildConfigTag, long int > { enum { enabled = false }; };
template<> struct MeshLocalIndexTag< TNLViewBuildConfigTag, short int > { enum { enabled = true }; };

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL

void setupConfig( Config::ConfigDescription& config )
{
   config.addDelimiter( "General settings:" );
   config.addEntry        < String >( "mesh", "Mesh file.", "mesh.tnl" );
   config.addEntry        < String >( "mesh-format", "Mesh file format.", "auto" );
   config.addRequiredList < String >( "input-files", "Input files." );
//   config.addList         < String >( "output-files", "Output files." );
   config.addEntry        < bool >  ( "check-output-file", "If the output file already exists, do not recreate it.", false );

   config.addDelimiter( "Grid settings:");
//   config.addList         < double >( "level-lines", "List of level sets which will be drawn." );
//   config.addEntry        < int >   ( "output-x-size", "X size of the output." );
//   config.addEntry        < int >   ( "output-y-size", "Y size of the output." );
//   config.addEntry        < int >   ( "output-z-size", "Z size of the output." );
   config.addEntry        < String >( "output-format", "Output file format.", "gnuplot" );
      config.addEntryEnum< String > ( "gnuplot" );
      config.addEntryEnum< String > ( "vtk" );
      config.addEntryEnum< String > ( "vtu" );
   config.addEntry        < int >   ( "verbose", "Set the verbosity of the program.", 1 );
}

int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;
   setupConfig( conf_desc );
   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   const String meshFile = parameters.getParameter< String >( "mesh" );
   const String meshFileFormat = parameters.getParameter< String >( "mesh-format" );
   auto wrapper = [&] ( const auto& reader, auto&& mesh )
   {
      using MeshType = std::decay_t< decltype(mesh) >;
      return processFiles< MeshType >( parameters );
   };
   return ! TNL::Meshes::resolveMeshType< TNLViewBuildConfigTag, Devices::Host >( wrapper, meshFile, meshFileFormat );
}
