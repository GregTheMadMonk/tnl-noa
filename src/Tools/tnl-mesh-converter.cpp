/***************************************************************************
                          tnl-mesh-converter.cpp  -  description
                             -------------------
    begin                : Oct 24, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <TNL/Meshes/Writers/VTKWriter.h>
#include <TNL/Meshes/Writers/VTUWriter.h>
//#include <TNL/Meshes/Writers/VTIWriter.h>
//#include <TNL/Meshes/Writers/NetgenWriter.h>

using namespace TNL;

struct MeshConverterConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

/****
 * Turn off support for float and long double.
 */
template<> struct GridRealTag< MeshConverterConfigTag, float > { enum { enabled = false }; };
template<> struct GridRealTag< MeshConverterConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct GridIndexTag< MeshConverterConfigTag, short int >{ enum { enabled = false }; };
template<> struct GridIndexTag< MeshConverterConfigTag, long int >{ enum { enabled = false }; };

/****
 * Unstructured meshes.
 */
template<> struct MeshCellTopologyTag< MeshConverterConfigTag, Topologies::Edge > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MeshConverterConfigTag, Topologies::Triangle > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MeshConverterConfigTag, Topologies::Quadrangle > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MeshConverterConfigTag, Topologies::Tetrahedron > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MeshConverterConfigTag, Topologies::Hexahedron > { enum { enabled = true }; };

// Meshes are enabled only for the space dimension equal to the cell dimension.
template< typename CellTopology, int SpaceDimension >
struct MeshSpaceDimensionTag< MeshConverterConfigTag, CellTopology, SpaceDimension >
{ enum { enabled = ( SpaceDimension == CellTopology::dimension ) }; };

// Meshes are enabled only for types explicitly listed below.
template<> struct MeshRealTag< MeshConverterConfigTag, float > { enum { enabled = true }; };
template<> struct MeshRealTag< MeshConverterConfigTag, double > { enum { enabled = true }; };
template<> struct MeshGlobalIndexTag< MeshConverterConfigTag, int > { enum { enabled = true }; };
template<> struct MeshGlobalIndexTag< MeshConverterConfigTag, long int > { enum { enabled = true }; };
template<> struct MeshLocalIndexTag< MeshConverterConfigTag, short int > { enum { enabled = true }; };

// Config tag specifying the MeshConfig template to use.
template<>
struct MeshConfigTemplateTag< MeshConverterConfigTag >
{
   template< typename Cell,
             int SpaceDimension = Cell::dimension,
             typename Real = double,
             typename GlobalIndex = int,
             typename LocalIndex = GlobalIndex >
   struct MeshConfig
   {
      using CellTopology = Cell;
      using RealType = Real;
      using GlobalIndexType = GlobalIndex;
      using LocalIndexType = LocalIndex;

      static constexpr int spaceDimension = SpaceDimension;
      static constexpr int meshDimension = Cell::dimension;

      template< typename EntityTopology >
      static constexpr bool subentityStorage( EntityTopology, int SubentityDimension )
      {
         return SubentityDimension == 0 && EntityTopology::dimension == meshDimension;
      }

      template< typename EntityTopology >
      static constexpr bool superentityStorage( EntityTopology, int SuperentityDimension )
      {
         return false;
      }

      template< typename EntityTopology >
      static constexpr bool entityTagsStorage( EntityTopology )
      {
         return false;
      }

      static constexpr bool dualGraphStorage()
      {
         return false;
      }
   };
};

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL


template< typename Mesh >
bool convertMesh( const Mesh& mesh, const String& inputFileName, const String& outputFileName, const String& outputFormat )
{
   if( outputFormat == "vtk" ) {
      using Writer = Meshes::Writers::VTKWriter< Mesh >;
      std::ofstream file( outputFileName );
      Writer writer( file );
      writer.template writeEntities< Mesh::getMeshDimension() >( mesh );
   }
   else if( outputFormat == "vtu" ) {
      using Writer = Meshes::Writers::VTUWriter< Mesh >;
      std::ofstream file( outputFileName );
      Writer writer( file );
      writer.template writeEntities< Mesh::getMeshDimension() >( mesh );
   }
   // FIXME: VTIWriter is not specialized for meshes
//   else if( outputFormat == "vti" ) {
//      using Writer = Meshes::Writers::VTIWriter< Mesh >;
//      std::ofstream file( outputFileName );
//      Writer writer( file );
//      writer.writeImageData( mesh );
//   }
   // FIXME: NetgenWriter is not specialized for grids
//   else if( outputFormat == "netgen" ) {
//      using NetgenWriter = Meshes::Writers::NetgenWriter< Mesh >;
//      std::fstream file( outputFileName );
//      NetgenWriter::writeMesh( mesh, file );
//   }

   return true;
}

void configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "General settings:" );
   config.addRequiredEntry< String >( "input-file", "Input file with the mesh." );
   config.addEntry< String >( "input-file-format", "Input mesh file format.", "auto" );
   config.addRequiredEntry< String >( "output-file", "Output mesh file path." );
   config.addRequiredEntry< String >( "output-file-format", "Output mesh file format." );
   config.addEntryEnum( "vtk" );
   config.addEntryEnum( "vtu" );
//   config.addEntryEnum( "vti" );
//   config.addEntryEnum( "netgen" );
}

int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   const String inputFileName = parameters.getParameter< String >( "input-file" );
   const String inputFileFormat = parameters.getParameter< String >( "input-file-format" );
   const String outputFileName = parameters.getParameter< String >( "output-file" );
   const String outputFileFormat = parameters.getParameter< String >( "output-file-format" );

   auto wrapper = [&] ( auto& reader, auto&& mesh ) -> bool
   {
      return convertMesh( mesh, inputFileName, outputFileName, outputFileFormat );
   };
   return ! Meshes::resolveAndLoadMesh< MeshConverterConfigTag, Devices::Host >( wrapper, inputFileName, inputFileFormat );
}
