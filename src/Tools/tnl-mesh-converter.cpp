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
template<> struct MeshCellTopologyTag< MeshConverterConfigTag, Topologies::Polygon > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MeshConverterConfigTag, Topologies::Tetrahedron > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MeshConverterConfigTag, Topologies::Hexahedron > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MeshConverterConfigTag, Topologies::Wedge > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MeshConverterConfigTag, Topologies::Pyramid > { enum { enabled = true }; };

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

      static constexpr bool subentityStorage( int entityDimension, int subentityDimension )
      {
         return subentityDimension == 0 && entityDimension == meshDimension;
      }

      static constexpr bool superentityStorage( int entityDimension, int superentityDimension )
      {
         return false;
      }

      static constexpr bool entityTagsStorage( int entityDimension )
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
bool convertMesh( const Mesh& mesh, const std::string& inputFileName, const std::string& outputFileName, const std::string& outputFormat )
{
   std::string format = outputFormat;
   if( outputFormat == "auto" ) {
      namespace fs = std::experimental::filesystem;
      format = fs::path( outputFileName ).extension();
      if( format.length() > 0 )
         // remove dot from the extension
         format = format.substr(1);
   }

   if( format == "vtk" ) {
      using Writer = Meshes::Writers::VTKWriter< Mesh >;
      std::ofstream file( outputFileName );
      Writer writer( file );
      writer.template writeEntities< Mesh::getMeshDimension() >( mesh );
      return true;
   }
   if( format == "vtu" ) {
      using Writer = Meshes::Writers::VTUWriter< Mesh >;
      std::ofstream file( outputFileName );
      Writer writer( file );
      writer.template writeEntities< Mesh::getMeshDimension() >( mesh );
      return true;
   }
   // FIXME: VTIWriter is not specialized for meshes
//   if( outputFormat == "vti" ) {
//      using Writer = Meshes::Writers::VTIWriter< Mesh >;
//      std::ofstream file( outputFileName );
//      Writer writer( file );
//      writer.writeImageData( mesh );
//      return true;
//   }
   // FIXME: NetgenWriter is not specialized for grids
//   if( outputFormat == "ng" ) {
//      using NetgenWriter = Meshes::Writers::NetgenWriter< Mesh >;
//      std::fstream file( outputFileName );
//      NetgenWriter::writeMesh( mesh, file );
//      return true;
//   }

   if( outputFormat == "auto" )
      std::cerr << "File '" << outputFileName << "' has unsupported format (based on the file extension): " << format << ".";
   else
      std::cerr << "Unsupported output file format: " << outputFormat << ".";
   std::cerr << " Supported formats are 'vtk' and 'vtu'." << std::endl;
   return false;
}

void configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "General settings:" );
   config.addRequiredEntry< std::string >( "input-file", "Input file with the mesh." );
   config.addEntry< std::string >( "input-file-format", "Input mesh file format.", "auto" );
   config.addRequiredEntry< std::string >( "output-file", "Output mesh file path." );
   config.addEntry< std::string >( "output-file-format", "Output mesh file format.", "auto" );
   config.addEntryEnum( "auto" );
   config.addEntryEnum( "vtk" );
   config.addEntryEnum( "vtu" );
//   config.addEntryEnum( "vti" );
//   config.addEntryEnum( "ng" );
}

int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   const std::string inputFileName = parameters.getParameter< std::string >( "input-file" );
   const std::string inputFileFormat = parameters.getParameter< std::string >( "input-file-format" );
   const std::string outputFileName = parameters.getParameter< std::string >( "output-file" );
   const std::string outputFileFormat = parameters.getParameter< std::string >( "output-file-format" );

   auto wrapper = [&] ( auto& reader, auto&& mesh ) -> bool
   {
      return convertMesh( mesh, inputFileName, outputFileName, outputFileFormat );
   };
   const bool status = Meshes::resolveAndLoadMesh< MeshConverterConfigTag, Devices::Host >( wrapper, inputFileName, inputFileFormat );
   return static_cast< int >( ! status );
}
