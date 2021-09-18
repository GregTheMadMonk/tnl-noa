#include <TNL/Config/parseCommandLine.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <TNL/Meshes/Writers/VTKWriter.h>
#include <TNL/Meshes/Writers/VTUWriter.h>
#include <TNL/Meshes/Geometry/getDecomposedMesh.h>

using namespace TNL;

struct MeshTriangulatorConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

/****
 * Turn off all grids.
 */
template<> struct GridRealTag< MeshTriangulatorConfigTag, float > { enum { enabled = false }; };
template<> struct GridRealTag< MeshTriangulatorConfigTag, double > { enum { enabled = false }; };
template<> struct GridRealTag< MeshTriangulatorConfigTag, long double > { enum { enabled = false }; };

/****
 * Unstructured meshes.
 */
template<> struct MeshCellTopologyTag< MeshTriangulatorConfigTag, Topologies::Polygon > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MeshTriangulatorConfigTag, Topologies::Polyhedron > { enum { enabled = true }; };

// Meshes are enabled only for the space dimension equal to the cell dimension.
template< typename CellTopology, int SpaceDimension >
struct MeshSpaceDimensionTag< MeshTriangulatorConfigTag, CellTopology, SpaceDimension >
{ enum { enabled = ( SpaceDimension == CellTopology::dimension ) }; };

// Meshes are enabled only for types explicitly listed below.
template<> struct MeshRealTag< MeshTriangulatorConfigTag, float > { enum { enabled = true }; };
template<> struct MeshRealTag< MeshTriangulatorConfigTag, double > { enum { enabled = true }; };
template<> struct MeshGlobalIndexTag< MeshTriangulatorConfigTag, long int > { enum { enabled = true }; };
template<> struct MeshGlobalIndexTag< MeshTriangulatorConfigTag, int > { enum { enabled = true }; };
template<> struct MeshLocalIndexTag< MeshTriangulatorConfigTag, short int > { enum { enabled = true }; };

// Config tag specifying the MeshConfig template to use.
template<>
struct MeshConfigTemplateTag< MeshTriangulatorConfigTag >
{
   template< typename Cell,
             int SpaceDimension = Cell::dimension,
             typename Real = float,
             typename GlobalIndex = int,
             typename LocalIndex = short int >
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
         return   (subentityDimension == 0 && entityDimension == meshDimension)
               || (subentityDimension == meshDimension - 1 && entityDimension == meshDimension )
               || (subentityDimension == 0 && entityDimension == meshDimension - 1 );
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
auto getDecomposedMeshHelper( const Mesh& mesh, const String& decompositionType )
{
   using namespace TNL::Meshes;

   if( decompositionType[0] == 'c' ) {
      if( decompositionType[1] == 'c' ) {
         return getDecomposedMesh< EntityDecomposerVersion::ConnectEdgesToCentroid,
                                   EntityDecomposerVersion::ConnectEdgesToCentroid >( mesh );
      }
      else { // decompositionType[1] == 'p'
         return getDecomposedMesh< EntityDecomposerVersion::ConnectEdgesToCentroid,
                                   EntityDecomposerVersion::ConnectEdgesToPoint >( mesh );
      }
   }
   else { // decompositionType[0] == 'p'
      if( decompositionType[1] == 'c' ) {
         return getDecomposedMesh< EntityDecomposerVersion::ConnectEdgesToPoint,
                                   EntityDecomposerVersion::ConnectEdgesToCentroid >( mesh );
      }
      else { // decompositionType[1] == 'p'
         return getDecomposedMesh< EntityDecomposerVersion::ConnectEdgesToPoint,
                                   EntityDecomposerVersion::ConnectEdgesToPoint >( mesh );
      }
   }
}

template< typename Mesh >
bool triangulateMesh( const Mesh& mesh, const String& outputFileName, const String& outputFormat, const String& decompositionType )
{
   const auto decomposedMesh = getDecomposedMeshHelper( mesh, decompositionType );

   if( outputFormat == "vtk" ) {
      using Writer = Meshes::Writers::VTKWriter< decltype( decomposedMesh ) >;
      std::ofstream file( outputFileName );
      Writer writer( file );
      writer.template writeEntities< Mesh::getMeshDimension() >( decomposedMesh );
   }
   else if( outputFormat == "vtu" ) {
      using Writer = Meshes::Writers::VTUWriter< decltype( decomposedMesh ) >;
      std::ofstream file( outputFileName );
      Writer writer( file );
      writer.template writeEntities< Mesh::getMeshDimension() >( decomposedMesh );
   }

   return true;
}

void configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "General settings:" );
   config.addRequiredEntry< String >( "input-file", "Input file with the mesh." );
   config.addRequiredEntry< String >( "output-file", "Output mesh file path." );
   config.addRequiredEntry< String >( "output-file-format", "Output mesh file format." );
   config.addEntryEnum( "vtk" );
   config.addEntryEnum( "vtu" );
   config.addRequiredEntry< String >( "decomposition-type", "Type of decomposition to use." );
   config.addEntryEnum( "cc" );
   config.addEntryEnum( "cp" );
   config.addEntryEnum( "pc" );
   config.addEntryEnum( "pp" );
}

int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   const String inputFileName = parameters.getParameter< String >( "input-file" );
   const String inputFileFormat = "auto";
   const String outputFileName = parameters.getParameter< String >( "output-file" );
   const String outputFileFormat = parameters.getParameter< String >( "output-file-format" );
   const String decompositionType = parameters.getParameter< String >( "decomposition-type" );

   auto wrapper = [&] ( auto& reader, auto&& mesh ) -> bool
   {
      return triangulateMesh( mesh, outputFileName, outputFileFormat, decompositionType );
   };
   return ! Meshes::resolveAndLoadMesh< MeshTriangulatorConfigTag, Devices::Host >( wrapper, inputFileName, inputFileFormat );
}
