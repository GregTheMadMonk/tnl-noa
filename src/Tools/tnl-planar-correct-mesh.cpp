#include <TNL/Config/parseCommandLine.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <TNL/Meshes/Writers/VTKWriter.h>
#include <TNL/Meshes/Writers/FPMAWriter.h>
#include <TNL/Meshes/Geometry/getPlanarMesh.h>

using namespace TNL;

struct MeshPlanarCorrectConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

/****
 * Turn off all grids.
 */
template<> struct GridRealTag< MeshPlanarCorrectConfigTag, float > { enum { enabled = false }; };
template<> struct GridRealTag< MeshPlanarCorrectConfigTag, double > { enum { enabled = false }; };
template<> struct GridRealTag< MeshPlanarCorrectConfigTag, long double > { enum { enabled = false }; };

/****
 * Unstructured meshes.
 */
template<> struct MeshCellTopologyTag< MeshPlanarCorrectConfigTag, Topologies::Polygon > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MeshPlanarCorrectConfigTag, Topologies::Polyhedron > { enum { enabled = true }; };

// Meshes are enabled only for the space dimension equal to 3
template< typename CellTopology, int SpaceDimension >
struct MeshSpaceDimensionTag< MeshPlanarCorrectConfigTag, CellTopology, SpaceDimension >
{ enum { enabled = ( SpaceDimension == 3 ) }; };

// Meshes are enabled only for types explicitly listed below.
template<> struct MeshRealTag< MeshPlanarCorrectConfigTag, float > { enum { enabled = true }; };
template<> struct MeshRealTag< MeshPlanarCorrectConfigTag, double > { enum { enabled = true }; };
template<> struct MeshGlobalIndexTag< MeshPlanarCorrectConfigTag, long int > { enum { enabled = true }; };
template<> struct MeshGlobalIndexTag< MeshPlanarCorrectConfigTag, int > { enum { enabled = true }; };
template<> struct MeshLocalIndexTag< MeshPlanarCorrectConfigTag, short int > { enum { enabled = true }; };

// Config tag specifying the MeshConfig template to use.
template<>
struct MeshConfigTemplateTag< MeshPlanarCorrectConfigTag >
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

using namespace TNL::Meshes;

template< typename Mesh >
auto getPlanarMeshHelper( const Mesh& mesh, const String& decompositionType )
{
   using namespace TNL::Meshes;

   if( decompositionType[0] == 'c' ) {
      return getPlanarMesh< EntityDecomposerVersion::ConnectEdgesToCentroid >( mesh );
   }
   else { // decompositionType[0] == 'p'
      return getPlanarMesh< EntityDecomposerVersion::ConnectEdgesToPoint >( mesh );
   }
}

template< typename Topology >
struct PlanarMeshWriter;

template<>
struct PlanarMeshWriter< Topologies::Polygon >
{
   template< typename Mesh >
   static void exec( const Mesh& mesh, const String& outputFileName )
   {
      using Writer = Meshes::Writers::VTKWriter< Mesh >;
      std::ofstream file( outputFileName );
      Writer writer( file );
      writer.template writeEntities< Mesh::getMeshDimension() >( mesh );
   }
};

template<>
struct PlanarMeshWriter< Topologies::Polyhedron >
{
   template< typename Mesh >
   static void exec( const Mesh& mesh, const String& outputFileName )
   {
      using Writer = Meshes::Writers::FPMAWriter< Mesh >;
      std::ofstream file( outputFileName );
      Writer writer( file );
      writer.writeEntities( mesh );
   }
};

template< typename Mesh >
bool triangulateMesh( const Mesh& mesh, const String& outputFileName, const String& decompositionType )
{
   const auto planarMesh = getPlanarMeshHelper( mesh, decompositionType );
   using PlanarMesh = decltype( planarMesh );
   using CellTopology = typename PlanarMesh::Cell::EntityTopology;
   PlanarMeshWriter< CellTopology >::exec( planarMesh, outputFileName );
   return true;
}

void configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "General settings:" );
   config.addRequiredEntry< String >( "input-file", "Input file with the mesh." );
   config.addRequiredEntry< String >( "output-file", "Output mesh file path." );
   config.addRequiredEntry< String >( "decomposition-type", "Type of decomposition to use for non-planar polygons." );
   config.addEntryEnum( "c" );
   config.addEntryEnum( "p" );
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
   const String decompositionType = parameters.getParameter< String >( "decomposition-type" );

   auto wrapper = [&] ( auto& reader, auto&& mesh ) -> bool
   {
      return triangulateMesh( mesh, outputFileName, decompositionType );
   };
   return ! Meshes::resolveAndLoadMesh< MeshPlanarCorrectConfigTag, Devices::Host >( wrapper, inputFileName, inputFileFormat );
}
