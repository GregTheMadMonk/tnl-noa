#include <TNL/Meshes/TypeResolver/TypeResolver.h>

struct GridToMeshConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

/****
 * Turn off support for float and long double.
 */
template<> struct GridRealTag< GridToMeshConfigTag, float > { enum { enabled = false }; };
template<> struct GridRealTag< GridToMeshConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct GridIndexTag< GridToMeshConfigTag, short int >{ enum { enabled = false }; };
template<> struct GridIndexTag< GridToMeshConfigTag, long int >{ enum { enabled = false }; };

/****
 * Unstructured meshes are disabled, only grids can be on input.
 */

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL


// FIXME: can't be deduced from GridType
using LocalIndexType = short int;

template< typename Mesh >
struct MeshCreator
{
   using MeshType = Mesh;

   static bool run( const Mesh& meshIn, Mesh& meshOut )
   {
      std::cerr << "Got a mesh on the input." << std::endl;
      return false;
   }
};

template< typename Real, typename Device, typename Index >
struct MeshCreator< TNL::Meshes::Grid< 1, Real, Device, Index > >
{
   using GridType = TNL::Meshes::Grid< 1, Real, Device, Index >;
   using CellTopology = TNL::Meshes::Topologies::Edge;
   using MeshConfig = TNL::Meshes::DefaultConfig< CellTopology,
                                                  CellTopology::dimension,
                                                  typename GridType::RealType,
                                                  typename GridType::GlobalIndexType,
                                                  LocalIndexType >;
   using MeshType = TNL::Meshes::Mesh< MeshConfig >;

   static bool run( const GridType& grid, MeshType& mesh )
   {
      const Index numberOfVertices = grid.template getEntitiesCount< typename GridType::Vertex >();
      const Index numberOfCells = grid.template getEntitiesCount< typename GridType::Cell >();

      TNL::Meshes::MeshBuilder< MeshType > meshBuilder;
      meshBuilder.setPointsCount( numberOfVertices );
      meshBuilder.setCellsCount( numberOfCells );

      for( Index i = 0; i < numberOfVertices; i++ ) {
         const auto vertex = grid.template getEntity< typename GridType::Vertex >( i );
         meshBuilder.setPoint( i, vertex.getCenter() );
      }

      for( Index i = 0; i < numberOfCells; i++ ) {
         const auto cell = grid.template getEntity< typename GridType::Cell >( i );
         const auto neighbors = cell.template getNeighborEntities< 0 >();
         meshBuilder.getCellSeed( i ).setCornerId( 0, neighbors.template getEntityIndex< -1 >() );
         meshBuilder.getCellSeed( i ).setCornerId( 1, neighbors.template getEntityIndex<  1 >() );
      }

      return meshBuilder.build( mesh );
   }
};

template< typename Real, typename Device, typename Index >
struct MeshCreator< TNL::Meshes::Grid< 2, Real, Device, Index > >
{
   using GridType = TNL::Meshes::Grid< 2, Real, Device, Index >;
   using CellTopology = TNL::Meshes::Topologies::Quadrilateral;
   using MeshConfig = TNL::Meshes::DefaultConfig< CellTopology,
                                                  CellTopology::dimension,
                                                  typename GridType::RealType,
                                                  typename GridType::GlobalIndexType,
                                                  LocalIndexType >;
   using MeshType = TNL::Meshes::Mesh< MeshConfig >;

   static bool run( const GridType& grid, MeshType& mesh )
   {
      const Index numberOfVertices = grid.template getEntitiesCount< typename GridType::Vertex >();
      const Index numberOfCells = grid.template getEntitiesCount< typename GridType::Cell >();

      TNL::Meshes::MeshBuilder< MeshType > meshBuilder;
      meshBuilder.setPointsCount( numberOfVertices );
      meshBuilder.setCellsCount( numberOfCells );

      for( Index i = 0; i < numberOfVertices; i++ ) {
         const auto vertex = grid.template getEntity< typename GridType::Vertex >( i );
         meshBuilder.setPoint( i, vertex.getCenter() );
      }

      for( Index i = 0; i < numberOfCells; i++ ) {
         const auto cell = grid.template getEntity< typename GridType::Cell >( i );
         const auto neighbors = cell.template getNeighborEntities< 0 >();
         meshBuilder.getCellSeed( i ).setCornerId( 0, neighbors.template getEntityIndex< -1, -1 >() );
         meshBuilder.getCellSeed( i ).setCornerId( 1, neighbors.template getEntityIndex<  1, -1 >() );
         meshBuilder.getCellSeed( i ).setCornerId( 2, neighbors.template getEntityIndex<  1,  1 >() );
         meshBuilder.getCellSeed( i ).setCornerId( 3, neighbors.template getEntityIndex< -1,  1 >() );
      }

      return meshBuilder.build( mesh );
   }
};

template< typename Real, typename Device, typename Index >
struct MeshCreator< TNL::Meshes::Grid< 3, Real, Device, Index > >
{
   using GridType = TNL::Meshes::Grid< 3, Real, Device, Index >;
   using CellTopology = TNL::Meshes::Topologies::Hexahedron;
   using MeshConfig = TNL::Meshes::DefaultConfig< CellTopology,
                                                  CellTopology::dimension,
                                                  typename GridType::RealType,
                                                  typename GridType::GlobalIndexType,
                                                  LocalIndexType >;
   using MeshType = TNL::Meshes::Mesh< MeshConfig >;

   static bool run( const GridType& grid, MeshType& mesh )
   {
      const Index numberOfVertices = grid.template getEntitiesCount< typename GridType::Vertex >();
      const Index numberOfCells = grid.template getEntitiesCount< typename GridType::Cell >();

      TNL::Meshes::MeshBuilder< MeshType > meshBuilder;
      meshBuilder.setPointsCount( numberOfVertices );
      meshBuilder.setCellsCount( numberOfCells );

      for( Index i = 0; i < numberOfVertices; i++ ) {
         const auto vertex = grid.template getEntity< typename GridType::Vertex >( i );
         meshBuilder.setPoint( i, vertex.getCenter() );
      }

      for( Index i = 0; i < numberOfCells; i++ ) {
         const auto cell = grid.template getEntity< typename GridType::Cell >( i );
         const auto neighbors = cell.template getNeighborEntities< 0 >();
         meshBuilder.getCellSeed( i ).setCornerId( 0, neighbors.template getEntityIndex< -1, -1, -1 >() );
         meshBuilder.getCellSeed( i ).setCornerId( 1, neighbors.template getEntityIndex<  1, -1, -1 >() );
         meshBuilder.getCellSeed( i ).setCornerId( 2, neighbors.template getEntityIndex<  1,  1, -1 >() );
         meshBuilder.getCellSeed( i ).setCornerId( 3, neighbors.template getEntityIndex< -1,  1, -1 >() );
         meshBuilder.getCellSeed( i ).setCornerId( 4, neighbors.template getEntityIndex< -1, -1,  1 >() );
         meshBuilder.getCellSeed( i ).setCornerId( 5, neighbors.template getEntityIndex<  1, -1,  1 >() );
         meshBuilder.getCellSeed( i ).setCornerId( 6, neighbors.template getEntityIndex<  1,  1,  1 >() );
         meshBuilder.getCellSeed( i ).setCornerId( 7, neighbors.template getEntityIndex< -1,  1,  1 >() );
      }

      return meshBuilder.build( mesh );
   }
};

template< typename Grid >
bool convertGrid( Grid& grid, const TNL::String& fileName, const TNL::String& outputFileName )
{
   using MeshCreator = MeshCreator< Grid >;
   using Mesh = typename MeshCreator::MeshType;

   grid.load( fileName );

   Mesh mesh;
   if( ! MeshCreator::run( grid, mesh ) ) {
      std::cerr << "Unable to build mesh from grid." << std::endl;
      return false;
   }

   try
   {
      mesh.save( outputFileName );
   }
   catch(...)
   {
      std::cerr << "Failed to save the mesh to file '" << outputFileName << "'." << std::endl;
      return false;
   }

   return true;
}

int
main( int argc, char* argv[] )
{
   using namespace TNL;

   if( argc < 3 ) {
      std::cerr << "Usage: " << argv[ 0 ] << " input-grid.tnl output-mesh.tnl" << std::endl;
      return EXIT_FAILURE;
   }

   String fileName( argv[ 1 ] );
   String outputFileName( argv[ 2 ] );

   auto wrapper = [&] ( const auto& reader, auto&& grid )
   {
      return convertGrid( grid, fileName, outputFileName );
   };
   return ! Meshes::resolveMeshType< GridToMeshConfigTag, Devices::Host >( wrapper, fileName );
}
