/***************************************************************************
                          tnl-grid-to-mesh.cpp  -  description
                             -------------------
    begin                : Oct 24, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Meshes/TypeResolver/TypeResolver.h>
#include <TNL/Meshes/Writers/VTKWriter.h>
#include <TNL/Meshes/Writers/VTUWriter.h>
#include <TNL/Meshes/Writers/NetgenWriter.h>

using namespace TNL;

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

// cannot be deduced from GridType
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
struct MeshCreator< Meshes::Grid< 1, Real, Device, Index > >
{
   using GridType = Meshes::Grid< 1, Real, Device, Index >;
   using CellTopology = Meshes::Topologies::Edge;
   using MeshConfig = Meshes::DefaultConfig< CellTopology,
                                                  CellTopology::dimension,
                                                  typename GridType::RealType,
                                                  typename GridType::GlobalIndexType,
                                                  LocalIndexType >;
   using MeshType = Meshes::Mesh< MeshConfig >;

   static bool run( const GridType& grid, MeshType& mesh )
   {
      const Index numberOfVertices = grid.template getEntitiesCount< typename GridType::Vertex >();
      const Index numberOfCells = grid.template getEntitiesCount< typename GridType::Cell >();

      Meshes::MeshBuilder< MeshType > meshBuilder;
      meshBuilder.setPointsCount( numberOfVertices );
      meshBuilder.setCellsCount( numberOfCells );

      for( Index i = 0; i < numberOfVertices; i++ ) {
         const auto vertex = grid.template getEntity< typename GridType::Vertex >( i );
         meshBuilder.setPoint( i, vertex.getCenter() );
      }

      for( Index i = 0; i < numberOfCells; i++ ) {
         auto cell = grid.template getEntity< typename GridType::Cell >( i );
         cell.refresh();
         const auto neighbors = cell.template getNeighborEntities< 0 >();
         meshBuilder.getCellSeed( i ).setCornerId( 0, neighbors.template getEntityIndex< -1 >() );
         meshBuilder.getCellSeed( i ).setCornerId( 1, neighbors.template getEntityIndex<  1 >() );
      }

      return meshBuilder.build( mesh );
   }
};

template< typename Real, typename Device, typename Index >
struct MeshCreator< Meshes::Grid< 2, Real, Device, Index > >
{
   using GridType = Meshes::Grid< 2, Real, Device, Index >;
   using CellTopology = Meshes::Topologies::Quadrangle;
   using MeshConfig = Meshes::DefaultConfig< CellTopology,
                                                  CellTopology::dimension,
                                                  typename GridType::RealType,
                                                  typename GridType::GlobalIndexType,
                                                  LocalIndexType >;
   using MeshType = Meshes::Mesh< MeshConfig >;

   static bool run( const GridType& grid, MeshType& mesh )
   {
      const Index numberOfVertices = grid.template getEntitiesCount< typename GridType::Vertex >();
      const Index numberOfCells = grid.template getEntitiesCount< typename GridType::Cell >();

      Meshes::MeshBuilder< MeshType > meshBuilder;
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
struct MeshCreator< Meshes::Grid< 3, Real, Device, Index > >
{
   using GridType = Meshes::Grid< 3, Real, Device, Index >;
   using CellTopology = Meshes::Topologies::Hexahedron;
   using MeshConfig = Meshes::DefaultConfig< CellTopology,
                                                  CellTopology::dimension,
                                                  typename GridType::RealType,
                                                  typename GridType::GlobalIndexType,
                                                  LocalIndexType >;
   using MeshType = Meshes::Mesh< MeshConfig >;

   static bool run( const GridType& grid, MeshType& mesh )
   {
      const Index numberOfVertices = grid.template getEntitiesCount< typename GridType::Vertex >();
      const Index numberOfCells = grid.template getEntitiesCount< typename GridType::Cell >();

      Meshes::MeshBuilder< MeshType > meshBuilder;
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
bool convertGrid( Grid& grid, const String& fileName, const String& outputFileName, const String& outputFormat )
{
   using MeshCreator = MeshCreator< Grid >;
   using Mesh = typename MeshCreator::MeshType;

   grid.load( fileName );

   Mesh mesh;
   if( ! MeshCreator::run( grid, mesh ) ) {
      std::cerr << "Unable to build mesh from grid." << std::endl;
      return false;
   }

   if( outputFormat == "vtk" ) {
      using VTKWriter = Meshes::Writers::VTKWriter< Mesh >;
      std::ofstream file( outputFileName.getString() );
      VTKWriter writer( file );
      writer.template writeEntities< Mesh::getMeshDimension() >( mesh );
   }
   else if( outputFormat == "vtu" ) {
      using VTKWriter = Meshes::Writers::VTUWriter< Mesh >;
      std::ofstream file( outputFileName.getString() );
      VTKWriter writer( file );
      writer.template writeEntities< Mesh::getMeshDimension() >( mesh );
   }
   else if( outputFormat == "netgen" ) {
      using NetgenWriter = Meshes::Writers::NetgenWriter< Mesh >;
      std::fstream file( outputFileName.getString() );
      NetgenWriter::writeMesh( mesh, file );
   }

   return true;
}

void configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "General settings:" );
   config.addRequiredEntry< String >( "input-file", "Input file with the mesh." );
   config.addRequiredEntry< String >( "output-file", "Output mesh file path." );
   config.addRequiredEntry< String >( "output-file-format", "Output mesh file format." );
   config.addEntryEnum( "tnl" );
   config.addEntryEnum( "vtk" );
   config.addEntryEnum( "vtu" );
   config.addEntryEnum( "netgen" );
}

int
main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   const String inputFileName = parameters.getParameter< String >( "input-file" );
   const String outputFileName = parameters.getParameter< String >( "output-file" );
   const String outputFileFormat = parameters.getParameter< String >( "output-file-format" );

   auto wrapper = [&] ( const auto& reader, auto&& grid )
   {
      return convertGrid( grid, inputFileName, outputFileName, outputFileFormat );
   };
   return ! Meshes::resolveMeshType< GridToMeshConfigTag, Devices::Host >( wrapper, inputFileName );
}
