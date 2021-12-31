#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Meshes/Readers/VTKReader.h>
#include <TNL/Meshes/Writers/VTKWriter.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

#include "data/loader.h"
#include "MeshReaderTest.h"

using namespace TNL::Meshes;

static const char* TEST_FILE_NAME = "test_VTKReaderTest.vtk";

struct MyConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

// disable all grids
template< int Dimension, typename Real, typename Device, typename Index >
struct GridTag< MyConfigTag, Grid< Dimension, Real, Device, Index > >{ static constexpr bool enabled = false; };

// enable meshes used in the tests
//template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Edge > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Triangle > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Tetrahedron > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Polygon > { static constexpr bool enabled = true; };

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL

// TODO: test case for 1D mesh of edges

// ASCII data, produced by Gmsh
TEST( VTKReaderTest, mrizka_1 )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Triangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTKReader >( "triangles/mrizka_1.vtk" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 142 );
   EXPECT_EQ( cells, 242 );

   test_reader< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTKWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// ASCII data, produced by Gmsh
TEST( VTKReaderTest, tetrahedrons )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Tetrahedron > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTKReader >( "tetrahedrons/cube1m_1.vtk" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 395 );
   EXPECT_EQ( cells, 1312 );

   test_reader< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTKWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// binary data, produced by RF writer
TEST( VTKReaderTest, triangles_2x2x2_original_with_metadata_and_cell_data )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Triangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTKReader >( "triangles_2x2x2/original_with_metadata_and_cell_data.vtk" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 9 );
   EXPECT_EQ( cells, 8 );

   test_reader< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTKWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// ASCII data, produced by TNL writer
TEST( VTKReaderTest, triangles_2x2x2_minimized_ascii )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Triangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTKReader >( "triangles_2x2x2/minimized_ascii.vtk" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 9 );
   EXPECT_EQ( cells, 8 );

   test_reader< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTKWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// binary data, produced by TNL writer
TEST( VTKReaderTest, triangles_2x2x2_minimized_binary )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Triangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTKReader >( "triangles_2x2x2/minimized_binary.vtk" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 9 );
   EXPECT_EQ( cells, 8 );

   test_reader< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTKWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// ASCII data, produced by TNL writer
TEST( VTKReaderTest, polygons )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Polygon > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTKReader >( "polygons/unicorn.vtk" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 193 );
   EXPECT_EQ( cells, 90 );

   test_reader< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTKWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// TODO: test case for DataFile version 5.1: triangles_2x2x2/DataFile_version_5.1_exported_from_paraview.vtk
#endif

#include "../main.h"
