#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Meshes/Readers/VTIReader.h>
#include <TNL/Meshes/Writers/VTIWriter.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

#include "MeshReaderTest.h"

using namespace TNL::Meshes;

static const char* TEST_FILE_NAME = "test_VTIReaderTest.vti";

struct MyConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

// enable all index types in the GridTypeResolver
template<> struct GridIndexTag< MyConfigTag, short int >{ enum { enabled = true }; };
template<> struct GridIndexTag< MyConfigTag, int >{ enum { enabled = true }; };
template<> struct GridIndexTag< MyConfigTag, long int >{ enum { enabled = true }; };

// disable float and long double (RealType is not stored in VTI and double is the default)
template<> struct GridRealTag< MyConfigTag, float > { enum { enabled = false }; };
template<> struct GridRealTag< MyConfigTag, long double > { enum { enabled = false }; };

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL

TEST( VTIReaderTest, Grid1D )
{
   using GridType = Grid< 1, double, TNL::Devices::Host, short int >;
   using PointType = GridType::PointType;
   using CoordinatesType = GridType::CoordinatesType;

   GridType grid;
   grid.setDomain( PointType( 1 ), PointType( 2 ) );
   grid.setDimensions( CoordinatesType( 10 ) );

   test_reader< Readers::VTIReader, Writers::VTIWriter >( grid, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTIWriter, MyConfigTag >( grid, TEST_FILE_NAME );
   test_meshfunction< Readers::VTIReader, Writers::VTIWriter >( grid, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTIReader, Writers::VTIWriter >( grid, TEST_FILE_NAME, "CellData" );
}

TEST( VTIReaderTest, Grid2D )
{
   using GridType = Grid< 2, double, TNL::Devices::Host, int >;
   using PointType = GridType::PointType;
   using CoordinatesType = GridType::CoordinatesType;

   GridType grid;
   grid.setDomain( PointType( 1, 2 ), PointType( 3, 4 ) );
   grid.setDimensions( CoordinatesType( 10, 20 ) );

   test_reader< Readers::VTIReader, Writers::VTIWriter >( grid, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTIWriter, MyConfigTag >( grid, TEST_FILE_NAME );
   test_meshfunction< Readers::VTIReader, Writers::VTIWriter >( grid, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTIReader, Writers::VTIWriter >( grid, TEST_FILE_NAME, "CellData" );
}

TEST( VTIReaderTest, Grid3D )
{
   using GridType = Grid< 3, double, TNL::Devices::Host, long int >;
   using PointType = GridType::PointType;
   using CoordinatesType = GridType::CoordinatesType;

   GridType grid;
   grid.setDomain( PointType( 1, 2, 3 ), PointType( 4, 5, 6 ) );
   grid.setDimensions( CoordinatesType( 10, 20, 30 ) );

   test_reader< Readers::VTIReader, Writers::VTIWriter >( grid, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTIWriter, MyConfigTag >( grid, TEST_FILE_NAME );
   test_meshfunction< Readers::VTIReader, Writers::VTIWriter >( grid, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTIReader, Writers::VTIWriter >( grid, TEST_FILE_NAME, "CellData" );
}
#endif

#include "../main.h"
