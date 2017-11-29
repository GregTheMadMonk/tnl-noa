#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/DefaultConfig.h>
#include <TNL/Meshes/BuildConfigTags.h>
#include <TNL/Meshes/TypeResolver/TypeResolver.h>

#ifdef HAVE_VTK
#include <TNL/Meshes/Readers/VTKReader_libvtk.h>
#endif

#include <TNL/Debugging/MemoryUsage.h>

#include "MeshTest.h"

// TODO: remove this after refactoring with clang-rename
#include "MeshEntityTest.h"
#include "BoundaryTagsTest.h"

using namespace TNL;
using namespace TNL::Meshes;


template< typename Cell,
          int WorldDimension = Cell::dimension,
          typename Real = double,
          typename GlobalIndex = int,
          typename LocalIndex = GlobalIndex,
          typename Id = void >
struct MyMeshConfig
   : public DefaultConfig< Cell, WorldDimension, Real, GlobalIndex, LocalIndex, Id >
{
   static constexpr bool entityStorage( int dimension )
   {
      return true;
//      return dimension == 0 || dimension == Cell::dimension;
   }

   template< typename EntityTopology >
   static constexpr bool subentityStorage( EntityTopology, int SubentityDimension )
   {
//      return entityStorage( EntityTopology::dimension );
      return entityStorage( EntityTopology::dimension ) &&
             SubentityDimension == 0;
   }

   template< typename EntityTopology >
   static constexpr bool subentityOrientationStorage( EntityTopology, int SubentityDimension ) 
   {
      return false;
   }

   template< typename EntityTopology >
   static constexpr bool superentityStorage( EntityTopology, int SuperentityDimension )
   {
//      return entityStorage( EntityTopology::dimension );
      return entityStorage( EntityTopology::dimension ) &&
             SuperentityDimension == Cell::dimension;
   }

   template< typename EntityTopology >
   static constexpr bool boundaryTagsStorage( EntityTopology )
   {
      using BaseType = DefaultConfig< Cell, WorldDimension, Real, GlobalIndex, LocalIndex, Id >;
      using FaceTopology = typename Topologies::Subtopology< Cell, BaseType::meshDimension - 1 >::Topology;
      return entityStorage( BaseType::meshDimension - 1 ) &&
             superentityStorage( FaceTopology(), BaseType::meshDimension ) &&
             ( EntityTopology::dimension >= BaseType::meshDimension - 1 || subentityStorage( FaceTopology(), EntityTopology::dimension ) );
      //return false;
   }
};


// specialization of BuildConfigTags
struct MyBuildConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

// disable grids
template< int Dimension, typename Real, typename Device, typename Index >
struct GridTag< MyBuildConfigTag, Grid< Dimension, Real, Device, Index > >
{ enum { enabled = false }; };

// enable all cell topologies
template<> struct MeshCellTopologyTag< MyBuildConfigTag, Topologies::Edge > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MyBuildConfigTag, Topologies::Triangle > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MyBuildConfigTag, Topologies::Quadrilateral > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MyBuildConfigTag, Topologies::Tetrahedron > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MyBuildConfigTag, Topologies::Hexahedron > { enum { enabled = true }; };

template< typename CellTopology, int WorldDimension >
struct MeshWorldDimensionTag< MyBuildConfigTag, CellTopology, WorldDimension >
{ enum { enabled = ( WorldDimension == CellTopology::dimension ) }; };

template< typename Real > struct MeshRealTag< MyBuildConfigTag, Real > { enum { enabled = false }; };
template<> struct MeshRealTag< MyBuildConfigTag, float > { enum { enabled = true }; };
template<> struct MeshRealTag< MyBuildConfigTag, double > { enum { enabled = true }; };

template< typename GlobalIndex > struct MeshGlobalIndexTag< MyBuildConfigTag, GlobalIndex > { enum { enabled = false }; };
template<> struct MeshGlobalIndexTag< MyBuildConfigTag, int > { enum { enabled = true }; };

template< typename LocalIndex > struct MeshLocalIndexTag< MyBuildConfigTag, LocalIndex > { enum { enabled = false }; };
template<> struct MeshLocalIndexTag< MyBuildConfigTag, short int > { enum { enabled = true }; };

template<>
struct MeshConfigTemplateTag< MyBuildConfigTag >
{
   template< typename Cell, int WorldDimension, typename Real, typename GlobalIndex, typename LocalIndex, typename Id >
   using MeshConfig = MyMeshConfig< Cell, WorldDimension, Real, GlobalIndex, LocalIndex, Id >;
};

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL


template< typename MeshType >
class MeshTester
{
public:
   static bool run( const String& fileName )
   {
      std::cout << "pre-init\t";
      Debugging::printMemoryUsage();

#ifdef HAVE_VTK
      MeshType mesh_libvtk;
      Readers::VTKReader_libvtk<> reader;
      if( ! reader.readMesh( fileName, mesh_libvtk ) )
         return false;

      std::cout << "libvtk vertices: " << mesh_libvtk.template getEntitiesCount< 0 >() << std::endl;
      std::cout << "libvtk faces: " << mesh_libvtk.template getEntitiesCount< MeshType::getMeshDimension() - 1 >() << std::endl;
      std::cout << "libvtk cells: " << mesh_libvtk.template getEntitiesCount< MeshType::getMeshDimension() >() << std::endl;
#endif

      MeshType mesh;
      if( ! loadMesh( fileName, mesh ) )
         return false;

      std::cout << "vertices: " << mesh.template getEntitiesCount< 0 >() << std::endl;
      std::cout << "faces: " << mesh.template getEntitiesCount< MeshType::getMeshDimension() - 1 >() << std::endl;
      std::cout << "cells: " << mesh.template getEntitiesCount< MeshType::getMeshDimension() >() << std::endl;

      std::cout << "post-init\t";
      Debugging::printMemoryUsage();

#ifdef HAVE_VTK
      std::cout << "mesh_libvtk == mesh: " << std::boolalpha << (mesh_libvtk == mesh) << std::endl;
      if( mesh_libvtk != mesh ) {
         std::cerr << "mesh_libvtk:\n" << mesh_libvtk << "\n\nmesh:\n" << mesh << std::endl;
         return false;
      }
#endif

#ifdef HAVE_GTEST 
      std::cout << "Running basic I/O tests..." << std::endl;
      MeshTest::testFinishedMesh( mesh );
#endif
//      mesh.save( "mesh-test.tnl" );

      return true;
   }
};

int main( int argc, char* argv[] )
{
   if( argc < 2 ) {
      std::cerr << "Usage: " << argv[ 0 ] << " filename.[tnl|ng|vtk] ..." << std::endl;
      return EXIT_FAILURE;
   }

   bool result = true;

   for( int i = 1; i < argc; i++ ) {
      String fileName = argv[ i ];
      
      result &= resolveMeshType< MyBuildConfigTag, Devices::Host, MeshTester >
                  ( fileName,
                    fileName  // passed to MeshTester::run
                  );
   }

   std::cout << "final\t";
   Debugging::printMemoryUsage();

   return ! result;
}
