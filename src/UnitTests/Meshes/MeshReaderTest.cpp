#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshConfigBase.h>
#include <TNL/Meshes/BuildConfigTags.h>
#include <TNL/Meshes/TypeResolver/TypeResolver.h>

#include <TNL/Debugging/MemoryUsage.h>

using namespace TNL;
using namespace TNL::Meshes;


template< typename Cell,
          int WorldDimension = Cell::dimension,
          typename Real = double,
          typename GlobalIndex = int,
          typename LocalIndex = GlobalIndex,
          typename Id = void >
struct MyMeshConfig
   : public MeshConfigBase< Cell, WorldDimension, Real, GlobalIndex, LocalIndex, Id >
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
      using BaseType = MeshConfigBase< Cell, WorldDimension, Real, GlobalIndex, LocalIndex, Id >;
      using FaceTopology = typename MeshSubtopology< Cell, BaseType::meshDimension - 1 >::Topology;
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
template<> struct MeshCellTopologyTag< MyBuildConfigTag, MeshEdgeTopology > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MyBuildConfigTag, MeshTriangleTopology > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MyBuildConfigTag, MeshQuadrilateralTopology > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MyBuildConfigTag, MeshTetrahedronTopology > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MyBuildConfigTag, MeshHexahedronTopology > { enum { enabled = true }; };

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
      MeshType mesh;

      std::cout << "pre-init\t";
      Debugging::printMemoryUsage();

      if( ! loadMesh( fileName, mesh ) )
         return false;

      // TODO: add tests
      std::cout << "NOTE: there is no real test, but the file was loaded fine..." << std::endl;

      std::cout << "vertices: " << mesh.template getEntitiesCount< 0 >() << std::endl;
      std::cout << "cells: " << mesh.template getEntitiesCount< MeshType::getMeshDimension() >() << std::endl;

      std::cout << "post-init\t";
      Debugging::printMemoryUsage();

      mesh.save( "mesh-test.tnl" );

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
