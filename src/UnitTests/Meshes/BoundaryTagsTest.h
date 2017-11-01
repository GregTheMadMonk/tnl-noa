#pragma once

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <vector>

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/MeshConfigBase.h>
#include <TNL/Meshes/Topologies/Quadrilateral.h>
#include <TNL/Meshes/MeshDetails/initializer/MeshInitializer.h>
#include <TNL/Meshes/MeshBuilder.h>

namespace BoundaryTagsTest {

using namespace TNL;
using namespace TNL::Meshes;

using RealType = double;
using Device = Devices::Host;
using IndexType = int;

class TestQuadrilateralMeshConfig : public MeshConfigBase< Topologies::Quadrilateral >
{
public:
   static constexpr bool entityStorage( int dimensions ) { return true; }
   template< typename EntityTopology > static constexpr bool subentityStorage( EntityTopology, int SubentityDimensions ) { return true; }
   template< typename EntityTopology > static constexpr bool subentityOrientationStorage( EntityTopology, int SubentityDimensions ) { return ( SubentityDimensions % 2 != 0 ); }
   template< typename EntityTopology > static constexpr bool superentityStorage( EntityTopology, int SuperentityDimensions ) { return true; }
   template< typename EntityTopology > static constexpr bool boundaryTagsStorage( EntityTopology ) { return true; }
};

TEST( MeshTest, RegularMeshOfQuadrilateralsTest )
{
   using QuadrilateralMeshEntityType = MeshEntity< TestQuadrilateralMeshConfig, Devices::Host, Topologies::Quadrilateral >;
   using EdgeMeshEntityType = typename QuadrilateralMeshEntityType::SubentityTraits< 1 >::SubentityType;
   using VertexMeshEntityType = typename QuadrilateralMeshEntityType::SubentityTraits< 0 >::SubentityType;

   using PointType = typename VertexMeshEntityType::PointType;
   ASSERT_TRUE( PointType::getType() == ( Containers::StaticVector< 2, RealType >::getType() ) );

   const IndexType xSize( 3 ), ySize( 4 );
   const RealType width( 1.0 ), height( 1.0 );
   const RealType hx( width / ( RealType ) xSize ),
                  hy( height / ( RealType ) ySize );
   const IndexType numberOfCells = xSize * ySize;
   const IndexType numberOfVertices = ( xSize + 1 ) * ( ySize + 1 );

   typedef Mesh< TestQuadrilateralMeshConfig > TestQuadrilateralMesh;
   TestQuadrilateralMesh mesh, mesh2;
   MeshBuilder< TestQuadrilateralMesh > meshBuilder;
   meshBuilder.setPointsCount( numberOfVertices );
   meshBuilder.setCellsCount( numberOfCells );

   /****
    * Setup vertices
    */
   for( IndexType j = 0; j <= ySize; j++ )
      for( IndexType i = 0; i <= xSize; i++ )
         meshBuilder.setPoint( j * ( xSize + 1 ) + i, PointType( i * hx, j * hy ) );

   /****
    * Setup cells
    */
   IndexType cellIdx( 0 );
   for( IndexType j = 0; j < ySize; j++ )
      for( IndexType i = 0; i < xSize; i++ )
      {
         const IndexType vertex0 = j * ( xSize + 1 ) + i;
         const IndexType vertex1 = j * ( xSize + 1 ) + i + 1;
         const IndexType vertex2 = ( j + 1 ) * ( xSize + 1 ) + i + 1;
         const IndexType vertex3 = ( j + 1 ) * ( xSize + 1 ) + i;

         meshBuilder.getCellSeed( cellIdx   ).setCornerId( 0, vertex0 );
         meshBuilder.getCellSeed( cellIdx   ).setCornerId( 1, vertex1 );
         meshBuilder.getCellSeed( cellIdx   ).setCornerId( 2, vertex2 );
         meshBuilder.getCellSeed( cellIdx++ ).setCornerId( 3, vertex3 );
      }

   ASSERT_TRUE( meshBuilder.build( mesh ) );

   std::vector< IndexType > boundaryCells = {0, 1, 2, 3, 5, 6, 8, 9, 10, 11};
   std::vector< IndexType > interiorCells = {4, 7};

   // Test boundary cells
   EXPECT_EQ( mesh.template getBoundaryEntitiesCount< 2 >(), boundaryCells.size() );
   for( size_t i = 0; i < boundaryCells.size(); i++ ) {
      EXPECT_TRUE( mesh.template isBoundaryEntity< 2 >( boundaryCells[ i ] ) );
      EXPECT_EQ( mesh.template getBoundaryEntityIndex< 2 >( i ), boundaryCells[ i ] );
   }
   // Test interior cells
   EXPECT_EQ( mesh.template getInteriorEntitiesCount< 2 >(), interiorCells.size() );
   for( size_t i = 0; i < interiorCells.size(); i++ ) {
      EXPECT_FALSE( mesh.template isBoundaryEntity< 2 >( interiorCells[ i ] ) );
      EXPECT_EQ( mesh.template getInteriorEntityIndex< 2 >( i ), interiorCells[ i ] );
   }

   std::vector< IndexType > boundaryFaces = {0, 3, 4, 7, 8, 12, 15, 19, 22, 25, 26, 28, 29, 30};
   std::vector< IndexType > interiorFaces = {1, 2, 5, 6, 9, 10, 11, 13, 14, 16, 17, 18, 20, 21, 23, 24, 27};

   // Test boundary faces
   EXPECT_EQ( mesh.template getBoundaryEntitiesCount< 1 >(), boundaryFaces.size() );
   for( size_t i = 0; i < boundaryFaces.size(); i++ ) {
      EXPECT_TRUE( mesh.template isBoundaryEntity< 1 >( boundaryFaces[ i ] ) );
      EXPECT_EQ( mesh.template getBoundaryEntityIndex< 1 >( i ), boundaryFaces[ i ] );
   }
   // Test interior faces
   EXPECT_EQ( mesh.template getInteriorEntitiesCount< 1 >(), interiorFaces.size() );
   for( size_t i = 0; i < interiorFaces.size(); i++ ) {
      EXPECT_FALSE( mesh.template isBoundaryEntity< 1 >( interiorFaces[ i ] ) );
      EXPECT_EQ( mesh.template getInteriorEntityIndex< 1 >( i ), interiorFaces[ i ] );
   }
}

} // namespace BoundaryTagsTest

#endif
