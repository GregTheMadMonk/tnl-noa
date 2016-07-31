/***************************************************************************
                          tnlMeshEntityTester.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLMESHENTITYTESTER_H_
#define TNLMESHENTITYTESTER_H_

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <TNL/mesh/tnlMeshEntity.h>
#include <TNL/mesh/config/tnlMeshConfigBase.h>
#include <TNL/mesh/topologies/tnlMeshVertexTopology.h>
#include <TNL/mesh/topologies/tnlMeshEdgeTopology.h>
#include <TNL/mesh/topologies/tnlMeshTriangleTopology.h>
#include <TNL/mesh/topologies/tnlMeshTetrahedronTopology.h>
 
using namespace TNL;

//typedef tnlMeshConfigBase< tnlMeshTriangleTopology, 2, double, int, int, void > TestTriangleEntityTopology;
typedef tnlMeshConfigBase< tnlMeshEdgeTopology, 2, double, int, int, void > TestEdgeEntityTopology;
typedef tnlMeshConfigBase< tnlMeshVertexTopology, 2, double, int, int, void > TestVertexEntityTopology;

class TestTriangleMeshConfig : public tnlMeshConfigBase< tnlMeshTriangleTopology >
{
   public:
 
      template< typename MeshEntity >
      static constexpr bool subentityStorage( MeshEntity entity, int subentityDimensions )
      {
         return true;
      }
 
      template< typename MeshEntity >
      static constexpr bool superentityStorage( MeshEntity entity, int superentityDimensions )
      {
         return true;
      }
};

class TestTetrahedronMeshConfig : public tnlMeshConfigBase< tnlMeshTetrahedronTopology >
{
   public:
 
      template< typename MeshEntity >
      static constexpr bool subentityStorage( MeshEntity entity, int subentityDimensions )
      {
         return true;
      }
 
      template< typename MeshEntity >
      static constexpr bool superentityStorage( MeshEntity entity, int superentityDimensions )
      {
         return true;
      }
};

template< typename RealType, typename Device, typename IndexType >
class tnlMeshEntityTester : public CppUnit :: TestCase
{
   public:
   typedef tnlMeshEntityTester< RealType, Device, IndexType > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;

   tnlMeshEntityTester(){};

   virtual
   ~tnlMeshEntityTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlMeshEntityTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "vertexMeshEntityTest", &TesterType::vertexMeshEntityTest ) );
      suiteOfTests -> addTest( new TestCallerType( "edgeMeshEntityTest", &TesterType::edgeMeshEntityTest ) );
      suiteOfTests -> addTest( new TestCallerType( "triangleMeshEntityTest", &TesterType::triangleMeshEntityTest ) );
      suiteOfTests -> addTest( new TestCallerType( "tetrahedronMeshEntityTest", &TesterType::tetrahedronMeshEntityTest ) );
      suiteOfTests -> addTest( new TestCallerType( "twoTrianglesTest", &TesterType::twoTrianglesTest ) );

      return suiteOfTests;
   }
 

   void vertexMeshEntityTest()
   {
      typedef tnlMeshConfigBase< tnlMeshEdgeTopology, 2, RealType, IndexType, IndexType, void > TestEntityTopology;
      typedef tnlMeshEntity< TestEntityTopology, tnlMeshVertexTopology > VertexMeshEntityType;
      typedef typename VertexMeshEntityType::PointType PointType;

      CPPUNIT_ASSERT( PointType::getType() == ( StaticVector< 2, RealType >::getType() ) );
      VertexMeshEntityType vertexEntity;
      PointType point;

      point.x() = 1.0;
      point.y() = 2.0;
      vertexEntity.setPoint( point );
      CPPUNIT_ASSERT( vertexEntity.getPoint() == point );
   }

   void edgeMeshEntityTest()
   {
      typedef tnlMeshEntity< TestEdgeEntityTopology, tnlMeshEdgeTopology > EdgeMeshEntityType;
      typedef tnlMeshEntity< TestEdgeEntityTopology, tnlMeshVertexTopology > VertexMeshEntityType;
 
      typedef typename VertexMeshEntityType::PointType PointType;
      CPPUNIT_ASSERT( PointType::getType() == ( StaticVector< 2, RealType >::getType() ) );

      /****
       *
       * Here we test the following simple example:
       *
 
                point2
                   |\
                   | \
                   |  \
                   |   \
 
                     ....
                edge1     edge0
                     ....


                   |                 \
                   |                  \
                   ---------------------
                point0   edge2        point1

       */
 
      PointType point0( 0.0, 0.0 ),
                point1( 1.0, 0.0 ),
                point2( 0.0, 1.0 );
 
      tnlStaticArray< 3, VertexMeshEntityType > vertexEntities;
      vertexEntities[ 0 ].setPoint( point0 );
      vertexEntities[ 1 ].setPoint( point1 );
      vertexEntities[ 2 ].setPoint( point2 );

      CPPUNIT_ASSERT( vertexEntities[ 0 ].getPoint() == point0 );
      CPPUNIT_ASSERT( vertexEntities[ 1 ].getPoint() == point1 );
      CPPUNIT_ASSERT( vertexEntities[ 2 ].getPoint() == point2 );

      tnlStaticArray< 3, EdgeMeshEntityType > edgeEntities;
      edgeEntities[ 0 ].setVertexIndex( 0, 0 );
      edgeEntities[ 0 ].setVertexIndex( 1, 1 );
      edgeEntities[ 1 ].setVertexIndex( 0, 1 );
      edgeEntities[ 1 ].setVertexIndex( 1, 2 );
      edgeEntities[ 2 ].setVertexIndex( 0, 2 );
      edgeEntities[ 2 ].setVertexIndex( 1, 0 );
      edgeEntities[ 0 ].setId( 0 );
      edgeEntities[ 1 ].setId( 1 );
      edgeEntities[ 2 ].setId( 2 );

      CPPUNIT_ASSERT( vertexEntities[ edgeEntities[ 0 ].getVertexIndex( 0 ) ].getPoint() == point0 );
      CPPUNIT_ASSERT( vertexEntities[ edgeEntities[ 0 ].getVertexIndex( 1 ) ].getPoint() == point1 );
      CPPUNIT_ASSERT( vertexEntities[ edgeEntities[ 1 ].getVertexIndex( 0 ) ].getPoint() == point1 );
      CPPUNIT_ASSERT( vertexEntities[ edgeEntities[ 1 ].getVertexIndex( 1 ) ].getPoint() == point2 );
      CPPUNIT_ASSERT( vertexEntities[ edgeEntities[ 2 ].getVertexIndex( 0 ) ].getPoint() == point2 );
      CPPUNIT_ASSERT( vertexEntities[ edgeEntities[ 2 ].getVertexIndex( 1 ) ].getPoint() == point0 );
   }

   void triangleMeshEntityTest()
   {
      typedef tnlMeshEntity< TestTriangleMeshConfig, tnlMeshTriangleTopology > TriangleMeshEntityType;

      static_assert( TriangleMeshEntityType::SubentityTraits< 1 >::storageEnabled, "Testing triangular mesh does not store edges as required." );
      static_assert( TriangleMeshEntityType::SubentityTraits< 0 >::storageEnabled, "" );
      typedef tnlMeshEntity< TestEdgeEntityTopology, tnlMeshEdgeTopology > EdgeMeshEntityType;
      typedef tnlMeshEntity< TestVertexEntityTopology, tnlMeshVertexTopology > VertexMeshEntityType;
      typedef typename VertexMeshEntityType::PointType PointType;
      CPPUNIT_ASSERT( PointType::getType() == ( StaticVector< 2, RealType >::getType() ) );

      /****
       * We set-up the same situation as in the test above
       */
      PointType point0( 0.0, 0.0 ),
                point1( 1.0, 0.0 ),
                point2( 0.0, 1.0 );

      tnlStaticArray< 3, VertexMeshEntityType > vertexEntities;
      vertexEntities[ 0 ].setPoint( point0 );
      vertexEntities[ 1 ].setPoint( point1 );
      vertexEntities[ 2 ].setPoint( point2 );

      CPPUNIT_ASSERT( vertexEntities[ 0 ].getPoint() == point0 );
      CPPUNIT_ASSERT( vertexEntities[ 1 ].getPoint() == point1 );
      CPPUNIT_ASSERT( vertexEntities[ 2 ].getPoint() == point2 );

      tnlStaticArray< 3, EdgeMeshEntityType > edgeEntities;
      edgeEntities[ 0 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTriangleTopology, tnlMeshEdgeTopology, 0, 0 >::index );
      edgeEntities[ 0 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTriangleTopology, tnlMeshEdgeTopology, 0, 1 >::index );
      edgeEntities[ 1 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTriangleTopology, tnlMeshEdgeTopology, 1, 0 >::index );
      edgeEntities[ 1 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTriangleTopology, tnlMeshEdgeTopology, 1, 1 >::index );
      edgeEntities[ 2 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTriangleTopology, tnlMeshEdgeTopology, 2, 0 >::index );
      edgeEntities[ 2 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTriangleTopology, tnlMeshEdgeTopology, 2, 1 >::index );

      CPPUNIT_ASSERT( edgeEntities[ 0 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTriangleTopology, tnlMeshEdgeTopology, 0, 0 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 0 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTriangleTopology, tnlMeshEdgeTopology, 0, 1 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 1 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTriangleTopology, tnlMeshEdgeTopology, 1, 0 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 1 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTriangleTopology, tnlMeshEdgeTopology, 1, 1 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 2 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTriangleTopology, tnlMeshEdgeTopology, 2, 0 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 2 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTriangleTopology, tnlMeshEdgeTopology, 2, 1 >::index ) );

      TriangleMeshEntityType triangleEntity;

      triangleEntity.template setSubentityIndex< 0 >( 0 , 0 );
      triangleEntity.template setSubentityIndex< 0 >( 1 , 1 );
      triangleEntity.template setSubentityIndex< 0 >( 2 , 2 );

      CPPUNIT_ASSERT( triangleEntity.template getSubentityIndex< 0 >( 0 ) == 0 );
      CPPUNIT_ASSERT( triangleEntity.template getSubentityIndex< 0 >( 1 ) == 1 );
      CPPUNIT_ASSERT( triangleEntity.template getSubentityIndex< 0 >( 2 ) == 2 );

      triangleEntity.template setSubentityIndex< 1 >( 0 , 0 );
      triangleEntity.template setSubentityIndex< 1 >( 1 , 1 );
      triangleEntity.template setSubentityIndex< 1 >( 2 , 2 );

      CPPUNIT_ASSERT( triangleEntity.template getSubentityIndex< 1 >( 0 ) == 0 );
      CPPUNIT_ASSERT( triangleEntity.template getSubentityIndex< 1 >( 1 ) == 1 );
      CPPUNIT_ASSERT( triangleEntity.template getSubentityIndex< 1 >( 2 ) == 2 );
   };

   void tetrahedronMeshEntityTest()
   {
      //typedef tnlMeshConfigBase< tnlMeshTetrahedronTopology, 3, RealType, IndexType, IndexType, void > TestTetrahedronEntityTopology;
      typedef tnlMeshConfigBase< tnlMeshTriangleTopology, 3, RealType, IndexType, IndexType, void > TestTriangleEntityTopology;
      typedef tnlMeshConfigBase< tnlMeshEdgeTopology, 3, RealType, IndexType, IndexType, void > TestEdgeEntityTopology;
      typedef tnlMeshConfigBase< tnlMeshVertexTopology, 3, RealType, IndexType, IndexType, void > TestVertexEntityTopology;

      typedef tnlMeshEntity< TestTetrahedronMeshConfig, tnlMeshTetrahedronTopology > TetrahedronMeshEntityType;
      typedef tnlMeshEntity< TestTriangleMeshConfig, tnlMeshTriangleTopology > TriangleMeshEntityType;
      typedef tnlMeshEntity< TestEdgeEntityTopology, tnlMeshEdgeTopology > EdgeMeshEntityType;
      typedef tnlMeshEntity< TestVertexEntityTopology, tnlMeshVertexTopology > VertexMeshEntityType;
      typedef typename VertexMeshEntityType::PointType PointType;
      CPPUNIT_ASSERT( PointType::getType() == ( StaticVector< 3, RealType >::getType() ) );

      /****
       * We set-up similar situation as above but with
       * tetrahedron.
       */
      PointType point0( 0.0, 0.0, 0.0),
                point1( 1.0, 0.0, 0.0 ),
                point2( 0.0, 1.0, 0.0 ),
                point3( 0.0, 0.0, 1.0 );
 
      tnlStaticArray< tnlMeshSubtopology< tnlMeshTetrahedronTopology, 0 >::count,
                      VertexMeshEntityType > vertexEntities;

      vertexEntities[ 0 ].setPoint( point0 );
      vertexEntities[ 1 ].setPoint( point1 );
      vertexEntities[ 2 ].setPoint( point2 );
      vertexEntities[ 3 ].setPoint( point3 );

      CPPUNIT_ASSERT( vertexEntities[ 0 ].getPoint() == point0 );
      CPPUNIT_ASSERT( vertexEntities[ 1 ].getPoint() == point1 );
      CPPUNIT_ASSERT( vertexEntities[ 2 ].getPoint() == point2 );
      CPPUNIT_ASSERT( vertexEntities[ 3 ].getPoint() == point3 );

      tnlStaticArray< tnlMeshSubtopology< tnlMeshTetrahedronTopology, 1 >::count,
                      EdgeMeshEntityType > edgeEntities;
      edgeEntities[ 0 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 0, 0 >::index );
      edgeEntities[ 0 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 0, 1 >::index );
      edgeEntities[ 1 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 1, 0 >::index );
      edgeEntities[ 1 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 1, 1 >::index );
      edgeEntities[ 2 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 2, 0 >::index );
      edgeEntities[ 2 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 2, 1 >::index );
      edgeEntities[ 3 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 3, 0 >::index );
      edgeEntities[ 3 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 3, 1 >::index );
      edgeEntities[ 4 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 4, 0 >::index );
      edgeEntities[ 4 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 4, 1 >::index );
      edgeEntities[ 5 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 5, 0 >::index );
      edgeEntities[ 5 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 5, 1 >::index );

      CPPUNIT_ASSERT( edgeEntities[ 0 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 0, 0 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 0 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 0, 1 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 1 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 1, 0 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 1 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 1, 1 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 2 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 2, 0 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 2 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 2, 1 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 3 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 3, 0 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 3 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 3, 1 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 4 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 4, 0 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 4 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 4, 1 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 5 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 5, 0 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 5 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 5, 1 >::index ) );

      tnlStaticArray< tnlMeshSubtopology< tnlMeshTetrahedronTopology, 2 >::count,
                      TriangleMeshEntityType > triangleEntities;
      triangleEntities[ 0 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 0, 0 >::index );
      triangleEntities[ 0 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 0, 1 >::index );
      triangleEntities[ 0 ].setVertexIndex( 2, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 0, 2 >::index );
      triangleEntities[ 1 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 1, 0 >::index );
      triangleEntities[ 1 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 1, 1 >::index );
      triangleEntities[ 1 ].setVertexIndex( 2, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 1, 2 >::index );
      triangleEntities[ 2 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 2, 0 >::index );
      triangleEntities[ 2 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 2, 1 >::index );
      triangleEntities[ 2 ].setVertexIndex( 2, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 2, 2 >::index );
      triangleEntities[ 3 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 3, 0 >::index );
      triangleEntities[ 3 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 3, 1 >::index );
      triangleEntities[ 3 ].setVertexIndex( 2, tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 3, 2 >::index );

      CPPUNIT_ASSERT( triangleEntities[ 0 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 0, 0 >::index ) );
      CPPUNIT_ASSERT( triangleEntities[ 0 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 0, 1 >::index ) );
      CPPUNIT_ASSERT( triangleEntities[ 0 ].getVertexIndex( 2 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 0, 2 >::index ) );
      CPPUNIT_ASSERT( triangleEntities[ 1 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 1, 0 >::index ) );
      CPPUNIT_ASSERT( triangleEntities[ 1 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 1, 1 >::index ) );
      CPPUNIT_ASSERT( triangleEntities[ 1 ].getVertexIndex( 2 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 1, 2 >::index ) );
      CPPUNIT_ASSERT( triangleEntities[ 2 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 2, 0 >::index ) );
      CPPUNIT_ASSERT( triangleEntities[ 2 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 2, 1 >::index ) );
      CPPUNIT_ASSERT( triangleEntities[ 2 ].getVertexIndex( 2 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 2, 2 >::index ) );

      TetrahedronMeshEntityType tetrahedronEntity;
      tetrahedronEntity.setVertexIndex( 0, 0 );
      tetrahedronEntity.setVertexIndex( 1, 1 );
      tetrahedronEntity.setVertexIndex( 2, 2 );
      tetrahedronEntity.setVertexIndex( 3, 3 );

      CPPUNIT_ASSERT( tetrahedronEntity.getVertexIndex( 0 ) == 0 );
      CPPUNIT_ASSERT( tetrahedronEntity.getVertexIndex( 1 ) == 1 );
      CPPUNIT_ASSERT( tetrahedronEntity.getVertexIndex( 2 ) == 2 );
      CPPUNIT_ASSERT( tetrahedronEntity.getVertexIndex( 3 ) == 3 );

      tetrahedronEntity.template setSubentityIndex< 2 >( 0, 0 );
      tetrahedronEntity.template setSubentityIndex< 2 >( 1, 1 );
      tetrahedronEntity.template setSubentityIndex< 2 >( 2, 2 );
      tetrahedronEntity.template setSubentityIndex< 2 >( 3, 3 );

      CPPUNIT_ASSERT( tetrahedronEntity.template getSubentityIndex< 2 >( 0 ) == 0 );
      CPPUNIT_ASSERT( tetrahedronEntity.template getSubentityIndex< 2 >( 1 ) == 1 );
      CPPUNIT_ASSERT( tetrahedronEntity.template getSubentityIndex< 2 >( 2 ) == 2 );
      CPPUNIT_ASSERT( tetrahedronEntity.template getSubentityIndex< 2 >( 3 ) == 3 );

      tetrahedronEntity.template setSubentityIndex< 1 >( 0, 0 );
      tetrahedronEntity.template setSubentityIndex< 1 >( 1, 1 );
      tetrahedronEntity.template setSubentityIndex< 1 >( 2, 2 );
      tetrahedronEntity.template setSubentityIndex< 1 >( 3, 3 );
      tetrahedronEntity.template setSubentityIndex< 1 >( 4, 4 );
      tetrahedronEntity.template setSubentityIndex< 1 >( 5, 5 );

      CPPUNIT_ASSERT( tetrahedronEntity.template getSubentityIndex< 1 >( 0 ) == 0 );
      CPPUNIT_ASSERT( tetrahedronEntity.template getSubentityIndex< 1 >( 1 ) == 1 );
      CPPUNIT_ASSERT( tetrahedronEntity.template getSubentityIndex< 1 >( 2 ) == 2 );
      CPPUNIT_ASSERT( tetrahedronEntity.template getSubentityIndex< 1 >( 3 ) == 3 );
      CPPUNIT_ASSERT( tetrahedronEntity.template getSubentityIndex< 1 >( 4 ) == 4 );
      CPPUNIT_ASSERT( tetrahedronEntity.template getSubentityIndex< 1 >( 5 ) == 5 );
   };

   void twoTrianglesTest()
   {

       typedef tnlMeshEntity< TestTriangleMeshConfig, tnlMeshTriangleTopology > TriangleMeshEntityType;
       typedef tnlMeshEntity< TestTriangleMeshConfig, tnlMeshEdgeTopology > EdgeMeshEntityType;
       typedef tnlMeshEntity< TestTriangleMeshConfig, tnlMeshVertexTopology > VertexMeshEntityType;
       typedef typename VertexMeshEntityType::PointType PointType;
       CPPUNIT_ASSERT( PointType::getType() == ( StaticVector< 2, RealType >::getType() ) );

       /****
        * We set-up the following situation
                point2   edge3       point3
                   |\-------------------|
                   | \                  |
                   |  \   triangle1     |
                   |   \                |

                          ....
                edge1     edge0        edge4
                          ....


                   |   triangle0     \  |
                   |                  \ |
                   ---------------------|
                point0   edge2        point1
        */

       PointType point0( 0.0, 0.0 ),
                 point1( 1.0, 0.0 ),
                 point2( 0.0, 1.0 ),
                 point3( 1.0, 1.0 );

       tnlStaticArray< 4, VertexMeshEntityType > vertexEntities;
       vertexEntities[ 0 ].setPoint( point0 );
       vertexEntities[ 1 ].setPoint( point1 );
       vertexEntities[ 2 ].setPoint( point2 );
       vertexEntities[ 3 ].setPoint( point3 );

       CPPUNIT_ASSERT( vertexEntities[ 0 ].getPoint() == point0 );
       CPPUNIT_ASSERT( vertexEntities[ 1 ].getPoint() == point1 );
       CPPUNIT_ASSERT( vertexEntities[ 2 ].getPoint() == point2 );
       CPPUNIT_ASSERT( vertexEntities[ 3 ].getPoint() == point3 );

       tnlStaticArray< 5, EdgeMeshEntityType > edgeEntities;
       edgeEntities[ 0 ].setVertexIndex( 0, 1 );
       edgeEntities[ 0 ].setVertexIndex( 1, 2 );
       edgeEntities[ 1 ].setVertexIndex( 0, 2 );
       edgeEntities[ 1 ].setVertexIndex( 1, 0 );
       edgeEntities[ 2 ].setVertexIndex( 0, 0 );
       edgeEntities[ 2 ].setVertexIndex( 1, 1 );
       edgeEntities[ 3 ].setVertexIndex( 0, 2 );
       edgeEntities[ 3 ].setVertexIndex( 1, 3 );
       edgeEntities[ 4 ].setVertexIndex( 0, 3 );
       edgeEntities[ 4 ].setVertexIndex( 1, 1 );

       CPPUNIT_ASSERT( edgeEntities[ 0 ].getVertexIndex( 0 ) == 1 );
       CPPUNIT_ASSERT( edgeEntities[ 0 ].getVertexIndex( 1 ) == 2 );
       CPPUNIT_ASSERT( edgeEntities[ 1 ].getVertexIndex( 0 ) == 2 );
       CPPUNIT_ASSERT( edgeEntities[ 1 ].getVertexIndex( 1 ) == 0 );
       CPPUNIT_ASSERT( edgeEntities[ 2 ].getVertexIndex( 0 ) == 0 );
       CPPUNIT_ASSERT( edgeEntities[ 2 ].getVertexIndex( 1 ) == 1 );
       CPPUNIT_ASSERT( edgeEntities[ 3 ].getVertexIndex( 0 ) == 2 );
       CPPUNIT_ASSERT( edgeEntities[ 3 ].getVertexIndex( 1 ) == 3 );
       CPPUNIT_ASSERT( edgeEntities[ 4 ].getVertexIndex( 0 ) == 3 );
       CPPUNIT_ASSERT( edgeEntities[ 4 ].getVertexIndex( 1 ) == 1 );

       tnlStaticArray< 2, TriangleMeshEntityType > triangleEntities;

       triangleEntities[ 0 ].template setSubentityIndex< 0 >( 0 , 0 );
       triangleEntities[ 0 ].template setSubentityIndex< 0 >( 1 , 1 );
       triangleEntities[ 0 ].template setSubentityIndex< 0 >( 2 , 2 );
       triangleEntities[ 0 ].template setSubentityIndex< 1 >( 0 , 0 );
       triangleEntities[ 0 ].template setSubentityIndex< 1 >( 1 , 1 );
       triangleEntities[ 0 ].template setSubentityIndex< 1 >( 2 , 2 );
       triangleEntities[ 1 ].template setSubentityIndex< 0 >( 0 , 0 );
       triangleEntities[ 1 ].template setSubentityIndex< 0 >( 1 , 2 );
       triangleEntities[ 1 ].template setSubentityIndex< 0 >( 2 , 3 );
       triangleEntities[ 1 ].template setSubentityIndex< 1 >( 0 , 0 );
       triangleEntities[ 1 ].template setSubentityIndex< 1 >( 1 , 3 );
       triangleEntities[ 1 ].template setSubentityIndex< 1 >( 2 , 4 );

       CPPUNIT_ASSERT( triangleEntities[ 0 ].template getSubentityIndex< 0 >( 0 ) == 0 );
       CPPUNIT_ASSERT( triangleEntities[ 0 ].template getSubentityIndex< 0 >( 1 ) == 1 );
       CPPUNIT_ASSERT( triangleEntities[ 0 ].template getSubentityIndex< 0 >( 2 ) == 2 );
       CPPUNIT_ASSERT( triangleEntities[ 0 ].template getSubentityIndex< 1 >( 0 ) == 0 );
       CPPUNIT_ASSERT( triangleEntities[ 0 ].template getSubentityIndex< 1 >( 1 ) == 1 );
       CPPUNIT_ASSERT( triangleEntities[ 0 ].template getSubentityIndex< 1 >( 2 ) == 2 );
       CPPUNIT_ASSERT( triangleEntities[ 1 ].template getSubentityIndex< 0 >( 0 ) == 0 );
       CPPUNIT_ASSERT( triangleEntities[ 1 ].template getSubentityIndex< 0 >( 1 ) == 2 );
       CPPUNIT_ASSERT( triangleEntities[ 1 ].template getSubentityIndex< 0 >( 2 ) == 3 );
       CPPUNIT_ASSERT( triangleEntities[ 1 ].template getSubentityIndex< 1 >( 0 ) == 0 );
       CPPUNIT_ASSERT( triangleEntities[ 1 ].template getSubentityIndex< 1 >( 1 ) == 3 );
       CPPUNIT_ASSERT( triangleEntities[ 1 ].template getSubentityIndex< 1 >( 2 ) == 4 );
 
       /*vertexEntities[ 0 ].template setNumberOfSuperentities< 1 >( 2 );
       vertexEntities[ 0 ].template setSuperentityIndex< 1 >( 0, 2 );
       vertexEntities[ 0 ].template setSuperentityIndex< 1 >( 1, 1 );

       vertexEntities[ 1 ].template setNumberOfSuperentities< 1 >( 3 );
       vertexEntities[ 1 ].template setSuperentityIndex< 1 >( 0, 0 );
       vertexEntities[ 1 ].template setSuperentityIndex< 1 >( 1, 2 );
       vertexEntities[ 1 ].template setSuperentityIndex< 1 >( 2, 4 );

       vertexEntities[ 1 ].template setNumberOfSuperentities< 2 >( 2 );
       vertexEntities[ 1 ].template setSuperentityIndex< 2 >( 0, 0 );
       vertexEntities[ 1 ].template setSuperentityIndex< 2 >( 1, 1 );*/

       CPPUNIT_ASSERT( vertexEntities[ 0 ].template getNumberOfSuperentities< 1 >() == 2 );
       CPPUNIT_ASSERT( vertexEntities[ 0 ].template getSuperentityIndex< 1 >( 0 ) == 2 );
       CPPUNIT_ASSERT( vertexEntities[ 0 ].template getSuperentityIndex< 1 >( 1 ) == 1 );

       CPPUNIT_ASSERT( vertexEntities[ 1 ].template getNumberOfSuperentities< 1 >() == 3 );
       CPPUNIT_ASSERT( vertexEntities[ 1 ].template getSuperentityIndex< 1 >( 0 ) == 0 );
       CPPUNIT_ASSERT( vertexEntities[ 1 ].template getSuperentityIndex< 1 >( 1 ) == 2 );
       CPPUNIT_ASSERT( vertexEntities[ 1 ].template getSuperentityIndex< 1 >( 2 ) == 4 );

       CPPUNIT_ASSERT( vertexEntities[ 1 ].template getNumberOfSuperentities< 2 >() == 2 );
       CPPUNIT_ASSERT( vertexEntities[ 1 ].template getSuperentityIndex< 2 >( 0 ) == 0 );
       CPPUNIT_ASSERT( vertexEntities[ 1 ].template getSuperentityIndex< 2 >( 1 ) == 1 );

       /*edgeEntities[ 0 ].template setNumberOfSuperentities< 2 >( 2 );
       edgeEntities[ 0 ].template setSuperentityIndex< 2 >( 0, 0 );
       edgeEntities[ 0 ].template setSuperentityIndex< 2 >( 1, 1 );*/

       /*CPPUNIT_ASSERT( edgeEntities[ 0 ].template getNumberOfSuperentities< 2 >() == 2  );
       CPPUNIT_ASSERT( edgeEntities[ 0 ].template getSuperentityIndex< 2 >( 0 ) == 0 );*/
    };

};

#endif



#endif /* TNLMESHENTITYTESTER_H_ */
