/***************************************************************************
                          tnlMeshTester.h  -  description
                             -------------------
    begin                : Feb 18, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLMESHTESTER_H_
#define TNLMESHTESTER_H_

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <mesh/tnlMesh.h>
#include <mesh/tnlMeshEntity.h>
#include <mesh/config/tnlMeshConfigBase.h>
#include <mesh/topologies/tnlMeshVertexTag.h>
#include <mesh/topologies/tnlMeshEdgeTag.h>
#include <mesh/topologies/tnlMeshTriangleTag.h>
#include <mesh/topologies/tnlMeshTetrahedronTag.h>
#include <mesh/tnlMeshInitializer.h>

 typedef tnlMeshConfigBase< 2, double, int, int, void > MeshConfigBaseType;
 struct TestTriangleMeshConfig : public MeshConfigBaseType
 {
     typedef tnlMeshTriangleTag CellTag;
 };

 template< int Dimensions >
 struct tnlMeshSuperentityStorage< TestTriangleMeshConfig, tnlMeshVertexTag, Dimensions >
 {
    enum { enabled = true };
 };

 template< int Dimensions >
 struct tnlMeshSuperentityStorage< TestTriangleMeshConfig, tnlMeshEdgeTag, Dimensions >
 {
    enum { enabled = true };
 };

template< typename RealType, typename Device, typename IndexType >
class tnlMeshTester : public CppUnit :: TestCase
{
   public:
   typedef tnlMeshTester< RealType, Device, IndexType > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;

   tnlMeshTester(){};

   virtual
   ~tnlMeshTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlMeshTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "twoTrianglesTest", &TesterType::twoTrianglesTest ) );

      return suiteOfTests;
   }

   void twoTrianglesTest()
   {

       typedef tnlMeshEntity< TestTriangleMeshConfig, tnlMeshTriangleTag > TriangleMeshEntityType;
       typedef tnlMeshEntity< TestTriangleMeshConfig, tnlMeshEdgeTag > EdgeMeshEntityType;
       typedef tnlMeshEntity< TestTriangleMeshConfig, tnlMeshVertexTag > VertexMeshEntityType;
       typedef typename VertexMeshEntityType::PointType PointType;
       CPPUNIT_ASSERT( PointType::getType() == ( tnlStaticVector< 2, RealType >::getType() ) );

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

       tnlMesh< TestTriangleMeshConfig > mesh;
       mesh.setNumberOfVertices( 4 );
       mesh.setVertex( 0, PointType( 0.0, 0.0 ) );
       mesh.setVertex( 1, PointType( 1.0, 0.0 ) );
       mesh.setVertex( 2, PointType( 0.0, 1.0 ) );
       mesh.setVertex( 3, PointType( 1.0, 1.0 ) );

       mesh.setNumberOfEntities< 2 >( 2 );
       mesh.getEntity< 2 >( 0 ).setVertexIndex( 0, 0 );
       mesh.getEntity< 2 >( 0 ).setVertexIndex( 1, 1 );
       mesh.getEntity< 2 >( 0 ).setVertexIndex( 2, 2 );
       mesh.getEntity< 2 >( 1 ).setVertexIndex( 0, 1 );
       mesh.getEntity< 2 >( 1 ).setVertexIndex( 1, 2 );
       mesh.getEntity< 2 >( 1 ).setVertexIndex( 2, 3 );

       tnlMeshInitializer< TestTriangleMeshConfig > meshInitializer;
       cout << tnlMeshTraits< TestTriangleMeshConfig >::meshDimensions << endl;
       meshInitializer.initMesh( mesh );

/*       tnlStaticArray< 4, VertexMeshEntityType > vertexEntities;
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

       vertexEntities[ 0 ].template setNumberOfSuperentities< 1 >( 2 );
       vertexEntities[ 0 ].template setSuperentityIndex< 1 >( 0, 2 );
       vertexEntities[ 0 ].template setSuperentityIndex< 1 >( 1, 1 );

       vertexEntities[ 1 ].template setNumberOfSuperentities< 1 >( 3 );
       vertexEntities[ 1 ].template setSuperentityIndex< 1 >( 0, 0 );
       vertexEntities[ 1 ].template setSuperentityIndex< 1 >( 1, 2 );
       vertexEntities[ 1 ].template setSuperentityIndex< 1 >( 2, 4 );

       vertexEntities[ 1 ].template setNumberOfSuperentities< 2 >( 2 );
       vertexEntities[ 1 ].template setSuperentityIndex< 2 >( 0, 0 );
       vertexEntities[ 1 ].template setSuperentityIndex< 2 >( 1, 1 );

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

       edgeEntities[ 0 ].template setNumberOfSuperentities< 2 >( 2 );
       edgeEntities[ 0 ].template setSuperentityIndex< 2 >( 0, 0 );
       edgeEntities[ 0 ].template setSuperentityIndex< 2 >( 1, 1 );

       CPPUNIT_ASSERT( edgeEntities[ 0 ].template getNumberOfSuperentities< 2 >() == 2  );
       CPPUNIT_ASSERT( edgeEntities[ 0 ].template getSuperentityIndex< 2 >( 0 ) == 0 );
       */
    };

};

#endif





#endif /* TNLMESHTESTER_H_ */
