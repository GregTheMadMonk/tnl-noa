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

 typedef tnlMeshConfigBase< 2, double, int, int, void > Mesh2dConfigBaseType;
 struct TestTriangleMeshConfig : public Mesh2dConfigBaseType
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

 typedef tnlMeshConfigBase< 3, double, int, int, void > Mesh3dConfigBaseType;
 struct TestTetrahedronMeshConfig : public Mesh3dConfigBaseType
 {
     typedef tnlMeshTetrahedronTag CellTag;
 };

 template< int Dimensions >
 struct tnlMeshSuperentityStorage< TestTetrahedronMeshConfig, tnlMeshVertexTag, Dimensions >
 {
    enum { enabled = true };
 };

 template< int Dimensions >
 struct tnlMeshSuperentityStorage< TestTetrahedronMeshConfig, tnlMeshEdgeTag, Dimensions >
 {
    enum { enabled = true };
 };

 template< int Dimensions >
 struct tnlMeshSuperentityStorage< TestTetrahedronMeshConfig, tnlMeshTriangleTag, Dimensions >
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
      suiteOfTests -> addTest( new TestCallerType( "tetrahedronsTest", &TesterType::tetrahedronsTest ) );

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

       tnlMesh< TestTriangleMeshConfig > mesh, mesh2;
       mesh.setName( "mesh" );
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
       meshInitializer.initMesh( mesh );

       CPPUNIT_ASSERT( mesh.getNumberOfEntities< 2 >() == 2 );
       CPPUNIT_ASSERT( mesh.getNumberOfEntities< 1 >() == 5 );
       CPPUNIT_ASSERT( mesh.getNumberOfEntities< 0 >() == 4 );

       CPPUNIT_ASSERT( mesh.save( "mesh.tnl" ) );
       CPPUNIT_ASSERT( mesh2.load( "mesh.tnl" ) );
       CPPUNIT_ASSERT( mesh == mesh2 );

       //mesh2.setName( "mesh2" );
       //mesh.print( cout );
       //mesh2.print( cout );


    };

   void tetrahedronsTest()
   {
      typedef tnlMeshEntity< TestTetrahedronMeshConfig, tnlMeshTriangleTag > TriangleMeshEntityType;
      typedef tnlMeshEntity< TestTetrahedronMeshConfig, tnlMeshEdgeTag > EdgeMeshEntityType;
      typedef tnlMeshEntity< TestTetrahedronMeshConfig, tnlMeshVertexTag > VertexMeshEntityType;
      typedef typename VertexMeshEntityType::PointType PointType;
      tnlMesh< TestTetrahedronMeshConfig > mesh;
      mesh.setNumberOfVertices( 13 );
      mesh.setVertex(  0, PointType(  0.000000, 0.000000, 0.000000 ) );
      mesh.setVertex(  1, PointType(  0.000000, 0.000000, 8.000000 ) );
      mesh.setVertex(  2, PointType(  0.000000, 8.000000, 0.000000 ) );
      mesh.setVertex(  3, PointType( 15.000000, 0.000000, 0.000000 ) );
      mesh.setVertex(  4, PointType(  0.000000, 8.000000, 8.000000 ) );
      mesh.setVertex(  5, PointType( 15.000000, 0.000000, 8.000000 ) );
      mesh.setVertex(  6, PointType( 15.000000, 8.000000, 0.000000 ) );
      mesh.setVertex(  7, PointType( 15.000000, 8.000000, 8.000000 ) );
      mesh.setVertex(  8, PointType(  7.470740, 8.000000, 8.000000 ) );
      mesh.setVertex(  9, PointType(  7.470740, 0.000000, 8.000000 ) );
      mesh.setVertex( 10, PointType(  7.504125, 8.000000, 0.000000 ) );
      mesh.setVertex( 11, PointType(  7.212720, 0.000000, 0.000000 ) );
      mesh.setVertex( 12, PointType( 11.184629, 3.987667, 3.985835 ) );

      /****
       * Setup the following tetrahedrons:
       * ( Generated by Netgen )
       *
       *  12        8        7        5
       *  12        7        8       10
       *  12       11        8        9
       *  10       11        2        8
       *  12        7        6        5
       *   9       12        5        8
       *  12       11        9        3
       *   9        4       11        8
       *  12        9        5        3
       *   1        2        0       11
       *   8       11        2        4
       *   1        2       11        4
       *   9        4        1       11
       *  10       11        8       12
       *  12        6        7       10
       *  10       11       12        3
       *  12        6        3        5
       *  12        3        6       10
       */
      
      mesh.setNumberOfEntities< 3 >( 1 );

       //  12        8        7        5
      mesh.getEntities< 3 >()[ 0 ].getVerticesIndices()[ 0 ] = 12;
      mesh.getEntities< 3 >()[ 0 ].getVerticesIndices()[ 1 ] = 8;
      mesh.getEntities< 3 >()[ 0 ].getVerticesIndices()[ 2 ] = 7;
      mesh.getEntities< 3 >()[ 0 ].getVerticesIndices()[ 3 ] = 5;

       //  12        7        8       10
      /*mesh.getEntities< 3 >()[ 1 ].getVerticesIndices()[ 0 ] = 12;
      mesh.getEntities< 3 >()[ 1 ].getVerticesIndices()[ 1 ] = 7;
      mesh.getEntities< 3 >()[ 1 ].getVerticesIndices()[ 2 ] = 8;
      mesh.getEntities< 3 >()[ 1 ].getVerticesIndices()[ 3 ] = 10;
                 
       //  12       11        8        9
      /*mesh.getEntities< 3 >()[ 2 ].getVerticesIndices()[ 0 ] = 12;
      mesh.getEntities< 3 >()[ 2 ].getVerticesIndices()[ 1 ] = 11;
      mesh.getEntities< 3 >()[ 2 ].getVerticesIndices()[ 2 ] = 8;
      mesh.getEntities< 3 >()[ 2 ].getVerticesIndices()[ 3 ] = 9;
                 
       //  10       11        2        8
      mesh.getEntities< 3 >()[ 3 ].getVerticesIndices()[ 0 ] = 10;
      mesh.getEntities< 3 >()[ 3 ].getVerticesIndices()[ 1 ] = 11;
      mesh.getEntities< 3 >()[ 3 ].getVerticesIndices()[ 2 ] = 2;
      mesh.getEntities< 3 >()[ 3 ].getVerticesIndices()[ 3 ] = 8;
                 
       //  12        7        6        5
      mesh.getEntities< 3 >()[ 4 ].getVerticesIndices()[ 0 ] = 12;
      mesh.getEntities< 3 >()[ 4 ].getVerticesIndices()[ 1 ] = 7;
      mesh.getEntities< 3 >()[ 4 ].getVerticesIndices()[ 2 ] = 6;
      mesh.getEntities< 3 >()[ 4 ].getVerticesIndices()[ 3 ] = 5;
                 
       //   9       12        5        8
      mesh.getEntities< 3 >()[ 5 ].getVerticesIndices()[ 0 ] = 9;
      mesh.getEntities< 3 >()[ 5 ].getVerticesIndices()[ 1 ] = 12;
      mesh.getEntities< 3 >()[ 5 ].getVerticesIndices()[ 2 ] = 5;
      mesh.getEntities< 3 >()[ 5 ].getVerticesIndices()[ 3 ] = 8;
                 
       //  12       11        9        3
      mesh.getEntities< 3 >()[ 6 ].getVerticesIndices()[ 0 ] = 12;
      mesh.getEntities< 3 >()[ 6 ].getVerticesIndices()[ 1 ] = 11;
      mesh.getEntities< 3 >()[ 6 ].getVerticesIndices()[ 2 ] = 9;
      mesh.getEntities< 3 >()[ 6 ].getVerticesIndices()[ 3 ] = 3;
                 
       //   9        4       11        8
      mesh.getEntities< 3 >()[ 7 ].getVerticesIndices()[ 0 ] = 9;
      mesh.getEntities< 3 >()[ 7 ].getVerticesIndices()[ 1 ] = 4;
      mesh.getEntities< 3 >()[ 7 ].getVerticesIndices()[ 2 ] = 11;
      mesh.getEntities< 3 >()[ 7 ].getVerticesIndices()[ 3 ] = 8;
                
       //  12        9        5        3
      mesh.getEntities< 3 >()[ 8 ].getVerticesIndices()[ 0 ] = 12;
      mesh.getEntities< 3 >()[ 8 ].getVerticesIndices()[ 1 ] = 9;
      mesh.getEntities< 3 >()[ 8 ].getVerticesIndices()[ 2 ] = 5;
      mesh.getEntities< 3 >()[ 8 ].getVerticesIndices()[ 3 ] = 3;
                 
       //   1        2        0       11
      mesh.getEntities< 3 >()[ 9 ].getVerticesIndices()[ 0 ] = 1;
      mesh.getEntities< 3 >()[ 9 ].getVerticesIndices()[ 1 ] = 2;
      mesh.getEntities< 3 >()[ 9 ].getVerticesIndices()[ 2 ] = 0;
      mesh.getEntities< 3 >()[ 9 ].getVerticesIndices()[ 3 ] = 11;
                 
       //   8       11        2        4
      mesh.getEntities< 3 >()[ 10 ].getVerticesIndices()[ 0 ] = 8;
      mesh.getEntities< 3 >()[ 10 ].getVerticesIndices()[ 1 ] = 11;
      mesh.getEntities< 3 >()[ 10 ].getVerticesIndices()[ 2 ] = 2;
      mesh.getEntities< 3 >()[ 10 ].getVerticesIndices()[ 3 ] = 4;
                 
       //   1        2       11        4
      mesh.getEntities< 3 >()[ 11 ].getVerticesIndices()[ 0 ] = 1;
      mesh.getEntities< 3 >()[ 11 ].getVerticesIndices()[ 1 ] = 2;
      mesh.getEntities< 3 >()[ 11 ].getVerticesIndices()[ 2 ] = 11;
      mesh.getEntities< 3 >()[ 11 ].getVerticesIndices()[ 3 ] = 4;
                 
       //   9        4        1       11
      mesh.getEntities< 3 >()[ 12 ].getVerticesIndices()[ 0 ] = 9;
      mesh.getEntities< 3 >()[ 12 ].getVerticesIndices()[ 1 ] = 4;
      mesh.getEntities< 3 >()[ 12 ].getVerticesIndices()[ 2 ] = 1;
      mesh.getEntities< 3 >()[ 12 ].getVerticesIndices()[ 3 ] = 11;
                 
       //  10       11        8       12
      mesh.getEntities< 3 >()[ 13 ].getVerticesIndices()[ 0 ] = 10;
      mesh.getEntities< 3 >()[ 13 ].getVerticesIndices()[ 1 ] = 11;
      mesh.getEntities< 3 >()[ 13 ].getVerticesIndices()[ 2 ] = 8;
      mesh.getEntities< 3 >()[ 13 ].getVerticesIndices()[ 3 ] = 12;
                 
       //  12        6        7       10
      mesh.getEntities< 3 >()[ 14 ].getVerticesIndices()[ 0 ] = 12;
      mesh.getEntities< 3 >()[ 14 ].getVerticesIndices()[ 1 ] = 6;
      mesh.getEntities< 3 >()[ 14 ].getVerticesIndices()[ 2 ] = 7;
      mesh.getEntities< 3 >()[ 14 ].getVerticesIndices()[ 3 ] = 10;
                 
       //  10       11       12        3
      mesh.getEntities< 3 >()[ 15 ].getVerticesIndices()[ 0 ] = 10;
      mesh.getEntities< 3 >()[ 15 ].getVerticesIndices()[ 1 ] = 11;
      mesh.getEntities< 3 >()[ 15 ].getVerticesIndices()[ 2 ] = 12;
      mesh.getEntities< 3 >()[ 15 ].getVerticesIndices()[ 3 ] = 3;

       //  12        6        3        5
      mesh.getEntities< 3 >()[ 16 ].getVerticesIndices()[ 0 ] = 12;
      mesh.getEntities< 3 >()[ 16 ].getVerticesIndices()[ 1 ] = 6;
      mesh.getEntities< 3 >()[ 16 ].getVerticesIndices()[ 2 ] = 3;
      mesh.getEntities< 3 >()[ 16 ].getVerticesIndices()[ 3 ] = 5;
                 
       //  12        3        6       10
      mesh.getEntities< 3 >()[ 17 ].getVerticesIndices()[ 0 ] = 12;
      mesh.getEntities< 3 >()[ 17 ].getVerticesIndices()[ 1 ] = 3;
      mesh.getEntities< 3 >()[ 17 ].getVerticesIndices()[ 2 ] = 6;
      mesh.getEntities< 3 >()[ 17 ].getVerticesIndices()[ 3 ] = 10;
      */
                 
      tnlMeshInitializer< TestTetrahedronMeshConfig > meshInitializer;
      meshInitializer.initMesh( mesh );

      mesh.print( cout );

   }

};

#endif





#endif /* TNLMESHTESTER_H_ */
