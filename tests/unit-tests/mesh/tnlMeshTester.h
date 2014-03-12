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
#include <mesh/topologies/tnlMeshQuadrilateralTag.h>
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
 struct TestQuadrilateralMeshConfig : public Mesh3dConfigBaseType
 {
     typedef tnlMeshQuadrilateralTag CellTag;
 };

 template< int Dimensions >
 struct tnlMeshSuperentityStorage< TestQuadrilateralMeshConfig, tnlMeshVertexTag, Dimensions >
 {
    enum { enabled = true };
 };

 template< int Dimensions >
 struct tnlMeshSuperentityStorage< TestQuadrilateralMeshConfig, tnlMeshEdgeTag, Dimensions >
 {
    enum { enabled = true };
 };

 template< int Dimensions >
 struct tnlMeshSuperentityStorage< TestQuadrilateralMeshConfig, tnlMeshTriangleTag, Dimensions >
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
      //suiteOfTests -> addTest( new TestCallerType( "quadrilateralsTest", &TesterType::quadrilateralsTest ) );

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
       mesh.print( cout );
       CPPUNIT_ASSERT( mesh.getNumberOfEntities< 2 >() == 2 );
       CPPUNIT_ASSERT( mesh.getNumberOfEntities< 1 >() == 5 );
       CPPUNIT_ASSERT( mesh.getNumberOfEntities< 0 >() == 4 );

       mesh.save( "mesh.tnl" );
       tnlMesh< TestTriangleMeshConfig > mesh2;
       mesh2.load( "mesh.tnl" );
       cout << "===================== Mesh2 =========================" << endl;
       mesh2.print( cout );
       cout << "=====================================================" << endl;
       //CPPUNIT_ASSERT( mesh == mesh2 );

    };

   void quadrilateralsTest()
   {
      typedef tnlMeshEntity< TestQuadrilateralMeshConfig, tnlMeshTriangleTag > TriangleMeshEntityType;
      typedef tnlMeshEntity< TestQuadrilateralMeshConfig, tnlMeshEdgeTag > EdgeMeshEntityType;
      typedef tnlMeshEntity< TestQuadrilateralMeshConfig, tnlMeshVertexTag > VertexMeshEntityType;
      typedef typename VertexMeshEntityType::PointType PointType;
      tnlMesh< TestQuadrilateralMeshConfig > mesh;
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

      //mesh.setNumberOfEntities< 3 >( 18 );
      //mesh.getEntities< 3 >[ 0 ].getVertices()[ 0 ] = 13;
      /*9        8        6
                 13        8        9       11
                 13       12        9       10
                 11       12        3        9
                 13        8        7        6
                 10       13        6        9
                 13       12       10        4
                 10        5       12        9
                 13       10        6        4
                  2        3        1       12
                  9       12        3        5
                  2        3       12        5
                 10        5        2       12
                 11       12        9       13
                 13        7        8       11
                11       12       13        4
                 13        7        4        6
                13        4        7       11*/


   }

};

#endif





#endif /* TNLMESHTESTER_H_ */
