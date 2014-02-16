/***************************************************************************
                          tnlMeshEntityTester.h  -  description
                             -------------------
    begin                : Feb 11, 2014
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

#ifndef TNLMESHENTITYTESTER_H_
#define TNLMESHENTITYTESTER_H_

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <mesh/tnlMeshEntity.h>
#include <mesh/config/tnlMeshConfigBase.h>
#include <mesh/topologies/tnlMeshVertexTag.h>
#include <mesh/topologies/tnlMeshEdgeTag.h>
#include <mesh/topologies/tnlMeshTriangleTag.h>
#include <mesh/topologies/tnlMeshTetrahedronTag.h>
    
 typedef tnlMeshConfigBase< 2, double, int, int, void > MeshConfigBaseType;
 struct TestTriangleEntityTag : public MeshConfigBaseType
 {
     typedef tnlMeshTriangleTag CellTag;
 };
 struct TestEdgeEntityTag : public MeshConfigBaseType
 {
     typedef tnlMeshEdgeTag CellTag;
 };
 struct TestVertexEntityTag : public MeshConfigBaseType
 {
     typedef tnlMeshVertexTag CellTag;
 };

 template< int Dimensions >
 struct tnlMeshSuperentityStorage< TestTriangleEntityTag, tnlMeshVertexTag, Dimensions >
 {
    enum { enabled = true };
 };

 template< int Dimensions >
 struct tnlMeshSuperentityStorage< TestTriangleEntityTag, tnlMeshEdgeTag, Dimensions >
 {
    enum { enabled = true };
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
      typedef tnlMeshConfigBase< 2, RealType, IndexType, IndexType, void > MeshConfigBaseType;
      struct TestEntityTag : public MeshConfigBaseType
      {
         typedef tnlMeshEdgeTag CellTag;
      };
      typedef tnlMeshEntity< TestEntityTag, tnlMeshVertexTag > VertexMeshEntityType;
      typedef typename VertexMeshEntityType::PointType PointType;

      CPPUNIT_ASSERT( PointType::getType() == ( tnlStaticVector< 2, RealType >::getType() ) );
      VertexMeshEntityType vertexEntity;
      PointType point;

      point.x() = 1.0;
      point.y() = 2.0;
      vertexEntity.setPoint( point );
      CPPUNIT_ASSERT( vertexEntity.getPoint() == point );
   }

   void edgeMeshEntityTest()
   {
      typedef tnlMeshEntity< TestEdgeEntityTag, tnlMeshEdgeTag > EdgeMeshEntityType;
      typedef tnlMeshEntity< TestEdgeEntityTag, tnlMeshVertexTag > VertexMeshEntityType;
      
      typedef typename VertexMeshEntityType::PointType PointType;
      CPPUNIT_ASSERT( PointType::getType() == ( tnlStaticVector< 2, RealType >::getType() ) );

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

      CPPUNIT_ASSERT( vertexEntities[ edgeEntities[ 0 ].getVertexIndex( 0 ) ].getPoint() == point0 );
      CPPUNIT_ASSERT( vertexEntities[ edgeEntities[ 0 ].getVertexIndex( 1 ) ].getPoint() == point1 );
      CPPUNIT_ASSERT( vertexEntities[ edgeEntities[ 1 ].getVertexIndex( 0 ) ].getPoint() == point1 );
      CPPUNIT_ASSERT( vertexEntities[ edgeEntities[ 1 ].getVertexIndex( 1 ) ].getPoint() == point2 );
      CPPUNIT_ASSERT( vertexEntities[ edgeEntities[ 2 ].getVertexIndex( 0 ) ].getPoint() == point2 );
      CPPUNIT_ASSERT( vertexEntities[ edgeEntities[ 2 ].getVertexIndex( 1 ) ].getPoint() == point0 );
   }

   void triangleMeshEntityTest()
   {
      typedef tnlMeshEntity< TestTriangleEntityTag, tnlMeshTriangleTag > TriangleMeshEntityType;
      typedef tnlMeshEntity< TestEdgeEntityTag, tnlMeshEdgeTag > EdgeMeshEntityType;
      typedef tnlMeshEntity< TestVertexEntityTag, tnlMeshVertexTag > VertexMeshEntityType;
      typedef typename VertexMeshEntityType::PointType PointType;
      CPPUNIT_ASSERT( PointType::getType() == ( tnlStaticVector< 2, RealType >::getType() ) );

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
      edgeEntities[ 0 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTriangleTag, tnlMeshEdgeTag, 0, 0 >::index );
      edgeEntities[ 0 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTriangleTag, tnlMeshEdgeTag, 0, 1 >::index );
      edgeEntities[ 1 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTriangleTag, tnlMeshEdgeTag, 1, 0 >::index );
      edgeEntities[ 1 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTriangleTag, tnlMeshEdgeTag, 1, 1 >::index );
      edgeEntities[ 2 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTriangleTag, tnlMeshEdgeTag, 2, 0 >::index );
      edgeEntities[ 2 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTriangleTag, tnlMeshEdgeTag, 2, 1 >::index );

      CPPUNIT_ASSERT( edgeEntities[ 0 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTriangleTag, tnlMeshEdgeTag, 0, 0 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 0 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTriangleTag, tnlMeshEdgeTag, 0, 1 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 1 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTriangleTag, tnlMeshEdgeTag, 1, 0 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 1 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTriangleTag, tnlMeshEdgeTag, 1, 1 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 2 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTriangleTag, tnlMeshEdgeTag, 2, 0 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 2 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTriangleTag, tnlMeshEdgeTag, 2, 1 >::index ) );

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
      typedef tnlMeshConfigBase< 3, RealType, IndexType, IndexType, void > MeshConfigBaseType;
      struct TestTetrahedronEntityTag : public MeshConfigBaseType
      {
          typedef tnlMeshTetrahedronTag CellTag;
      };
      struct TestTriangleEntityTag : public MeshConfigBaseType
      {
          typedef tnlMeshTriangleTag CellTag;
      };
      struct TestEdgeEntityTag : public MeshConfigBaseType
      {
          typedef tnlMeshEdgeTag CellTag;
      };
      struct TestVertexEntityTag : public MeshConfigBaseType
      {
          typedef tnlMeshVertexTag CellTag;
      };
      typedef tnlMeshEntity< TestTetrahedronEntityTag, tnlMeshTetrahedronTag > TetrahedronMeshEntityType;
      typedef tnlMeshEntity< TestTriangleEntityTag, tnlMeshTriangleTag > TriangleMeshEntityType;
      typedef tnlMeshEntity< TestEdgeEntityTag, tnlMeshEdgeTag > EdgeMeshEntityType;
      typedef tnlMeshEntity< TestVertexEntityTag, tnlMeshVertexTag > VertexMeshEntityType;
      typedef typename VertexMeshEntityType::PointType PointType;
      CPPUNIT_ASSERT( PointType::getType() == ( tnlStaticVector< 3, RealType >::getType() ) );

      /****
       * We set-up similar situation as above but with
       * tetrahedron.
       */
      PointType point0( 0.0, 0.0, 0.0),
                point1( 1.0, 0.0, 0.0 ),
                point2( 0.0, 1.0, 0.0 ),
                point3( 0.0, 0.0, 1.0 );
      
      tnlStaticArray< tnlSubentities< tnlMeshTetrahedronTag, 0 >::count,
                      VertexMeshEntityType > vertexEntities;

      vertexEntities[ 0 ].setPoint( point0 );
      vertexEntities[ 1 ].setPoint( point1 );
      vertexEntities[ 2 ].setPoint( point2 );
      vertexEntities[ 3 ].setPoint( point3 );

      CPPUNIT_ASSERT( vertexEntities[ 0 ].getPoint() == point0 );
      CPPUNIT_ASSERT( vertexEntities[ 1 ].getPoint() == point1 );
      CPPUNIT_ASSERT( vertexEntities[ 2 ].getPoint() == point2 );
      CPPUNIT_ASSERT( vertexEntities[ 3 ].getPoint() == point3 );

      tnlStaticArray< tnlSubentities< tnlMeshTetrahedronTag, 1 >::count,
                      EdgeMeshEntityType > edgeEntities;
      edgeEntities[ 0 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 0, 0 >::index );
      edgeEntities[ 0 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 0, 1 >::index );
      edgeEntities[ 1 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 1, 0 >::index );
      edgeEntities[ 1 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 1, 1 >::index );
      edgeEntities[ 2 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 2, 0 >::index );
      edgeEntities[ 2 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 2, 1 >::index );
      edgeEntities[ 3 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 3, 0 >::index );
      edgeEntities[ 3 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 3, 1 >::index );
      edgeEntities[ 4 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 4, 0 >::index );
      edgeEntities[ 4 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 4, 1 >::index );
      edgeEntities[ 5 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 5, 0 >::index );
      edgeEntities[ 5 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 5, 1 >::index );

      CPPUNIT_ASSERT( edgeEntities[ 0 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 0, 0 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 0 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 0, 1 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 1 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 1, 0 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 1 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 1, 1 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 2 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 2, 0 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 2 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 2, 1 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 3 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 3, 0 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 3 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 3, 1 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 4 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 4, 0 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 4 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 4, 1 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 5 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 5, 0 >::index ) );
      CPPUNIT_ASSERT( edgeEntities[ 5 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 5, 1 >::index ) );

      tnlStaticArray< tnlSubentities< tnlMeshTetrahedronTag, 2 >::count,
                      TriangleMeshEntityType > triangleEntities;
      triangleEntities[ 0 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 0, 0 >::index );
      triangleEntities[ 0 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 0, 1 >::index );
      triangleEntities[ 0 ].setVertexIndex( 2, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 0, 2 >::index );
      triangleEntities[ 1 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 1, 0 >::index );
      triangleEntities[ 1 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 1, 1 >::index );
      triangleEntities[ 1 ].setVertexIndex( 2, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 1, 2 >::index );
      triangleEntities[ 2 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 2, 0 >::index );
      triangleEntities[ 2 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 2, 1 >::index );
      triangleEntities[ 2 ].setVertexIndex( 2, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 2, 2 >::index );
      triangleEntities[ 3 ].setVertexIndex( 0, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 3, 0 >::index );
      triangleEntities[ 3 ].setVertexIndex( 1, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 3, 1 >::index );
      triangleEntities[ 3 ].setVertexIndex( 2, tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 3, 2 >::index );

      CPPUNIT_ASSERT( triangleEntities[ 0 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 0, 0 >::index ) );
      CPPUNIT_ASSERT( triangleEntities[ 0 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 0, 1 >::index ) );
      CPPUNIT_ASSERT( triangleEntities[ 0 ].getVertexIndex( 2 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 0, 2 >::index ) );
      CPPUNIT_ASSERT( triangleEntities[ 1 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 1, 0 >::index ) );
      CPPUNIT_ASSERT( triangleEntities[ 1 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 1, 1 >::index ) );
      CPPUNIT_ASSERT( triangleEntities[ 1 ].getVertexIndex( 2 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 1, 2 >::index ) );
      CPPUNIT_ASSERT( triangleEntities[ 2 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 2, 0 >::index ) );
      CPPUNIT_ASSERT( triangleEntities[ 2 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 2, 1 >::index ) );
      CPPUNIT_ASSERT( triangleEntities[ 2 ].getVertexIndex( 2 ) == ( tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 2, 2 >::index ) );

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

       typedef tnlMeshEntity< TestTriangleEntityTag, tnlMeshTriangleTag > TriangleMeshEntityType;
       typedef tnlMeshEntity< TestTriangleEntityTag, tnlMeshEdgeTag > EdgeMeshEntityType;
       typedef tnlMeshEntity< TestTriangleEntityTag, tnlMeshVertexTag > VertexMeshEntityType;
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
    };

};

#endif



#endif /* TNLMESHENTITYTESTER_H_ */
