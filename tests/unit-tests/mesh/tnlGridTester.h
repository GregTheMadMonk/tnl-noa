/***************************************************************************
                          tnlGridTester.h  -  description
                             -------------------
    begin                : Jul 28, 2014
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
#ifndef TNLGRIDTESTER_H_
#define TNLGRIDTESTER_H_


#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <mesh/tnlGrid.h>


template< int Dimensions, typename RealType, typename Device, typename IndexType >
class tnlGridTester{};

template< typename RealType, typename Device, typename IndexType >
class tnlGridTester< 1, RealType, Device, IndexType >: public CppUnit :: TestCase
{
   public:
   typedef tnlGridTester< 1, RealType, Device, IndexType > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;
   typedef tnlGrid< 1, RealType, Device, IndexType > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;
   typedef typename GridType::VertexType VertexType;

   tnlGridTester(){};

   virtual
   ~tnlGridTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlGridTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "setDomainTest", &TesterType::setDomainTest ) );
      suiteOfTests -> addTest( new TestCallerType( "cellIndexingTest", &TesterType::cellIndexingTest ) );
      suiteOfTests -> addTest( new TestCallerType( "vertexIndexingTest", &TesterType::vertexIndexingTest ) );
      return suiteOfTests;
   }

   void setDomainTest()
   {
      GridType grid;
      grid.setDomain( VertexType( 0.0 ), VertexType( 1.0 ) );
      grid.setDimensions( 10 );

      CPPUNIT_ASSERT( grid.getCellProportions().x() == 0.1 );
   }

   void cellIndexingTest()
   {
      const IndexType xSize( 13 );
      GridType grid;
      grid.setDimensions( xSize );
      for( IndexType i = 0; i < xSize; i++ )
      {
         CoordinatesType cellCoordinates( i );
         CPPUNIT_ASSERT( grid.getCellIndex( cellCoordinates ) >= 0 );
         CPPUNIT_ASSERT( grid.getCellIndex( cellCoordinates ) < grid.getNumberOfCells() );
         CPPUNIT_ASSERT( grid.getCellCoordinates( grid.getCellIndex( cellCoordinates ) ) == cellCoordinates );
      }
   }

   void vertexIndexingTest()
   {
      const IndexType xSize( 13 );
      GridType grid;
      grid.setDimensions( xSize );
      for( IndexType i = 0; i < xSize + 1; i++ )
      {
         CoordinatesType vertexCoordinates( i );
         CPPUNIT_ASSERT( grid.getVertexIndex( vertexCoordinates ) >= 0 );
         CPPUNIT_ASSERT( grid.getVertexIndex( vertexCoordinates ) < grid.getNumberOfVertices() );
         CPPUNIT_ASSERT( grid.getVertexCoordinates( grid.getVertexIndex( vertexCoordinates ) ) == vertexCoordinates );
      }
   }

};

template< typename RealType, typename Device, typename IndexType >
class tnlGridTester< 2, RealType, Device, IndexType >: public CppUnit :: TestCase
{
   public:
   typedef tnlGridTester< 2, RealType, Device, IndexType > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;
   typedef tnlGrid< 2, RealType, Device, IndexType > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;
   typedef typename GridType::VertexType VertexType;


   tnlGridTester(){};

   virtual
   ~tnlGridTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlGridTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "setDomainTest", &TesterType::setDomainTest ) );
      suiteOfTests -> addTest( new TestCallerType( "cellIndexingTest", &TesterType::cellIndexingTest ) );
      suiteOfTests -> addTest( new TestCallerType( "faceIndexingTest", &TesterType::faceIndexingTest ) );
      suiteOfTests -> addTest( new TestCallerType( "vertexIndexingTest", &TesterType::vertexIndexingTest ) );

      return suiteOfTests;
   }

   void setDomainTest()
   {
      GridType grid;
      grid.setDomain( VertexType( 0.0, 0.0 ), VertexType( 1.0, 1.0 ) );
      grid.setDimensions( 10, 20 );

      CPPUNIT_ASSERT( grid.getCellProportions().x() == 0.1 );
      CPPUNIT_ASSERT( grid.getCellProportions().y() == 0.05 );
   }

   void cellIndexingTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      GridType grid;
      grid.setDimensions( xSize, ySize );
      for( IndexType j = 0; j < ySize; j++ )
         for( IndexType i = 0; i < xSize; i++ )
         {
            CoordinatesType cellCoordinates( i, j );
            const IndexType cellIndex = grid.getCellIndex( cellCoordinates );
            CPPUNIT_ASSERT( cellIndex >= 0 );
            CPPUNIT_ASSERT( cellIndex < grid.getNumberOfCells() );
            CPPUNIT_ASSERT( grid.getCellCoordinates( cellIndex ) == cellCoordinates );
         }
   }

   void faceIndexingTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      GridType grid;
      grid.setDimensions( xSize, ySize );

      int nx, ny;
      for( IndexType j = 0; j < ySize; j++ )
         for( IndexType i = 0; i < xSize + 1; i++ )
         {
            CoordinatesType faceCoordinates( i, j );
            const IndexType faceIndex = grid.template getFaceIndex< 1, 0 >( faceCoordinates );
            CPPUNIT_ASSERT( faceIndex >= 0 );
            CPPUNIT_ASSERT( faceIndex < grid.getNumberOfFaces() );
            CPPUNIT_ASSERT( grid.getFaceCoordinates( faceIndex, nx, ny ) == faceCoordinates );
            CPPUNIT_ASSERT( nx == 1 );
            CPPUNIT_ASSERT( ny == 0 );
         }

      for( IndexType j = 0; j < ySize + 1; j++ )
         for( IndexType i = 0; i < xSize; i++ )
         {
            CoordinatesType faceCoordinates( i, j );
            const IndexType faceIndex = grid.template getFaceIndex< 0, 1 >( faceCoordinates );
            CPPUNIT_ASSERT( faceIndex >= 0 );
            CPPUNIT_ASSERT( faceIndex < grid.getNumberOfFaces() );
            CPPUNIT_ASSERT( grid.getFaceCoordinates( faceIndex, nx, ny ) == faceCoordinates );
            CPPUNIT_ASSERT( nx == 0 );
            CPPUNIT_ASSERT( ny == 1 );
         }

   }

   void vertexIndexingTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      GridType grid;
      grid.setDimensions( xSize, ySize );
      for( IndexType j = 0; j < ySize + 1; j++ )
         for( IndexType i = 0; i < xSize + 1; i++ )
         {
            CoordinatesType vertexCoordinates( i, j );
            const IndexType vertexIndex = grid.getVertexIndex( vertexCoordinates );
            CPPUNIT_ASSERT( vertexIndex >= 0 );
            CPPUNIT_ASSERT( vertexIndex < grid.getNumberOfVertices() );
            CPPUNIT_ASSERT( grid.getVertexCoordinates( vertexIndex ) == vertexCoordinates );
         }
   }

};

template< typename RealType, typename Device, typename IndexType >
class tnlGridTester< 3, RealType, Device, IndexType >: public CppUnit :: TestCase
{
   public:
   typedef tnlGridTester< 3, RealType, Device, IndexType > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;
   typedef tnlGrid< 3, RealType, Device, IndexType > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;
   typedef typename GridType::VertexType VertexType;


   tnlGridTester(){};

   virtual
   ~tnlGridTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlGridTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "setDomainTest", &TesterType::setDomainTest ) );
      suiteOfTests -> addTest( new TestCallerType( "cellIndexingTest", &TesterType::cellIndexingTest ) );
      suiteOfTests -> addTest( new TestCallerType( "faceIndexingTest", &TesterType::faceIndexingTest ) );
      suiteOfTests -> addTest( new TestCallerType( "edgeIndexingTest", &TesterType::edgeIndexingTest ) );
      suiteOfTests -> addTest( new TestCallerType( "vertexIndexingTest", &TesterType::vertexIndexingTest ) );

      return suiteOfTests;
   }

   void setDomainTest()
   {
      GridType grid;
      grid.setDomain( VertexType( 0.0, 0.0, 0.0 ), VertexType( 1.0, 1.0, 1.0 ) );
      grid.setDimensions( 10, 20, 40 );

      CPPUNIT_ASSERT( grid.getCellProportions().x() == 0.1 );
      CPPUNIT_ASSERT( grid.getCellProportions().y() == 0.05 );
      CPPUNIT_ASSERT( grid.getCellProportions().z() == 0.025 );
   }

   void cellIndexingTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      const IndexType zSize( 19 );
      GridType grid;
      grid.setDimensions( xSize, ySize, zSize );
      for( IndexType k = 0; k < zSize; k++ )
         for( IndexType j = 0; j < ySize; j++ )
            for( IndexType i = 0; i < xSize; i++ )
            {
               CoordinatesType cellCoordinates( i, j, k );
               const IndexType cellIndex = grid.getCellIndex( cellCoordinates );
               CPPUNIT_ASSERT( cellIndex >= 0 );
               CPPUNIT_ASSERT( cellIndex < grid.getNumberOfCells() );
               CPPUNIT_ASSERT( grid.getCellCoordinates( cellIndex ) == cellCoordinates );
            }
   }

   void faceIndexingTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      const IndexType zSize( 19 );
      GridType grid;
      grid.setDimensions( xSize, ySize, zSize );

      int nx, ny, nz;
      for( IndexType k = 0; k < zSize; k++ )
         for( IndexType j = 0; j < ySize; j++ )
            for( IndexType i = 0; i < xSize + 1; i++ )
            {
               CoordinatesType faceCoordinates( i, j, k );
               const IndexType faceIndex = grid.template getFaceIndex< 1, 0, 0 >( faceCoordinates );
               CPPUNIT_ASSERT( faceIndex >= 0 );
               CPPUNIT_ASSERT( faceIndex < grid.getNumberOfFaces() );
               CPPUNIT_ASSERT( grid.getFaceCoordinates( faceIndex, nx, ny, nz ) == faceCoordinates );
               CPPUNIT_ASSERT( nx == 1 );
               CPPUNIT_ASSERT( ny == 0 );
               CPPUNIT_ASSERT( nz == 0 );
            }

      for( IndexType k = 0; k < zSize; k++ )
         for( IndexType j = 0; j < ySize + 1; j++ )
            for( IndexType i = 0; i < xSize; i++ )
            {
               CoordinatesType faceCoordinates( i, j, k );
               const IndexType faceIndex = grid.template getFaceIndex< 0, 1, 0 >( faceCoordinates );
               CPPUNIT_ASSERT( faceIndex >= 0 );
               CPPUNIT_ASSERT( faceIndex < grid.getNumberOfFaces() );
               CPPUNIT_ASSERT( grid.getFaceCoordinates( faceIndex, nx, ny, nz ) == faceCoordinates );
               CPPUNIT_ASSERT( nx == 0 );
               CPPUNIT_ASSERT( ny == 1 );
               CPPUNIT_ASSERT( nz == 0 );
            }

      for( IndexType k = 0; k < zSize + 1; k++ )
         for( IndexType j = 0; j < ySize; j++ )
            for( IndexType i = 0; i < xSize; i++ )
            {
               CoordinatesType faceCoordinates( i, j, k );
               const IndexType faceIndex = grid.template getFaceIndex< 0, 0, 1 >( faceCoordinates );
               CPPUNIT_ASSERT( faceIndex >= 0 );
               CPPUNIT_ASSERT( faceIndex < grid.getNumberOfFaces() );
               CPPUNIT_ASSERT( grid.getFaceCoordinates( faceIndex, nx, ny, nz ) == faceCoordinates );
               CPPUNIT_ASSERT( nx == 0 );
               CPPUNIT_ASSERT( ny == 0 );
               CPPUNIT_ASSERT( nz == 1 );
            }

   }

   void edgeIndexingTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      const IndexType zSize( 19 );
      GridType grid;
      grid.setDimensions( xSize, ySize, zSize );

      int dx, dy, dz;
      for( IndexType k = 0; k < zSize + 1; k++ )
         for( IndexType j = 0; j < ySize + 1; j++ )
            for( IndexType i = 0; i < xSize; i++ )
            {
               CoordinatesType edgeCoordinates( i, j, k );
               const IndexType edgeIndex = grid.template getEdgeIndex< 1, 0, 0 >( edgeCoordinates );
               CPPUNIT_ASSERT( edgeIndex >= 0 );
               CPPUNIT_ASSERT( edgeIndex < grid.getNumberOfEdges() );
               CPPUNIT_ASSERT( grid.getEdgeCoordinates( edgeIndex, dx, dy, dz ) == edgeCoordinates );
               CPPUNIT_ASSERT( dx == 1 );
               CPPUNIT_ASSERT( dy == 0 );
               CPPUNIT_ASSERT( dz == 0 );
            }

      for( IndexType k = 0; k < zSize + 1; k++ )
         for( IndexType j = 0; j < ySize; j++ )
            for( IndexType i = 0; i < xSize + 1; i++ )
            {
               CoordinatesType edgeCoordinates( i, j, k );
               const IndexType edgeIndex = grid.template getEdgeIndex< 0, 1, 0 >( edgeCoordinates );
               CPPUNIT_ASSERT( edgeIndex >= 0 );
               CPPUNIT_ASSERT( edgeIndex < grid.getNumberOfEdges() );
               CPPUNIT_ASSERT( grid.getEdgeCoordinates( edgeIndex, dx, dy, dz ) == edgeCoordinates );
               CPPUNIT_ASSERT( dx == 0 );
               CPPUNIT_ASSERT( dy == 1 );
               CPPUNIT_ASSERT( dz == 0 );
            }

      for( IndexType k = 0; k < zSize; k++ )
         for( IndexType j = 0; j < ySize + 1; j++ )
            for( IndexType i = 0; i < xSize + 1; i++ )
            {
               CoordinatesType edgeCoordinates( i, j, k );
               const IndexType edgeIndex = grid.template getEdgeIndex< 0, 0, 1 >( edgeCoordinates );
               CPPUNIT_ASSERT( edgeIndex >= 0 );
               CPPUNIT_ASSERT( edgeIndex < grid.getNumberOfEdges() );
               CPPUNIT_ASSERT( grid.getEdgeCoordinates( edgeIndex, dx, dy, dz ) == edgeCoordinates );
               CPPUNIT_ASSERT( dx == 0 );
               CPPUNIT_ASSERT( dy == 0 );
               CPPUNIT_ASSERT( dz == 1 );
            }

   }

   void vertexIndexingTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      const IndexType zSize( 19 );
      GridType grid;
      grid.setDimensions( xSize, ySize, zSize );
      for( IndexType k = 0; k < zSize + 1; k++ )
         for( IndexType j = 0; j < ySize + 1; j++ )
            for( IndexType i = 0; i < xSize + 1; i++ )
            {
               CoordinatesType vertexCoordinates( i, j, k );
               const IndexType vertexIndex = grid.getVertexIndex( vertexCoordinates );
               CPPUNIT_ASSERT( vertexIndex >= 0 );
               CPPUNIT_ASSERT( vertexIndex < grid.getNumberOfVertices() );
               CPPUNIT_ASSERT( grid.getVertexCoordinates( vertexIndex ) == vertexCoordinates );
            }
   }



};


#endif /* HAVE_CPPUNIT */

#endif /* TNLGRIDTESTER_H_ */
