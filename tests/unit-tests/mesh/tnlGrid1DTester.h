/***************************************************************************
                          tnlGrid1DTester.h  -  description
                             -------------------
    begin                : Feb 13, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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

#ifndef TESTS_UNIT_TESTS_MESH_TNLGRID1DTESTER_H_
#define TESTS_UNIT_TESTS_MESH_TNLGRID1DTESTER_H_

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
         CPPUNIT_ASSERT( grid.getCellIndex( cellCoordinates ) < grid.template getEntitiesCount< GridType::Cells >() );
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



#endif /* TESTS_UNIT_TESTS_MESH_TNLGRID1DTESTER_H_ */
