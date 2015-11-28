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
      
      typename GridType::template GridEntity< GridType::Cells > cell( grid );
      for( cell.getCoordinates().x() = 0;
           cell.getCoordinates().x() < xSize;
           cell.getCoordinates().x()++ )
      {
         CPPUNIT_ASSERT( grid.getEntityIndex( cell ) >= 0 );
         CPPUNIT_ASSERT( grid.getEntityIndex( cell ) < grid.template getEntitiesCount< GridType::Cells >() );
         CPPUNIT_ASSERT( grid.template getEntity< GridType::Cells >( grid.getEntityIndex( cell ) ).getCoordinates() == cell.getCoordinates() );
      }
   }

   void vertexIndexingTest()
   {
      const IndexType xSize( 13 );
      GridType grid;
      grid.setDimensions( xSize );

      typename GridType::template GridEntity< GridType::Vertices > vertex( grid );
      for( vertex.getCoordinates().x() = 0;
           vertex.getCoordinates().x() < xSize;
           vertex.getCoordinates().x()++ )
      {
         CPPUNIT_ASSERT( grid.getEntityIndex( vertex ) >= 0 );
         CPPUNIT_ASSERT( grid.getEntityIndex( vertex ) < grid.template getEntitiesCount< GridType::Vertices >() );
         CPPUNIT_ASSERT( grid.template getEntity< GridType::Vertices >( grid.getEntityIndex( vertex ) ).getCoordinates() == vertex.getCoordinates() );
      }
   }

};



#endif /* TESTS_UNIT_TESTS_MESH_TNLGRID1DTESTER_H_ */
