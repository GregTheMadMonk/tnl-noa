/***************************************************************************
                          tnlGrid1DTester.h  -  description
                             -------------------
    begin                : Feb 13, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

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

      CPPUNIT_ASSERT( grid.getSpaceSteps().x() == 0.1 );
   }

   void cellIndexingTest()
   {
      const IndexType xSize( 13 );
      GridType grid;
      grid.setDimensions( xSize );
 
      typename GridType::Cell cell( grid );
      for( cell.getCoordinates().x() = 0;
           cell.getCoordinates().x() < xSize;
           cell.getCoordinates().x()++ )
      {
         CPPUNIT_ASSERT( grid.getEntityIndex( cell ) >= 0 );
         CPPUNIT_ASSERT( grid.getEntityIndex( cell ) < grid.template getEntitiesCount< typename GridType::Cell >() );
         CPPUNIT_ASSERT( grid.template getEntity< typename GridType::Cell >( grid.getEntityIndex( cell ) ).getCoordinates() == cell.getCoordinates() );
      }
   }

   void vertexIndexingTest()
   {
      const IndexType xSize( 13 );
      GridType grid;
      grid.setDimensions( xSize );

      typename GridType::Vertex vertex( grid );
      for( vertex.getCoordinates().x() = 0;
           vertex.getCoordinates().x() < xSize;
           vertex.getCoordinates().x()++ )
      {
         CPPUNIT_ASSERT( grid.getEntityIndex( vertex ) >= 0 );
         CPPUNIT_ASSERT( grid.getEntityIndex( vertex ) < grid.template getEntitiesCount< typename GridType::Vertex >() );
         CPPUNIT_ASSERT( grid.template getEntity< typename GridType::Vertex >( grid.getEntityIndex( vertex ) ).getCoordinates() == vertex.getCoordinates() );
      }
   }

};



#endif /* TESTS_UNIT_TESTS_MESH_TNLGRID1DTESTER_H_ */
