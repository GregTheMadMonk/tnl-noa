/***************************************************************************
                          tnlGrid2DTester.h  -  description
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

#ifndef TESTS_UNIT_TESTS_MESH_TNLGRID2DTESTER_H_
#define TESTS_UNIT_TESTS_MESH_TNLGRID2DTESTER_H_

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
      suiteOfTests -> addTest( new TestCallerType( "getCellNextToCellTest", &TesterType::getCellNextToCellTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getFaceNextToCellTest", &TesterType::getFaceNextToCellTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getCellNextToFaceTest", &TesterType::getCellNextToFaceTest ) );

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

   void getCellNextToCellTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      GridType grid;
      grid.setDimensions( xSize, ySize );
      for( IndexType j = 0; j < ySize; j++ )
         for( IndexType i = 0; i < xSize; i++ )
         {
            const CoordinatesType cellCoordinates( i, j );
            const IndexType cellIndex = grid.getCellIndex( cellCoordinates );
            if( i > 0 )
            {
               const CoordinatesType auxCellCoordinates( i - 1, j );
               const IndexType auxCellIndex = grid.getCellIndex( auxCellCoordinates );
               CPPUNIT_ASSERT( ( auxCellIndex == grid.template getCellNextToCell< -1, 0 >( cellIndex ) ) );
            }
            if( i < xSize - 1 )
            {
               const CoordinatesType auxCellCoordinates( i + 1, j );
               const IndexType auxCellIndex = grid.getCellIndex( auxCellCoordinates );
               CPPUNIT_ASSERT( ( auxCellIndex == grid.template getCellNextToCell< 1, 0 >( cellIndex ) ) );
            }
            if( j > 0 )
            {
               const CoordinatesType auxCellCoordinates( i, j - 1 );
               const IndexType auxCellIndex = grid.getCellIndex( auxCellCoordinates );
               CPPUNIT_ASSERT( ( auxCellIndex == grid.template getCellNextToCell< 0, -1 >( cellIndex ) ) );
            }
            if( j < ySize - 1 )
            {
               const CoordinatesType auxCellCoordinates( i, j + 1 );
               const IndexType auxCellIndex = grid.getCellIndex( auxCellCoordinates );
               CPPUNIT_ASSERT( ( auxCellIndex == grid.template getCellNextToCell< 0, 1 >( cellIndex ) ) );
            }
         }
   }

   void getFaceNextToCellTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      GridType grid;
      grid.setDimensions( xSize, ySize );
      for( IndexType j = 0; j < ySize; j++ )
         for( IndexType i = 0; i < xSize; i++ )
         {
            const CoordinatesType cellCoordinates( i, j );
            const IndexType cellIndex = grid.getCellIndex( cellCoordinates );

            CoordinatesType faceCoordinates( i, j );
            IndexType faceIndex = grid.template getFaceIndex< 1, 0 >( faceCoordinates );
            CPPUNIT_ASSERT( ( faceIndex == grid.template getFaceNextToCell< -1, 0 >( cellIndex ) ) );

            faceCoordinates = CoordinatesType( i + 1, j );
            faceIndex = grid.template getFaceIndex< 1, 0 >( faceCoordinates );
            CPPUNIT_ASSERT( ( faceIndex == grid.template getFaceNextToCell< 1, 0 >( cellIndex ) ) );

            faceCoordinates = CoordinatesType( i, j );
            faceIndex = grid.template getFaceIndex< 0, 1 >( faceCoordinates );
            CPPUNIT_ASSERT( ( faceIndex == grid.template getFaceNextToCell< 0, -1 >( cellIndex ) ) );

            faceCoordinates = CoordinatesType( i, j + 1 );
            faceIndex = grid.template getFaceIndex< 0, 1 >( faceCoordinates );
            CPPUNIT_ASSERT( ( faceIndex == grid.template getFaceNextToCell< 0, 1 >( cellIndex ) ) );
         }
   }

   void getCellNextToFaceTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      GridType grid;
      grid.setDimensions( xSize, ySize );
      for( IndexType j = 0; j <= ySize; j++ )
         for( IndexType i = 0; i <= xSize; i++ )
         {
            const CoordinatesType faceCoordinates( i, j );
            if( j < ySize )
            {
               const IndexType faceIndex = grid.template getFaceIndex< 1, 0 >( faceCoordinates );

               if( i > 0 )
               {
                  CoordinatesType cellCoordinates( i - 1, j );
                  IndexType cellIndex = grid.getCellIndex( cellCoordinates );
                  CPPUNIT_ASSERT( ( cellIndex == grid.template getCellNextToFace< -1, 0 >( faceIndex ) ) );
               }
               if( i < xSize )
               {
                  CoordinatesType cellCoordinates( i, j );
                  IndexType cellIndex = grid.getCellIndex( cellCoordinates );
                  CPPUNIT_ASSERT( ( cellIndex == grid.template getCellNextToFace< 1, 0 >( faceIndex ) ) );
               }
            }
            if( i < xSize )
            {
               const IndexType faceIndex = grid.template getFaceIndex< 0, 1 >( faceCoordinates );
               if( j > 0 )
               {
                  CoordinatesType cellCoordinates( i, j - 1 );
                  IndexType cellIndex = grid.getCellIndex( cellCoordinates );
                  CPPUNIT_ASSERT( ( cellIndex == grid.template getCellNextToFace< 0, -1 >( faceIndex ) ) );
               }
               if( j < ySize )
               {
                  CoordinatesType cellCoordinates( i, j );
                  IndexType cellIndex = grid.getCellIndex( cellCoordinates );
                  CPPUNIT_ASSERT( ( cellIndex == grid.template getCellNextToFace< 0, 1 >( faceIndex ) ) );
               }
            }
         }
   }
};

#endif /* TESTS_UNIT_TESTS_MESH_TNLGRID2DTESTER_H_ */
