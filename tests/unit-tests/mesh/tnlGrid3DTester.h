/***************************************************************************
                          Grid3DTester.h  -  description
                             -------------------
    begin                : Feb 13, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TESTS_UNIT_TESTS_MESH_TNLGRID3DTESTER_H_
#define TESTS_UNIT_TESTS_MESH_TNLGRID3DTESTER_H_

using namespace TNL;

template< typename RealType, typename Device, typename IndexType >
class GridTester< 3, RealType, Device, IndexType >: public CppUnit :: TestCase
{
   public:
   typedef GridTester< 3, RealType, Device, IndexType > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;
   typedef Meshes::Grid< 3, RealType, Device, IndexType > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;
   typedef typename GridType::VertexType VertexType;


   GridTester(){};

   virtual
   ~GridTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "GridTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "setDomainTest", &TesterType::setDomainTest ) );
      suiteOfTests -> addTest( new TestCallerType( "cellIndexingTest", &TesterType::cellIndexingTest ) );
      suiteOfTests -> addTest( new TestCallerType( "faceIndexingTest", &TesterType::faceIndexingTest ) );
      suiteOfTests -> addTest( new TestCallerType( "edgeIndexingTest", &TesterType::edgeIndexingTest ) );
      suiteOfTests -> addTest( new TestCallerType( "vertexIndexingTest", &TesterType::vertexIndexingTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getCellNextToCellTest", &TesterType::getCellNextToCellTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getFaceNextToCellTest", &TesterType::getFaceNextToCellTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getCellNextToFaceTest", &TesterType::getCellNextToFaceTest ) );


      return suiteOfTests;
   }

   void setDomainTest()
   {
      GridType grid;
      grid.setDomain( VertexType( 0.0, 0.0, 0.0 ), VertexType( 1.0, 1.0, 1.0 ) );
      grid.setDimensions( 10, 20, 40 );

      CPPUNIT_ASSERT( grid.getSpaceSteps().x() == 0.1 );
      CPPUNIT_ASSERT( grid.getSpaceSteps().y() == 0.05 );
      CPPUNIT_ASSERT( grid.getSpaceSteps().z() == 0.025 );
   }

   void cellIndexingTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      const IndexType zSize( 19 );
      GridType grid;
      grid.setDimensions( xSize, ySize, zSize );
      typename GridType::Cell cell( grid );
      for( cell.getCoordinates().z() = 0;
           cell.getCoordinates().z() < zSize;
           cell.getCoordinates().z()++ )
         for( cell.getCoordinates().y() = 0;
              cell.getCoordinates().y() < ySize;
              cell.getCoordinates().y()++ )
            for( cell.getCoordinates().x() = 0;
                 cell.getCoordinates().x() < xSize;
                 cell.getCoordinates().x()++ )
            {
               const IndexType cellIndex = grid.template getEntityIndex< typename GridType::Cell >( cell );
               CPPUNIT_ASSERT( cellIndex >= 0 );
               CPPUNIT_ASSERT( cellIndex < grid.template getEntitiesCount< typename GridType::Cell >() );
               CPPUNIT_ASSERT( grid.template getEntity< typename GridType::Cell >( cellIndex ).getCoordinates() == cell.getCoordinates() );
            }
   }

   void faceIndexingTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      const IndexType zSize( 19 );
      GridType grid;
      grid.setDimensions( xSize, ySize, zSize );
 
      typedef typename GridType::template MeshEntity< 2 > FaceType;
      typedef typename FaceType::EntityOrientationType OrientationType;
      typedef typename FaceType::EntityBasisType BasisType;
      FaceType face( grid );
 
      face.setOrientation( OrientationType( 1, 0, 0 ) );
      for( face.getCoordinates().z() = 0;
            face.getCoordinates().z() < zSize;
            face.getCoordinates().z()++ )
         for( face.getCoordinates().y() = 0;
              face.getCoordinates().y() < ySize;
              face.getCoordinates().y()++ )
            for( face.getCoordinates().x() = 0;
                 face.getCoordinates().x() < xSize + 1;
                 face.getCoordinates().x()++ )
            {
               const IndexType faceIndex = grid.template getEntityIndex( face );
               CPPUNIT_ASSERT( faceIndex >= 0 );
               CPPUNIT_ASSERT( faceIndex < grid.template getEntitiesCount< typename GridType::Face >() );
               CPPUNIT_ASSERT( grid.template getEntity< typename GridType::Face >( faceIndex ).getCoordinates() == face.getCoordinates() );
               CPPUNIT_ASSERT( grid.template getEntity< typename GridType::Face >( faceIndex ).getOrientation() == OrientationType( 1, 0, 0 ) );
               // TODO: fix this - gives undefined reference - I do not know why
               //CPPUNIT_ASSERT( grid.template getEntity< 1 >( faceIndex ).getBasis() == BasisType( 0, 1 ) );
            }

      face.setOrientation( OrientationType( 0, 1, 0 ) );
      for( face.getCoordinates().z() = 0;
            face.getCoordinates().z() < zSize;
            face.getCoordinates().z()++ )
         for( face.getCoordinates().y() = 0;
              face.getCoordinates().y() < ySize + 1;
              face.getCoordinates().y()++ )
            for( face.getCoordinates().x() = 0;
                 face.getCoordinates().x() < xSize;
                 face.getCoordinates().x()++ )
            {
               const IndexType faceIndex = grid.template getEntityIndex( face );
               CPPUNIT_ASSERT( faceIndex >= 0 );
               CPPUNIT_ASSERT( faceIndex < grid.template getEntitiesCount< typename GridType::Face >() );
               CPPUNIT_ASSERT( grid.template getEntity< typename GridType::Face >( faceIndex ).getCoordinates() == face.getCoordinates() );
               CPPUNIT_ASSERT( grid.template getEntity< typename GridType::Face >( faceIndex ).getOrientation() == OrientationType( 0, 1, 0 ) );
               // TODO: fix this - gives undefined reference - I do not know why
               //CPPUNIT_ASSERT( grid.template getEntity< 1 >( faceIndex ).getBasis() == BasisType( 0, 1 ) );
            }

      face.setOrientation( OrientationType( 0, 0, 1 ) );
      for( face.getCoordinates().z() = 0;
            face.getCoordinates().z() < zSize + 1;
            face.getCoordinates().z()++ )
         for( face.getCoordinates().y() = 0;
              face.getCoordinates().y() < ySize;
              face.getCoordinates().y()++ )
            for( face.getCoordinates().x() = 0;
                 face.getCoordinates().x() < xSize;
                 face.getCoordinates().x()++ )
            {
               const IndexType faceIndex = grid.template getEntityIndex( face );
               CPPUNIT_ASSERT( faceIndex >= 0 );
               CPPUNIT_ASSERT( faceIndex < grid.template getEntitiesCount< typename GridType::Face >() );
               CPPUNIT_ASSERT( grid.template getEntity< typename GridType::Face >( faceIndex ).getCoordinates() == face.getCoordinates() );
               CPPUNIT_ASSERT( grid.template getEntity< typename GridType::Face >( faceIndex ).getOrientation() == OrientationType( 0, 0, 1 ) );
               // TODO: fix this - gives undefined reference - I do not know why
               //CPPUNIT_ASSERT( grid.template getEntity< 1 >( faceIndex ).getBasis() == BasisType( 0, 1 ) );
            }


   }

   void edgeIndexingTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      const IndexType zSize( 19 );
      GridType grid;
      grid.setDimensions( xSize, ySize, zSize );
 
      typedef typename GridType::template MeshEntity< 1 > EdgeType;
      typedef typename EdgeType::EntityOrientationType OrientationType;
      typedef typename EdgeType::EntityBasisType BasisType;
      EdgeType edge( grid );
 
      edge.setBasis( OrientationType( 1, 0, 0 ) );
      for( edge.getCoordinates().z() = 0;
           edge.getCoordinates().z() < zSize + 1;
           edge.getCoordinates().z()++ )
         for( edge.getCoordinates().y() = 0;
              edge.getCoordinates().y() < ySize + 1;
              edge.getCoordinates().y()++ )
            for( edge.getCoordinates().x() = 0;
                 edge.getCoordinates().x() < xSize;
                 edge.getCoordinates().x()++ )
            {
               const IndexType edgeIndex = grid.template getEntityIndex( edge );
               CPPUNIT_ASSERT( edgeIndex >= 0 );
               CPPUNIT_ASSERT( edgeIndex < grid.template getEntitiesCount< typename GridType::Edge >() );
               CPPUNIT_ASSERT( grid.template getEntity< typename GridType::Edge >( edgeIndex ).getCoordinates() == edge.getCoordinates() );
               //CPPUNIT_ASSERT( grid.template getEntity< 1 >( edgeIndex ).getOrientation() == OrientationType( 1, 0, 0 ) );
               // TODO: fix this - gives undefined reference - I do not know why
               //CPPUNIT_ASSERT( grid.template getEntity< 1 >( faceIndex ).getBasis() == BasisType( 0, 1 ) );
            }
 
      edge.setBasis( OrientationType( 0, 1, 0 ) );
      for( edge.getCoordinates().z() = 0;
           edge.getCoordinates().z() < zSize + 1;
           edge.getCoordinates().z()++ )
         for( edge.getCoordinates().y() = 0;
              edge.getCoordinates().y() < ySize;
              edge.getCoordinates().y()++ )
            for( edge.getCoordinates().x() = 0;
                 edge.getCoordinates().x() < xSize + 1;
                 edge.getCoordinates().x()++ )
            {
               const IndexType edgeIndex = grid.template getEntityIndex( edge );
               CPPUNIT_ASSERT( edgeIndex >= 0 );
               CPPUNIT_ASSERT( edgeIndex < grid.template getEntitiesCount< typename GridType::Edge >() );
               CPPUNIT_ASSERT( grid.template getEntity< typename GridType::Edge >( edgeIndex ).getCoordinates() == edge.getCoordinates() );
               //CPPUNIT_ASSERT( grid.template getEntity< 1 >( edgeIndex ).getOrientation() == OrientationType( 1, 0, 0 ) );
               // TODO: fix this - gives undefined reference - I do not know why
               //CPPUNIT_ASSERT( grid.template getEntity< 1 >( faceIndex ).getBasis() == BasisType( 0, 1 ) );
            }

      edge.setBasis( OrientationType( 0, 0, 1 ) );
      for( edge.getCoordinates().z() = 0;
           edge.getCoordinates().z() < zSize;
           edge.getCoordinates().z()++ )
         for( edge.getCoordinates().y() = 0;
              edge.getCoordinates().y() < ySize + 1;
              edge.getCoordinates().y()++ )
            for( edge.getCoordinates().x() = 0;
                 edge.getCoordinates().x() < xSize + 1;
                 edge.getCoordinates().x()++ )
            {
               const IndexType edgeIndex = grid.template getEntityIndex( edge );
               CPPUNIT_ASSERT( edgeIndex >= 0 );
               CPPUNIT_ASSERT( edgeIndex < grid.template getEntitiesCount< typename GridType::Edge >() );
               CPPUNIT_ASSERT( grid.template getEntity< typename GridType::Edge >( edgeIndex ).getCoordinates() == edge.getCoordinates() );
               //CPPUNIT_ASSERT( grid.template getEntity< 1 >( edgeIndex ).getOrientation() == OrientationType( 1, 0, 0 ) );
               // TODO: fix this - gives undefined reference - I do not know why
               //CPPUNIT_ASSERT( grid.template getEntity< 1 >( faceIndex ).getBasis() == BasisType( 0, 1 ) );
            }
   }

   void vertexIndexingTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      const IndexType zSize( 19 );
      GridType grid;
      grid.setDimensions( xSize, ySize, zSize );
 
      typedef typename GridType::template MeshEntity< 0 > VertexType;
      typedef typename VertexType::EntityOrientationType OrientationType;
      typedef typename VertexType::EntityBasisType BasisType;
      VertexType vertex( grid );
 
      for( vertex.getCoordinates().z() = 0;
           vertex.getCoordinates().z() < zSize + 1;
           vertex.getCoordinates().z()++ )
         for( vertex.getCoordinates().y() = 0;
              vertex.getCoordinates().y() < ySize + 1;
              vertex.getCoordinates().y()++ )
            for( vertex.getCoordinates().x() = 0;
                 vertex.getCoordinates().x() < xSize + 1;
                 vertex.getCoordinates().x()++ )
            {
               const IndexType vertexIndex = grid.getEntityIndex( vertex );
               CPPUNIT_ASSERT( vertexIndex >= 0 );
               CPPUNIT_ASSERT( vertexIndex < grid.template getEntitiesCount< typename GridType::Edge >() );
               CPPUNIT_ASSERT( grid.template getEntity< typename GridType::Edge >( vertexIndex ).getCoordinates() == vertex.getCoordinates() );
               //CPPUNIT_ASSERT( grid.template getEntity< 1 >( edgeIndex ).getOrientation() == OrientationType( 1, 0, 0 ) );
               // TODO: fix this - gives undefined reference - I do not know why
               //CPPUNIT_ASSERT( grid.template getEntity< 1 >( faceIndex ).getBasis() == BasisType( 0, 1 ) );
            }
   }

   void getCellNextToCellTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      const IndexType zSize( 19 );
      GridType grid;
      grid.setDimensions( xSize, ySize, zSize );
 
      typedef typename GridType::Cell CellType;
      CellType cell( grid );
      for( cell.getCoordinates().z() = 0;
           cell.getCoordinates().z() < zSize;
           cell.getCoordinates().z()++ )
         for( cell.getCoordinates().y() = 0;
              cell.getCoordinates().y() < ySize;
              cell.getCoordinates().y()++ )
            for( cell.getCoordinates().x() = 0;
                 cell.getCoordinates().x() < xSize;
                 cell.getCoordinates().x()++ )
            {
               const IndexType cellIndex = grid.getEntityIndex( cell );
               if( cell.getCoordinates().x() > 0 )
               {
                  CellType auxCell( grid, cell.getCoordinates() + CoordinatesType( -1, 0, 0 ) );
                  const IndexType auxCellIndex = grid.getEntityIndex( auxCell );
                  auto neighbourEntities = cell.getNeighbourEntities();
                  CPPUNIT_ASSERT( ( auxCellIndex == neighbourEntities.template getEntityIndex< -1, 0, 0 >() ) );
               }
               if( cell.getCoordinates().x() < xSize - 1 )
               {
                  CellType auxCell( grid, cell.getCoordinates() + CoordinatesType( 1, 0, 0 ) );
                  const IndexType auxCellIndex = grid.getEntityIndex( auxCell );
                  auto neighbourEntities = cell.getNeighbourEntities();
                  CPPUNIT_ASSERT( ( auxCellIndex == neighbourEntities.template getEntityIndex< 1, 0, 0 >() ) );
               }
               if( cell.getCoordinates().y() > 0 )
               {
                  CellType auxCell( grid, cell.getCoordinates() + CoordinatesType( 0, -1, 0 ) );
                  const IndexType auxCellIndex = grid.getEntityIndex( auxCell );
                  auto neighbourEntities = cell.getNeighbourEntities();
                  CPPUNIT_ASSERT( ( auxCellIndex == neighbourEntities.template getEntityIndex< 0, -1, 0 >() ) );
               }
               if( cell.getCoordinates().y() < ySize - 1 )
               {
                  CellType auxCell( grid, cell.getCoordinates() + CoordinatesType( 0, 1, 0 ) );
                  const IndexType auxCellIndex = grid.getEntityIndex( auxCell );
                  auto neighbourEntities = cell.getNeighbourEntities();
                  CPPUNIT_ASSERT( ( auxCellIndex == neighbourEntities.template getEntityIndex< 0, 1, 0 >() ) );
               }
               if( cell.getCoordinates().z() > 0 )
               {
                  CellType auxCell( grid, cell.getCoordinates() + CoordinatesType( 0, 0, -1 ) );
                  const IndexType auxCellIndex = grid.getEntityIndex( auxCell );
                  auto neighbourEntities = cell.getNeighbourEntities();
                  CPPUNIT_ASSERT( ( auxCellIndex == neighbourEntities.template getEntityIndex< 0, 0, -1 >() ) );
               }
               if( cell.getCoordinates().z() < zSize - 1 )
               {
                  CellType auxCell( grid, cell.getCoordinates() + CoordinatesType( 0, 0, 1 ) );
                  const IndexType auxCellIndex = grid.getEntityIndex( auxCell );
                  auto neighbourEntities = cell.getNeighbourEntities();
                  CPPUNIT_ASSERT( ( auxCellIndex == neighbourEntities.template getEntityIndex< 0, 0, 1 >() ) );
               }
            }
   }

   void getFaceNextToCellTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      const IndexType zSize( 19 );
      GridType grid;
      grid.setDimensions( xSize, ySize, zSize );
 
      typedef typename GridType::Cell CellType;
      typedef typename GridType::Face FaceType;
      typedef typename FaceType::EntityOrientationType EntityOrientationType;
      CellType cell( grid );
      FaceType face( grid );
      for( cell.getCoordinates().z() = 0;
           cell.getCoordinates().z() < zSize;
           cell.getCoordinates().z()++ )
         for( cell.getCoordinates().y() = 0;
              cell.getCoordinates().y() < ySize;
              cell.getCoordinates().y()++ )
            for( cell.getCoordinates().x() = 0;
                 cell.getCoordinates().x() < xSize;
                 cell.getCoordinates().x()++ )
            {
               //const IndexType cellIndex = grid.getEntityIndex( cell );
               cell.refresh();//setIndex( cellIndex );
               auto neighbourEntities = cell.template getNeighbourEntities< GridType::Face::entityDimensions >();
 

               face.setCoordinates( cell.getCoordinates() );
               face.setOrientation( EntityOrientationType( 1, 0, 0 ) );
               //CoordinatesType faceCoordinates( i, j, k );
               IndexType faceIndex = grid.getEntityIndex( face );
               CPPUNIT_ASSERT( ( faceIndex == neighbourEntities.template getEntityIndex< -1, 0, 0 >() ) );

               //faceCoordinates = CoordinatesType( i + 1, j, k );
               face.setCoordinates( cell.getCoordinates() + CoordinatesType( 1, 0, 0 ) );
               face.setOrientation( EntityOrientationType( 1, 0 , 0 ) );
               faceIndex = grid.getEntityIndex( face );
               CPPUNIT_ASSERT( ( faceIndex == neighbourEntities.template getEntityIndex< 1, 0, 0 >() ) );

               //faceCoordinates = CoordinatesType( i, j, k );
               face.setCoordinates( cell.getCoordinates() );
               face.setOrientation( EntityOrientationType( 0, -1, 0 ) );
               faceIndex = grid.getEntityIndex( face );
               CPPUNIT_ASSERT( ( faceIndex == neighbourEntities.template getEntityIndex< 0, -1, 0 >() ) );

               //faceCoordinates = CoordinatesType( i, j + 1, k );
               face.setCoordinates( cell.getCoordinates() + CoordinatesType( 0, 1, 0 ) );
               face.setOrientation( EntityOrientationType( 0, 1, 0 ) );
               faceIndex = grid.getEntityIndex( face );
               CPPUNIT_ASSERT( ( faceIndex == neighbourEntities.template getEntityIndex< 0, 1, 0 >() ) );

               //faceCoordinates = CoordinatesType( i, j, k );
               face.setCoordinates( cell.getCoordinates() );
               face.setOrientation( EntityOrientationType( 0, 0, -1 ) );
               faceIndex = grid.getEntityIndex( face );
               CPPUNIT_ASSERT( ( faceIndex == neighbourEntities.template getEntityIndex< 0, 0, -1 >() ) );

               //faceCoordinates = CoordinatesType( i, j, k + 1 );
               face.setCoordinates( cell.getCoordinates() + CoordinatesType( 0, 0, 1 ) );
               face.setOrientation( EntityOrientationType( 0, 0, 1 ) );
               faceIndex = grid.getEntityIndex( face );
               CPPUNIT_ASSERT( ( faceIndex == neighbourEntities.template getEntityIndex< 0, 0, 1 >() ) );
            }
   }

   void getCellNextToFaceTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      const IndexType zSize( 19 );
      GridType grid;
      grid.setDimensions( xSize, ySize, zSize );
 
      typedef typename GridType::Cell CellType;
      typedef typename GridType::Face FaceType;
      typedef typename FaceType::EntityOrientationType EntityOrientationType;
      CellType cell( grid );
      FaceType face( grid );

      for( face.getCoordinates().z() = 0;
           face.getCoordinates().z() <= zSize;
           face.getCoordinates().z()++ )
         for( face.getCoordinates().y() = 0;
              face.getCoordinates().y() <= ySize;
              face.getCoordinates().y()++ )
            for( face.getCoordinates().x() = 0;
                 face.getCoordinates().x() <= xSize;
                 face.getCoordinates().x()++ )
            {
               //const CoordinatesType faceCoordinates( i, j, k );
               if( face.getCoordinates().y() < ySize && face.getCoordinates().z() < zSize )
               {
                  face.setOrientation( EntityOrientationType( 1, 0, 0  ) );
                  //const IndexType faceIndex = grid.getEntityIndex( face );
                  face.refresh();//setIndex( faceIndex );
                  auto neighbourEntities = face.template getNeighbourEntities< GridType::Cell::entityDimensions >();


                  if( face.getCoordinates().x() > 0 )
                  {
                     CellType cell( grid, face.getCoordinates() + CoordinatesType( -1, 0, 0 ) );
                     IndexType cellIndex = grid.getEntityIndex( cell );
                     CPPUNIT_ASSERT( ( cellIndex == neighbourEntities.template getEntityIndex< -1, 0, 0 >() ) );
                  }
                  if( face.getCoordinates().x() < xSize )
                  {
                     CellType cell( grid, face.getCoordinates() );
                     IndexType cellIndex = grid.getEntityIndex( cell );
                     CPPUNIT_ASSERT( ( cellIndex == neighbourEntities.template getEntityIndex< 1, 0, 0 >() ) );
                  }
               }
               if( face.getCoordinates().x() < xSize && face.getCoordinates().z() < zSize )
               {
                  face.setOrientation( EntityOrientationType( 0, 1, 0  ) );
                  //const IndexType faceIndex = grid.getEntityIndex( face );
                  face.refresh();//setIndex( faceIndex );
                  auto neighbourEntities = face.template getNeighbourEntities< GridType::Cell::entityDimensions >();
 
                  if( face.getCoordinates().y() > 0 )
                  {
                     CellType cell( grid, face.getCoordinates() + CoordinatesType( 0, -1, 0 ) );
                     IndexType cellIndex = grid.getEntityIndex( cell );
                     CPPUNIT_ASSERT( ( cellIndex == neighbourEntities.template getEntityIndex< 0, -1, 0 >() ) );
                  }
                  if( face.getCoordinates().y() < ySize )
                  {
                     CellType cell( grid, face.getCoordinates() );
                     IndexType cellIndex = grid.getEntityIndex( cell );
                     CPPUNIT_ASSERT( ( cellIndex == neighbourEntities.template getEntityIndex< 0, 1, 0 >() ) );
                  }
               }
               if( face.getCoordinates().x() < xSize && face.getCoordinates().y() < ySize )
               {
                  face.setOrientation( EntityOrientationType( 0, 0, 1  ) );
                  //const IndexType faceIndex = grid.getEntityIndex( face );
                  face.refresh();//setIndex( faceIndex );
                  auto neighbourEntities = face.template getNeighbourEntities< GridType::Cell::entityDimensions >();
 
                  if( face.getCoordinates().z() > 0 )
                  {
                     CellType cell( grid, face.getCoordinates() + CoordinatesType( 0, 0, -1 ) );
                     IndexType cellIndex = grid.getEntityIndex( cell );
                     CPPUNIT_ASSERT( ( cellIndex == neighbourEntities.template getEntityIndex< 0, 0, -1 >() ) );
                  }
                  if( face.getCoordinates().z() < zSize )
                  {
                     CellType cell( grid, face.getCoordinates() );
                     IndexType cellIndex = grid.getEntityIndex( cell );
                     CPPUNIT_ASSERT( ( cellIndex == neighbourEntities.template getEntityIndex< 0, 0, 1 >() ) );
                  }
               }
            }
   }
};

#endif /* TESTS_UNIT_TESTS_MESH_TNLGRID3DTESTER_H_ */
