/***************************************************************************
                          Grid2DTester.h  -  description
                             -------------------
    begin                : Feb 13, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TESTS_UNIT_TESTS_MESH_TNLGRID2DTESTER_H_
#define TESTS_UNIT_TESTS_MESH_TNLGRID2DTESTER_H_

using namespace TNL;

template< typename RealType, typename Device, typename IndexType >
class GridTester< 2, RealType, Device, IndexType >: public CppUnit :: TestCase
{
   public:
   typedef GridTester< 2, RealType, Device, IndexType > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;
   typedef Meshes::Grid< 2, RealType, Device, IndexType > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;
   typedef typename GridType::PointType PointType;


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
      suiteOfTests -> addTest( new TestCallerType( "vertexIndexingTest", &TesterType::vertexIndexingTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getCellNextToCellTest", &TesterType::getCellNextToCellTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getFaceNextToCellTest", &TesterType::getFaceNextToCellTest ) );
      suiteOfTests -> addTest( new TestCallerType( "getCellNextToFaceTest", &TesterType::getCellNextToFaceTest ) );

      return suiteOfTests;
   }

   void setDomainTest()
   {
      GridType grid;
      grid.setDomain( PointType( 0.0, 0.0 ), PointType( 1.0, 1.0 ) );
      grid.setDimensions( 10, 20 );

      CPPUNIT_ASSERT( grid.getSpaceSteps().x() == 0.1 );
      CPPUNIT_ASSERT( grid.getSpaceSteps().y() == 0.05 );
   }

   void cellIndexingTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      GridType grid;
      grid.setDimensions( xSize, ySize );
      typename GridType::Cell cell( grid );
      for( cell.getCoordinates().y() = 0;
           cell.getCoordinates().y() < ySize;
           cell.getCoordinates().y()++ )
         for( cell.getCoordinates().x() = 0;
              cell.getCoordinates().x() < xSize;
              cell.getCoordinates().x()++ )
         {
            const IndexType cellIndex = grid.getEntityIndex( cell );
            CPPUNIT_ASSERT( cellIndex >= 0 );
            CPPUNIT_ASSERT( cellIndex < grid.template getEntitiesCount< typename GridType::Cell >() );
            CPPUNIT_ASSERT( grid.template getEntity< typename GridType::Cell >( cellIndex ).getCoordinates() == cell.getCoordinates() );
         }
   }

   void faceIndexingTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      GridType grid;
      grid.setDimensions( xSize, ySize );

      typedef typename GridType::template MeshEntity< 1 > FaceType;
      typedef typename FaceType::EntityOrientationType OrientationType;
      typedef typename FaceType::EntityBasisType BasisType;
      FaceType face( grid );
 
      face.setOrientation( OrientationType( 1, 0 ) );
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
            CPPUNIT_ASSERT( grid.template getEntity< typename GridType::Face >( faceIndex ).getOrientation() == OrientationType( 1, 0 ) );
            // TODO: fix this - gives undefined reference - I do not know why
            //CPPUNIT_ASSERT( grid.template getEntity< 1 >( faceIndex ).getBasis() == BasisType( 0, 1 ) );
         }

      face.setOrientation( OrientationType( 0, 1 ) );
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
            CPPUNIT_ASSERT( grid.template getEntity< typename GridType::Face >( faceIndex ).getOrientation() == OrientationType( 0, 1 ) );
            // TODO: fix this - gives undefined reference - I do not know why
            //CPPUNIT_ASSERT( grid.template getEntity< 1 >( faceIndex ).getBasis() == BasisType( 1, 0 ) );
         }

   }

   void vertexIndexingTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      GridType grid;
 
      typedef typename GridType::template MeshEntity< 0 > PointType;
      typedef typename PointType::EntityBasisType BasisType;
      PointType vertex( grid );
 
      CoordinatesType& vertexCoordinates = vertex.getCoordinates();
      grid.setDimensions( xSize, ySize );
      for( vertex.getCoordinates().y() = 0;
           vertex.getCoordinates().y() < ySize + 1;
           vertex.getCoordinates().y()++ )
         for( vertex.getCoordinates().x() = 0;
              vertex.getCoordinates().x() < xSize + 1;
              vertex.getCoordinates().x()++ )
         {
            const IndexType vertexIndex = grid.template getEntityIndex< typename GridType::Point >( vertex );
            CPPUNIT_ASSERT( vertexIndex >= 0 );
            CPPUNIT_ASSERT( vertexIndex < grid.template getEntitiesCount< typename GridType::Point >() );
            CPPUNIT_ASSERT( grid.template getEntity< typename GridType::Point >( vertexIndex ).getCoordinates() == vertex.getCoordinates() );
         }
   }

   void getCellNextToCellTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      GridType grid;
      grid.setDimensions( xSize, ySize );
 
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::Cell CellType;
      CellType cell( grid );

      for( cell.getCoordinates().y() = 0;
           cell.getCoordinates().y() < ySize;
           cell.getCoordinates().y()++ )
         for( cell.getCoordinates().x() = 0;
              cell.getCoordinates().x() < xSize;
              cell.getCoordinates().x()++ )
         {
            //const IndexType cellIndex = grid.getEntityIndex( cell );
            cell.refresh();
            if( cell.getCoordinates().x() > 0 )
            {
               const CellType auxCell( grid, cell.getCoordinates() + CoordinatesType( -1, 0 ) );
               const IndexType auxCellIndex = grid.getEntityIndex( auxCell );
               auto neighbourEntities = cell.getNeighbourEntities();
               CPPUNIT_ASSERT( ( auxCellIndex == neighbourEntities.template getEntityIndex< -1, 0 >() ) );
            }
            if( cell.getCoordinates().x() < xSize - 1 )
            {
               const CellType auxCell( grid, cell.getCoordinates() + CoordinatesType( 1, 0 ) );
               const IndexType auxCellIndex = grid.getEntityIndex( auxCell );
               auto neighbourEntities = cell.getNeighbourEntities();
               CPPUNIT_ASSERT( ( auxCellIndex == neighbourEntities.template getEntityIndex< 1, 0 >() ) );
            }
            if( cell.getCoordinates().y() > 0 )
            {
               const CellType auxCell( grid, cell.getCoordinates() + CoordinatesType( 0, -1 ) );
               const IndexType auxCellIndex = grid.getEntityIndex( auxCell );
               auto neighbourEntities = cell.getNeighbourEntities();
               CPPUNIT_ASSERT( ( auxCellIndex == neighbourEntities.template getEntityIndex< 0, -1 >() ) );
            }
            if( cell.getCoordinates().y() < ySize - 1 )
            {
               const CellType auxCell( grid, cell.getCoordinates() + CoordinatesType( 0, 1 ) );
               const IndexType auxCellIndex = grid.getEntityIndex( auxCell );
               auto neighbourEntities = cell.getNeighbourEntities();
               CPPUNIT_ASSERT( ( auxCellIndex == neighbourEntities.template getEntityIndex< 0, 1 >() ) );
            }
         }
   }

   void getFaceNextToCellTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      GridType grid;
      grid.setDimensions( xSize, ySize );
 
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::Cell CellType;
      typedef typename GridType::Face FaceType;
      typedef typename FaceType::EntityOrientationType EntityOrientationType;
      typedef typename FaceType::EntityBasisType EntityBasisType;
      CellType cell( grid );

      for( cell.getCoordinates().y() = 0;
           cell.getCoordinates().y() < ySize;
           cell.getCoordinates().y()++ )
         for( cell.getCoordinates().x() = 0;
              cell.getCoordinates().x() < xSize;
              cell.getCoordinates().x()++ )
         {
            //const IndexType cellIndex = grid.getEntityIndex( cell );
            cell.refresh(); //setIndex( cellIndex );
            auto neighbourEntities = cell.template getNeighbourEntities< GridType::Face::entityDimension >();

            FaceType face1( grid,
                            cell.getCoordinates(),
                            EntityOrientationType( -1, 0 ),
                            EntityBasisType( 0, 1 ) );
            IndexType face1Index = grid.template getEntityIndex( face1 );
            CPPUNIT_ASSERT( ( face1Index == neighbourEntities.template getEntityIndex< -1, 0 >() ) );

            FaceType face2( grid,
                            cell.getCoordinates() + CoordinatesType( 1, 0 ),
                            EntityOrientationType( 1, 0 ),
                            EntityBasisType( 0, 1 ) );
            IndexType face2Index = grid.template getEntityIndex( face2 );
            CPPUNIT_ASSERT( ( face2Index == neighbourEntities.template getEntityIndex< 1, 0 >() ) );

            FaceType face3( grid,
                            cell.getCoordinates(),
                            EntityOrientationType( 0, -1 ),
                            EntityBasisType( 1, 0 ) );
            IndexType face3Index = grid.template getEntityIndex( face3 );
            CPPUNIT_ASSERT( ( face3Index == neighbourEntities.template getEntityIndex< 0, -1 >() ) );
 
            FaceType face4( grid,
                            cell.getCoordinates() + CoordinatesType( 0, 1 ),
                            EntityOrientationType( 0, 1 ),
                            EntityBasisType( 1, 0 ) );
            IndexType face4Index = grid.template getEntityIndex( face4 );
            CPPUNIT_ASSERT( ( face4Index == neighbourEntities.template getEntityIndex< 0, 1 >() ) );
         }
   }

   void getCellNextToFaceTest()
   {
      const IndexType xSize( 13 );
      const IndexType ySize( 17 );
      GridType grid;
      grid.setDimensions( xSize, ySize );
 
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::Cell CellType;
      typedef typename GridType::Face FaceType;
      typedef typename FaceType::EntityOrientationType EntityOrientationType;
      FaceType face( grid );

      for( face.getCoordinates().y() = 0;
           face.getCoordinates().y() <= ySize;
           face.getCoordinates().y()++ )
         for( face.getCoordinates().x() = 0;
              face.getCoordinates().x() <= xSize;
              face.getCoordinates().x()++ )
         {
            if( face.getCoordinates().y() < ySize )
            {
               face.setOrientation( EntityOrientationType( 1, 0 ) );
               //const IndexType faceIndex = grid.getEntityIndex( face );
               face.refresh(); //setIndex( faceIndex );
               auto neighbourCells = face.template getNeighbourEntities< GridType::Cell::entityDimension >();


               if( face.getCoordinates().x() > 0 )
               {
                  CellType cell( grid, face.getCoordinates() + CoordinatesType( -1, 0 ) );
                  IndexType cellIndex = grid.getEntityIndex( cell );
                  CPPUNIT_ASSERT( ( cellIndex == neighbourCells.template getEntityIndex< -1, 0 >() ) );
               }
               if( face.getCoordinates().x() < xSize )
               {
                  CellType cell( grid, face.getCoordinates() + CoordinatesType( 0, 0 ) );
                  IndexType cellIndex = grid.getEntityIndex( cell );
                  CPPUNIT_ASSERT( ( cellIndex == neighbourCells.template getEntityIndex< 1, 0 >() ) );
               }
            }
            if( face.getCoordinates().x() < xSize )
            {
               face.setOrientation( EntityOrientationType( 0, 1 ) );
               //const IndexType faceIndex = grid.getEntityIndex( face );
               face.refresh();//setIndex( faceIndex );
               auto neighbourCells = face.template getNeighbourEntities< GridType::Cell::entityDimension >();
 
               if( face.getCoordinates().y() > 0 )
               {
                  CellType cell( grid, face.getCoordinates() + CoordinatesType( 0, -1 ) );
                  IndexType cellIndex = grid.getEntityIndex( cell );
                  CPPUNIT_ASSERT( ( cellIndex == neighbourCells.template getEntityIndex< 0, -1 >() ) );
               }
               if( face.getCoordinates().y() < ySize )
               {
                  CellType cell( grid, face.getCoordinates() + CoordinatesType( 0, 0 ) );
                  IndexType cellIndex = grid.getEntityIndex( cell );
                  CPPUNIT_ASSERT( ( cellIndex == neighbourCells.template getEntityIndex< 0, 1 >() ) );
               }
            }
         }
   }
};

#endif /* TESTS_UNIT_TESTS_MESH_TNLGRID2DTESTER_H_ */
