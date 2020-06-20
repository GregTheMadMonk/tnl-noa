/***************************************************************************
                          DistributedGridTest.cpp  -  description
                             -------------------
    begin                : Sep 6, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/


#ifdef HAVE_GTEST 
#include <gtest/gtest.h>

#ifdef HAVE_MPI    

#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Functions/MeshFunctionView.h>
#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Meshes/DistributedMeshes/SubdomainOverlapsGetter.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMeshSynchronizer.h>

#include "../../Functions/Functions.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Functions;
using namespace TNL::Devices;
using namespace TNL::Communicators;
using namespace TNL::Meshes::DistributedMeshes;

 

template<typename DofType>
void setDof_2D( DofType &dof, typename DofType::RealType value )
{
   for( int i = 0; i < dof.getSize(); i++ )
      dof[ i ] = value;
}

template<typename DofType,typename GridType>
void checkLeftEdge( const GridType &grid, const DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue )
{
    int maxx = grid.getDimensions().x();
    int maxy = grid.getDimensions().y();
    int begin = 0;
    int end = maxy;
    if( !with_first ) begin++;
    if( !with_last ) end--;
    
    for( int i=begin;i<end;i++ )
            EXPECT_EQ( dof[maxx*i], expectedValue) << "Left Edge test failed " << i<<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkRightEdge(const GridType &grid, const DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue)
{
    int maxx = grid.getDimensions().x();
    int maxy = grid.getDimensions().y();
    int begin = 0;
    int end = maxy;
    if( !with_first ) begin++;
    if( !with_last ) end--;
    
    for( int i = begin; i < end; i++ ) 
            EXPECT_EQ( dof[maxx*i+(maxx-1)], expectedValue) << "Right Edge test failed " << i <<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkUpEdge( const GridType &grid, const DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue )
{
    int maxx = grid.getDimensions().x();
    int maxy = grid.getDimensions().y();
    int begin = 0;
    int end = maxx;
    if( !with_first ) begin++;
    if( !with_last ) end--;
    
    for( int i=begin; i<end; i++ )
            EXPECT_EQ( dof[i], expectedValue) << "Up Edge test failed " << i<<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkDownEdge( const GridType &grid, const DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue )
{
    int maxx = grid.getDimensions().x();
    int maxy = grid.getDimensions().y();
    int begin = 0;
    int end = maxx;
    if( !with_first ) begin++;
    if( !with_last ) end--;
    
    for( int i=begin; i<end; i++ )
            EXPECT_EQ( dof[maxx*(maxy-1)+i], expectedValue) << "Down Edge test failed " << i<<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkLeftBoundary( const GridType &grid, const DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue )
{
   int maxx = grid.getDimensions().x();
   int maxy = grid.getDimensions().y();
   int begin = 1;
   int end = maxy - 1;
   if( !with_first ) begin++;
   if( !with_last ) end--;
    
   for( int i=begin;i<end;i++ )
      EXPECT_EQ( dof[ maxx * i + 1 ], expectedValue) << "Left Edge test failed " << i<<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkRightBoundary(const GridType &grid, const DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue)
{
   int maxx = grid.getDimensions().x();
   int maxy = grid.getDimensions().y();
   int begin = 1;
   int end = maxy - 1;
   if( !with_first ) begin++;
   if( !with_last ) end--;
    
   for( int i = begin; i < end; i++ ) 
     EXPECT_EQ( dof[ maxx * i + ( maxx - 2 ) ], expectedValue) << "Right Edge test failed " << i <<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkUpBoundary( const GridType &grid, const DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue )
{
   int maxx = grid.getDimensions().x();
   int maxy = grid.getDimensions().y();
   int begin = 1;
   int end = maxx - 1;
   if( !with_first ) begin++;
   if( !with_last ) end--;
    
   for( int i=begin; i<end; i++ )
      EXPECT_EQ( dof[ maxx + i ], expectedValue) << "Up Edge test failed " << i<<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkDownBoundary( const GridType &grid, const DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue )
{
   int maxx = grid.getDimensions().x();
   int maxy = grid.getDimensions().y();
   int begin = 1;
   int end = maxx - 1;
   if( !with_first ) begin++;
   if( !with_last ) end--;
   
   for( int i=begin; i<end; i++ )
      EXPECT_EQ( dof[ maxx * ( maxy-2 ) + i ], expectedValue) << "Down Edge test failed " << i<<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkCorner(const GridType &grid, const DofType &dof, bool up, bool left, typename DofType::RealType expectedValue )
{
    int maxx=grid.getDimensions().x();
    int maxy=grid.getDimensions().y();
    if(up&&left)
    {
        EXPECT_EQ( dof[0], expectedValue) << "Up Left Conner test failed ";
    }
    if(up && !left)
    {
        EXPECT_EQ( dof[maxx-1], expectedValue) << "Up Right Conner test failed ";
    }
    if(!up && left)
    {
        EXPECT_EQ( dof[(maxy-1)*maxx], expectedValue) << "Down Left Conner test failed ";
    }
    if(!up && !left)
    {
        EXPECT_EQ( dof[(maxy-1)*maxx+maxx-1], expectedValue) << "Down right Conner test failed ";
    }
}


/*expecting 9 processes*/
template<typename DofType,typename GridType>
void check_Boundary_2D(int rank, const GridType &grid, const DofType &dof, typename DofType::RealType expectedValue)
{    

    if(rank==0)//Up Left
    {
        checkUpEdge(grid,dof,true,false,expectedValue);//posledni je overlap
        checkLeftEdge(grid,dof,true,false, expectedValue);//posledni je overlap
    }
    
    if(rank==1)//Up Center
    {
        checkUpEdge(grid,dof,false,false, expectedValue);//prvni a posledni je overlap
    }
    
    if(rank==2)//Up Right
    {
        checkUpEdge(grid,dof,false,true,expectedValue);//prvni je overlap
        checkRightEdge(grid,dof,true,false,expectedValue);//posledni je overlap
    }
    
    if(rank==3)//Center Left
    {
        checkLeftEdge(grid,dof,false,false,expectedValue);//prvni a posledni je overlap
    }
    
    if(rank==4)//Center Center
    {
        //No boundary
    }
    
    if(rank==5)//Center Right
    {
        checkRightEdge(grid,dof,false,false,expectedValue);
    }
    
    if(rank==6)//Down Left
    {
        checkDownEdge(grid,dof,true,false,expectedValue);
        checkLeftEdge(grid,dof,false,true,expectedValue);
    }
    
    if(rank==7) //Down Center
    {
        checkDownEdge(grid,dof,false,false,expectedValue);
    }
    
    if(rank==8) //Down Right
    {
            checkDownEdge(grid,dof,false,true,expectedValue);
            checkRightEdge(grid,dof,false,true,expectedValue);
    }
};

/*expecting 9 processes
 * Known BUG of Traversars: Process boundary is writing over overlap.
 * it should be true, true, every where, but we dont check boundary overalp on boundary
 * so boundary overlap is not checked (it is filled incorectly by boundary condition).
 */
template<typename DofType,typename GridType>
void check_Overlap_2D(int rank, const GridType &grid, const DofType &dof, typename DofType::RealType expectedValue)
{
    if(rank==0)//Up Left
    {
        checkRightEdge(grid,dof,false,true,expectedValue);
        checkDownEdge(grid,dof,false,true,expectedValue);
    }
    
    if(rank==1)//Up Center
    {
        checkDownEdge(grid,dof,true,true,expectedValue);
        checkLeftEdge(grid,dof,false,true,expectedValue);
        checkRightEdge(grid,dof,false,true,expectedValue);
    }
    
    if(rank==2)//Up Right
    {
        checkDownEdge(grid,dof,true,false,expectedValue);//prvni je overlap
        checkLeftEdge(grid,dof,false,true,expectedValue);
    }
    
    if(rank==3)//Center Left
    {
        checkUpEdge(grid,dof,false,true,expectedValue);
        checkDownEdge(grid,dof,false,true,expectedValue);
        checkRightEdge(grid,dof,true,true,expectedValue);
    }
    
    if(rank==4)//Center Center
    {
        checkUpEdge(grid,dof,true,true,expectedValue);
        checkDownEdge(grid,dof,true,true,expectedValue);
        checkRightEdge(grid,dof,true,true,expectedValue);
        checkLeftEdge(grid,dof,true,true,expectedValue);
    }
    
    if(rank==5)//Center Right
    {
        checkUpEdge(grid,dof,true,false,expectedValue);
        checkDownEdge(grid,dof,true,false,expectedValue);
        checkLeftEdge(grid,dof,true,true,expectedValue);
    }
    
    if(rank==6)//Down Left
    {
        checkUpEdge(grid,dof,false,true,expectedValue);
        checkRightEdge(grid,dof,true,false,expectedValue);
    }
    
    if(rank==7) //Down Center
    {
        checkUpEdge(grid,dof,true,true,expectedValue);
        checkLeftEdge(grid,dof,true,false,expectedValue);
        checkRightEdge(grid,dof,true,false,expectedValue);
    }
    
    if(rank==8) //Down Right
    {
        checkUpEdge(grid,dof,true,false,expectedValue);
        checkLeftEdge(grid,dof,true,false,expectedValue);
    }
}



template<typename DofType,typename GridType>
void check_Inner_2D(int rank, const GridType& grid, const DofType& dof, typename DofType::RealType expectedValue)
{
    int maxx=grid.getDimensions().x();
    int maxy=grid.getDimensions().y();
    for(int j=1;j<maxy-1;j++)//prvni a posledni jsou buď hranice, nebo overlap
        for(int i=1;i<maxx-1;i++) //buď je vlevo hranice, nebo overlap
            EXPECT_EQ( dof[j*maxx+i], expectedValue) << " "<< j<<" "<<i << " " << maxx << " " << maxy;
}

/*
 * Light check of 2D distributed grid and its synchronization. 
 * expected 9 processes
 */
typedef MpiCommunicator CommunicatorType;
typedef Grid<2,double,Host,int> GridType;
typedef MeshFunctionView<GridType> MeshFunctionType;
typedef MeshFunctionView< GridType, GridType::getMeshDimension(), bool > MaskType;
typedef Vector<double,Host,int> DofType;
typedef Vector< bool, Host, int > MaskDofType;
typedef typename GridType::Cell Cell;
typedef typename GridType::IndexType IndexType; 
typedef typename GridType::PointType PointType; 
typedef DistributedMesh<GridType> DistributedGridType;
using Synchronizer = DistributedMeshSynchronizer< MeshFunctionType >;

class DistributedGridTest_2D : public ::testing::Test
{
    
   public:
      
      using CoordinatesType = typename GridType::CoordinatesType;

      DistributedGridType *distributedGrid;
      DofType *dof;
      MaskDofType maskDofs;

      Pointers::SharedPointer<GridType> gridPtr;
      Pointers::SharedPointer<MeshFunctionType> meshFunctionPtr;
      Pointers::SharedPointer< MaskType > maskPointer;

      MeshFunctionEvaluator< MeshFunctionType, ConstFunction<double,2> > constFunctionEvaluator;
      Pointers::SharedPointer< ConstFunction<double,2>, Host > constFunctionPtr;

      MeshFunctionEvaluator< MeshFunctionType, LinearFunction<double,2> > linearFunctionEvaluator;
      Pointers::SharedPointer< LinearFunction<double,2>, Host > linearFunctionPtr;

      int rank;
      int nproc;    

      void SetUp()
      {
         int size=10;
         rank=CommunicatorType::GetRank(CommunicatorType::AllGroup);
         nproc=CommunicatorType::GetSize(CommunicatorType::AllGroup);

         PointType globalOrigin;
         PointType globalProportions;
         GridType globalGrid;

         globalOrigin.x()=-0.5;
         globalOrigin.y()=-0.5;    
         globalProportions.x()=size;
         globalProportions.y()=size;

         globalGrid.setDimensions(size,size);
         globalGrid.setDomain(globalOrigin,globalProportions);

         distributedGrid=new DistributedGridType();
         distributedGrid->setDomainDecomposition( typename DistributedGridType::CoordinatesType( 3, 3 ) );
         distributedGrid->template setGlobalGrid<CommunicatorType>( globalGrid );
         typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
         SubdomainOverlapsGetter< GridType, CommunicatorType >::
            getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1 );
         distributedGrid->setOverlaps( lowerOverlap, upperOverlap );
         distributedGrid->setupGrid(*gridPtr);

         dof=new DofType(gridPtr->template getEntitiesCount< Cell >());

         meshFunctionPtr->bind(gridPtr,*dof);

         constFunctionPtr->Number=rank;
      }

      void TearDown()
      {
         delete dof;
         delete distributedGrid;
      }
};

TEST_F(DistributedGridTest_2D, evaluateAllEntities)
{
   //Check Traversars
   //All entities, without overlap
   setDof_2D(*dof,-1);
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   //Printer<GridType,DofType>::print_dof(rank,*gridPtr,*dof);
   check_Boundary_2D(rank, *gridPtr, *dof, rank);
   check_Overlap_2D(rank, *gridPtr, *dof, -1);
   check_Inner_2D(rank, *gridPtr, *dof, rank);
}


TEST_F(DistributedGridTest_2D, evaluateBoundaryEntities)
{
    //Boundary entities, without overlap
    setDof_2D(*dof,-1);
    constFunctionEvaluator.evaluateBoundaryEntities( meshFunctionPtr , constFunctionPtr );
    //print_dof_2D(rank,*gridPtr,dof);
    check_Boundary_2D(rank, *gridPtr, *dof, rank);
    check_Overlap_2D(rank, *gridPtr, *dof, -1);
    check_Inner_2D(rank, *gridPtr, *dof, -1);
}

TEST_F(DistributedGridTest_2D, evaluateInteriorEntities)
{
    //Inner entities, without overlap
    setDof_2D(*dof,-1);
    constFunctionEvaluator.evaluateInteriorEntities( meshFunctionPtr , constFunctionPtr );
    check_Boundary_2D(rank, *gridPtr, *dof, -1);
    check_Overlap_2D(rank, *gridPtr, *dof, -1);
    check_Inner_2D(rank, *gridPtr, *dof, rank);
}    

TEST_F(DistributedGridTest_2D, LinearFunctionTest)
{
    //fill meshfunction with linear function (physical center of cell corresponds with its coordinates in grid) 
    setDof_2D(*dof,-1);
    linearFunctionEvaluator.evaluateAllEntities(meshFunctionPtr, linearFunctionPtr);
    Synchronizer synchronizer;
    synchronizer.setDistributedGrid( meshFunctionPtr->getMesh().getDistributedMesh() );
    synchronizer.template synchronize<CommunicatorType>( *meshFunctionPtr );
    
    int count =gridPtr->template getEntitiesCount< Cell >();
    for(int i=0;i<count;i++)
    {
            auto entity= gridPtr->template getEntity< Cell >(i);
            entity.refresh();
            EXPECT_EQ(meshFunctionPtr->getValue(entity), (*linearFunctionPtr)(entity)) << "Linear function doesnt fit recievd data. " << entity.getCoordinates().x() << " "<<entity.getCoordinates().y() << " "<< gridPtr->getDimensions().x() <<" "<<gridPtr->getDimensions().y();
    }
}

TEST_F(DistributedGridTest_2D, SynchronizerNeighborTest )
{
   //Expect 9 processes
   setDof_2D(*dof,-1);
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   Synchronizer synchronizer;
   synchronizer.setDistributedGrid( meshFunctionPtr->getMesh().getDistributedMesh() );
   synchronizer.template synchronize<CommunicatorType>( *meshFunctionPtr );
    
   // checkNeighbor_2D(rank, *gridPtr, *dof);
   
    if(rank==0)//Up Left
    {
        checkRightEdge(*gridPtr, *dof, true,  false, 1 );
        checkDownEdge( *gridPtr, *dof, true,  false, 3 );
        checkCorner(   *gridPtr, *dof, false, false, 4 );
    }
    
    if(rank==1)//Up Center
    {
        checkLeftEdge( *gridPtr, *dof, true,  false, 0 );
        checkRightEdge(*gridPtr, *dof, true,  false, 2 );
        checkCorner(   *gridPtr, *dof, false, true,  3 );
        checkDownEdge( *gridPtr, *dof, false, false, 4 );
        checkCorner(   *gridPtr, *dof, false, false, 5 );
    }
    
    if(rank==2)//Up Right
    {
        checkLeftEdge( *gridPtr, *dof, true,  false, 1 );
        checkCorner(   *gridPtr, *dof, false, true,  4 );
        checkDownEdge( *gridPtr, *dof, false, true,  5 );
    }
    
    if(rank==3)//Center Left
    {
        checkUpEdge(    *gridPtr, *dof, true,  false, 0 );
        checkCorner(    *gridPtr, *dof, true,  false, 1 );
        checkRightEdge( *gridPtr, *dof, false, false, 4 );
        checkDownEdge(  *gridPtr, *dof, true,  false, 6 );
        checkCorner(    *gridPtr, *dof, false, false, 7 );
    }
    
    if(rank==4)//Center Center
    {
        checkCorner(    *gridPtr, *dof, true,  true,  0 );
        checkUpEdge(    *gridPtr, *dof, false, false, 1 );
        checkCorner(    *gridPtr, *dof, true,  false, 2 );
        checkLeftEdge(  *gridPtr, *dof, false, false, 3 );
        checkRightEdge( *gridPtr, *dof, false, false, 5 );
        checkCorner(    *gridPtr, *dof, false, true,  6 );
        checkDownEdge(  *gridPtr, *dof, false, false, 7 );
        checkCorner(    *gridPtr, *dof, false, false, 8 );
    }
    
    if(rank==5)//Center Right
    {
        checkCorner(   *gridPtr, *dof, true,  true,  1 );
        checkUpEdge(   *gridPtr, *dof, false, true,  2 );
        checkLeftEdge( *gridPtr, *dof, false, false, 4 );
        checkCorner(   *gridPtr, *dof, false, true,  7 );
        checkDownEdge( *gridPtr, *dof, false, true,  8 );
    }
    
    if(rank==6)//Down Left
    {
        checkUpEdge(    *gridPtr, *dof, true,  false, 3 );
        checkCorner(    *gridPtr, *dof, true,  false, 4 );
        checkRightEdge( *gridPtr, *dof, false, true,  7 );
    }
    
    if(rank==7) //Down Center
    {
        checkCorner(    *gridPtr, *dof, true,  true,  3 );
        checkUpEdge(    *gridPtr, *dof, false, false, 4 );
        checkCorner(    *gridPtr, *dof, true,  false, 5 );
        checkLeftEdge(  *gridPtr, *dof, false, true,  6 );
        checkRightEdge( *gridPtr, *dof, false, true,  8 );
    }
    
    if(rank==8) //Down Right
    {
        checkCorner(   *gridPtr, *dof, true,  true, 4 );
        checkUpEdge(   *gridPtr, *dof, false, true, 5 );
        checkLeftEdge( *gridPtr, *dof, false, true, 7 );
    }   
}

// TODO: Fix tests for periodic BC - 
// checkLeftBoundary -> checkLeft Overlap etc. for direction BoundaryToOverlap
// Fix the tests with mask to work with the direction OverlapToBoundary
/*
TEST_F(DistributedGridTest_2D, SynchronizerNeighborPeriodicBoundariesWithoutMask )
{
   // Setup periodic boundaries
   // TODO: I do not know how to do it better with GTEST - additional setup 
   // of the periodic boundaries
   typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< GridType, CommunicatorType >::
      getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1, 1, 1 );
   distributedGrid->setOverlaps( lowerOverlap, upperOverlap );
   distributedGrid->setupGrid(*gridPtr);
   dof->setSize( gridPtr->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( gridPtr, *dof );
   
   //Expecting 9 processes
   setDof_2D(*dof, -rank-1 );
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   //meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   meshFunctionPtr->template synchronize<CommunicatorType>( true );
   
   if( rank == 0 )
   {
      SCOPED_TRACE( "Up Left" );
      checkLeftBoundary( *gridPtr, *dof, false,  true, -3 );
      checkUpBoundary(   *gridPtr, *dof, false,  true, -7 );
   }
    
   if( rank == 1 )
   {
      SCOPED_TRACE( "Up Center" );
      checkUpBoundary( *gridPtr, *dof, true, true, -8 );
   }
    
   if( rank == 2 )
   {
      SCOPED_TRACE( "Up Right" );
      checkRightBoundary( *gridPtr, *dof, false, true, -1 );
      checkUpBoundary(    *gridPtr, *dof, true, false, -9 );
   }
    
   if( rank == 3 )
   {
      SCOPED_TRACE( "Center Left" );
      checkLeftBoundary( *gridPtr, *dof, true, true, -6 );
   } 
        
   if( rank == 5 )
   {
      SCOPED_TRACE( "Center Right" );
      checkRightBoundary( *gridPtr, *dof, true, true, -4 );
   }
    
   if( rank == 6 )
   {
      SCOPED_TRACE( "Down Left" );
      checkDownBoundary( *gridPtr, *dof, false,  true, -1 );
      checkLeftBoundary( *gridPtr, *dof, true,  false,  -9 );
   }
    
   if( rank == 7 )
   {
      SCOPED_TRACE( "Down Center" );
      checkDownBoundary( *gridPtr, *dof, true, true, -2 );
   }

   if( rank == 8 )
   {
      SCOPED_TRACE( "Down Right" );
      checkDownBoundary(  *gridPtr, *dof, true, false, -3 );
      checkRightBoundary( *gridPtr, *dof, true, false, -7 );
   }
}

TEST_F(DistributedGridTest_2D, SynchronizerNeighborPeriodicBoundariesWithActiveMask )
{
   // Setup periodic boundaries
   // TODO: I do not know how to do it better with GTEST - additional setup 
   // of the periodic boundaries
   typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< GridType, CommunicatorType >::
      getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1, 1, 1 );
   distributedGrid->setOverlaps( lowerOverlap, upperOverlap );
   distributedGrid->setupGrid(*gridPtr);
   dof->setSize( gridPtr->template getEntitiesCount< Cell >() );
   maskDofs.setSize( gridPtr->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( gridPtr, *dof );
   maskPointer->bind( gridPtr, maskDofs );
   
   //Expecting 9 processes
   setDof_2D(*dof, -rank-1 );
   maskDofs.setValue( true );
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   meshFunctionPtr->template synchronize<CommunicatorType>( true, maskPointer );

   if( rank == 0 )
   {
      SCOPED_TRACE( "Up Left" );
      checkLeftBoundary( *gridPtr, *dof, false,  true, -3 );
      checkUpBoundary(   *gridPtr, *dof, false,  true, -7 );
   }
    
   if( rank == 1 )
   {
      SCOPED_TRACE( "Up Center" );
      checkUpBoundary( *gridPtr, *dof, true, true, -8 );
   }
    
   if( rank == 2 )
   {
      SCOPED_TRACE( "Up Right" );
      checkRightBoundary( *gridPtr, *dof, false, true, -1 );
      checkUpBoundary(    *gridPtr, *dof, true, false, -9 );
   }
    
   if( rank == 3 )
   {
      SCOPED_TRACE( "Center Left" );
      checkLeftBoundary( *gridPtr, *dof, true, true, -6 );
   } 
        
   if( rank == 5 )
   {
      SCOPED_TRACE( "Center Right" );
      checkRightBoundary( *gridPtr, *dof, true, true, -4 );
   }
    
   if( rank == 6 )
   {
      SCOPED_TRACE( "Down Left" );
      checkDownBoundary( *gridPtr, *dof, false,  true, -1 );
      checkLeftBoundary( *gridPtr, *dof, true,  false,  -9 );
   }
    
   if( rank == 7 )
   {
      SCOPED_TRACE( "Down Center" );
      checkDownBoundary( *gridPtr, *dof, true, true, -2 );
   }

   if( rank == 8 )
   {
      SCOPED_TRACE( "Down Right" );
      checkDownBoundary(  *gridPtr, *dof, true, false, -3 );
      checkRightBoundary( *gridPtr, *dof, true, false, -7 );
   }
}

TEST_F(DistributedGridTest_2D, SynchronizerNeighborPeriodicBoundariesWithInactiveMaskOnLeft )
{
   // Setup periodic boundaries
   // TODO: I do not know how to do it better with GTEST - additional setup 
   // of the periodic boundaries
   typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< GridType, CommunicatorType >::
      getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1, 1, 1 );
   distributedGrid->setOverlaps( lowerOverlap, upperOverlap );
   distributedGrid->setupGrid(*gridPtr);
   dof->setSize( gridPtr->template getEntitiesCount< Cell >() );
   maskDofs.setSize( gridPtr->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( gridPtr, *dof );
   maskPointer->bind( gridPtr, maskDofs );
   
   //Expecting 9 processes
   setDof_2D(*dof, -rank-1 );
   maskDofs.setValue( true );
   if( distributedGrid->getNeighbors()[ ZzYzXm ] == -1 )
   {
      for( IndexType i = 0; i < gridPtr->getDimensions().y(); i++ )
      {
         typename GridType::Cell cell( *gridPtr );
         cell.getCoordinates() = CoordinatesType( 1, i );
         cell.refresh();
         maskPointer->getData().setElement( cell.getIndex(), false );
      }
   }
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   meshFunctionPtr->template synchronize<CommunicatorType>( true, maskPointer );
   
   if( rank == 0 )
   {
      SCOPED_TRACE( "Up Left" );
      checkLeftBoundary( *gridPtr, *dof, false,  true, 0 );
      checkUpBoundary(   *gridPtr, *dof, false,  true, -7 );
   }
    
   if( rank == 1 )
   {
      SCOPED_TRACE( "Up Center" );
      checkUpBoundary( *gridPtr, *dof, true, true, -8 );
   }
    
   if( rank == 2 )
   {
      SCOPED_TRACE( "Up Right" );
      checkRightBoundary( *gridPtr, *dof, false, true, -1 );
      checkUpBoundary(    *gridPtr, *dof, true, false, -9 );
   }
    
   if( rank == 3 )
   {
      SCOPED_TRACE( "Center Left" );
      checkLeftBoundary( *gridPtr, *dof, true, true, 3 );
   } 
        
   if( rank == 5 )
   {
      SCOPED_TRACE( "Center Right" );
      checkRightBoundary( *gridPtr, *dof, true, true, -4 );
   }
    
   if( rank == 6 )
   {
      SCOPED_TRACE( "Down Left" );
      checkDownBoundary( *gridPtr, *dof, false,  true, -1 );
      checkLeftBoundary( *gridPtr, *dof, true,  false,  6 );
   }
    
   if( rank == 7 )
   {
      SCOPED_TRACE( "Down Center" );
      checkDownBoundary( *gridPtr, *dof, true, true, -2 );
   }

   if( rank == 8 )
   {
      SCOPED_TRACE( "Down Right" );
      checkDownBoundary(  *gridPtr, *dof, true, false, -3 );
      checkRightBoundary( *gridPtr, *dof, true, false, -7 );
   }
}

TEST_F(DistributedGridTest_2D, SynchronizerNeighborPeriodicBoundariesWithInActiveMaskOnRight )
{
   // Setup periodic boundaries
   // TODO: I do not know how to do it better with GTEST - additional setup 
   // of the periodic boundaries
   typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< GridType, CommunicatorType >::
      getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1, 1, 1 );
   distributedGrid->setOverlaps( lowerOverlap, upperOverlap );
   distributedGrid->setupGrid(*gridPtr);
   dof->setSize( gridPtr->template getEntitiesCount< Cell >() );
   maskDofs.setSize( gridPtr->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( gridPtr, *dof );
   maskPointer->bind( gridPtr, maskDofs );
   
   //Expecting 9 processes
   setDof_2D(*dof, -rank-1 );
   maskDofs.setValue( true );
   if( distributedGrid->getNeighbors()[ ZzYzXp ] == -1 )
   {
      for( IndexType i = 0; i < gridPtr->getDimensions().y(); i++ )
      {
         typename GridType::Cell cell( *gridPtr );
         cell.getCoordinates() = CoordinatesType( gridPtr->getDimensions().x() - 2, i );
         cell.refresh();
         maskPointer->getData().setElement( cell.getIndex(), false );
      }
   }
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   meshFunctionPtr->template synchronize<CommunicatorType>( true, maskPointer );
   
   if( rank == 0 )
   {
      SCOPED_TRACE( "Up Left" );
      checkLeftBoundary( *gridPtr, *dof, false,  true, -3 );
      checkUpBoundary(   *gridPtr, *dof, false,  true, -7 );
   }
    
   if( rank == 1 )
   {
      SCOPED_TRACE( "Up Center" );
      checkUpBoundary( *gridPtr, *dof, true, true, -8 );
   }
    
   if( rank == 2 )
   {
      SCOPED_TRACE( "Up Right" );
      checkRightBoundary( *gridPtr, *dof, false, true, 2 );
      checkUpBoundary(    *gridPtr, *dof, true, false, -9 );
   }
    
   if( rank == 3 )
   {
      SCOPED_TRACE( "Center Left" );
      checkLeftBoundary( *gridPtr, *dof, true, true, -6 );
   } 
        
   if( rank == 5 )
   {
      SCOPED_TRACE( "Center Right" );
      checkRightBoundary( *gridPtr, *dof, true, true, 5 );
   }
    
   if( rank == 6 )
   {
      SCOPED_TRACE( "Down Left" );
      checkDownBoundary( *gridPtr, *dof, false,  true, -1 );
      checkLeftBoundary( *gridPtr, *dof, true,  false,  -9 );
   }
    
   if( rank == 7 )
   {
      SCOPED_TRACE( "Down Center" );
      checkDownBoundary( *gridPtr, *dof, true, true, -2 );
   }

   if( rank == 8 )
   {
      SCOPED_TRACE( "Down Right" );
      checkDownBoundary(  *gridPtr, *dof, true, false, -3 );
      checkRightBoundary( *gridPtr, *dof, true, false, 8 );
   }
}

TEST_F(DistributedGridTest_2D, SynchronizerNeighborPeriodicBoundariesWithInActiveMaskUp )
{
   // Setup periodic boundaries
   // TODO: I do not know how to do it better with GTEST - additional setup 
   // of the periodic boundaries
   typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< GridType, CommunicatorType >::
      getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1, 1, 1 );
   distributedGrid->setOverlaps( lowerOverlap, upperOverlap );
   distributedGrid->setupGrid(*gridPtr);
   dof->setSize( gridPtr->template getEntitiesCount< Cell >() );
   maskDofs.setSize( gridPtr->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( gridPtr, *dof );
   maskPointer->bind( gridPtr, maskDofs );
   
   //Expecting 9 processes
   setDof_2D(*dof, -rank-1 );
   maskDofs.setValue( true );
   if( distributedGrid->getNeighbors()[ ZzYmXz ] == -1 )
   {
      for( IndexType i = 0; i < gridPtr->getDimensions().x(); i++ )
      {
         typename GridType::Cell cell( *gridPtr );
         cell.getCoordinates() = CoordinatesType( i, 1 );
         cell.refresh();
         maskPointer->getData().setElement( cell.getIndex(), false );
      }
   }
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   meshFunctionPtr->template synchronize<CommunicatorType>( true, maskPointer );
   
   if( rank == 0 )
   {
      SCOPED_TRACE( "Up Left" );
      checkLeftBoundary( *gridPtr, *dof, false,  true, -3 );
      checkUpBoundary(   *gridPtr, *dof, false,  true, 0 );
   }
    
   if( rank == 1 )
   {
      SCOPED_TRACE( "Up Center" );
      checkUpBoundary( *gridPtr, *dof, true, true, 1 );
   }
    
   if( rank == 2 )
   {
      SCOPED_TRACE( "Up Right" );
      checkRightBoundary( *gridPtr, *dof, false, true, -1 );
      checkUpBoundary(    *gridPtr, *dof, true, false, 2 );
   }
    
   if( rank == 3 )
   {
      SCOPED_TRACE( "Center Left" );
      checkLeftBoundary( *gridPtr, *dof, true, true, -6 );
   } 
        
   if( rank == 5 )
   {
      SCOPED_TRACE( "Center Right" );
      checkRightBoundary( *gridPtr, *dof, true, true, -4 );
   }
    
   if( rank == 6 )
   {
      SCOPED_TRACE( "Down Left" );
      checkDownBoundary( *gridPtr, *dof, false,  true, -1 );
      checkLeftBoundary( *gridPtr, *dof, true,  false,  -9 );
   }
    
   if( rank == 7 )
   {
      SCOPED_TRACE( "Down Center" );
      checkDownBoundary( *gridPtr, *dof, true, true, -2 );
   }

   if( rank == 8 )
   {
      SCOPED_TRACE( "Down Right" );
      checkDownBoundary(  *gridPtr, *dof, true, false, -3 );
      checkRightBoundary( *gridPtr, *dof, true, false, -7 );
   }
}

TEST_F(DistributedGridTest_2D, SynchronizerNeighborPeriodicBoundariesWithInActiveMaskDown )
{
   // Setup periodic boundaries
   // TODO: I do not know how to do it better with GTEST - additional setup 
   // of the periodic boundaries
   typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< GridType, CommunicatorType >::
      getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1, 1, 1 );
   distributedGrid->setOverlaps( lowerOverlap, upperOverlap );
   distributedGrid->setupGrid(*gridPtr);
   dof->setSize( gridPtr->template getEntitiesCount< Cell >() );
   maskDofs.setSize( gridPtr->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( gridPtr, *dof );
   maskPointer->bind( gridPtr, maskDofs );
   
   //Expecting 9 processes
   setDof_2D(*dof, -rank-1 );
   maskDofs.setValue( true );
   if( distributedGrid->getNeighbors()[ ZzYpXz ] == -1 )
   {
      for( IndexType i = 0; i < gridPtr->getDimensions().x(); i++ )
      {
         typename GridType::Cell cell( *gridPtr );
         cell.getCoordinates() = CoordinatesType( i, gridPtr->getDimensions().y() - 2 );
         cell.refresh();
         maskPointer->getData().setElement( cell.getIndex(), false );
      }
   }
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   meshFunctionPtr->template synchronize<CommunicatorType>( true, maskPointer );
   
   if( rank == 0 )
   {
      SCOPED_TRACE( "Up Left" );
      checkLeftBoundary( *gridPtr, *dof, false,  true, -3 );
      checkUpBoundary(   *gridPtr, *dof, false,  true, -7 );
   }
    
   if( rank == 1 )
   {
      SCOPED_TRACE( "Up Center" );
      checkUpBoundary( *gridPtr, *dof, true, true, -8 );
   }
    
   if( rank == 2 )
   {
      SCOPED_TRACE( "Up Right" );
      checkRightBoundary( *gridPtr, *dof, false, true, -1 );
      checkUpBoundary(    *gridPtr, *dof, true, false, -9 );
   }
    
   if( rank == 3 )
   {
      SCOPED_TRACE( "Center Left" );
      checkLeftBoundary( *gridPtr, *dof, true, true, -6 );
   } 
        
   if( rank == 5 )
   {
      SCOPED_TRACE( "Center Right" );
      checkRightBoundary( *gridPtr, *dof, true, true, -4 );
   }
    
   if( rank == 6 )
   {
      SCOPED_TRACE( "Down Left" );
      checkDownBoundary( *gridPtr, *dof, false,  true, 6 );
      checkLeftBoundary( *gridPtr, *dof, true,  false,  -9 );
   }
    
   if( rank == 7 )
   {
      SCOPED_TRACE( "Down Center" );
      checkDownBoundary( *gridPtr, *dof, true, true, 7 );
   }

   if( rank == 8 )
   {
      SCOPED_TRACE( "Down Right" );
      checkDownBoundary(  *gridPtr, *dof, true, false, 8 );
      checkRightBoundary( *gridPtr, *dof, true, false, -7 );
   }
}
*/ 
#endif

#endif

#include "../../main_mpi.h"
