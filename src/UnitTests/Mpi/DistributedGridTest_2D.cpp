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
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Meshes/DistributedMeshes/SubdomainOverlapsGetter.h>

#include "Functions.h"

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
void checkLeftEdge(GridType &grid, DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue)
{
    int maxx=grid.getDimensions().x();
    int maxy=grid.getDimensions().y();
    int begin=0;
    int end=maxy;
    if(!with_first)
        begin++;
    if(!with_last)
        end--;
    
    for(int i=begin;i<end;i++) //posledni je overlap
            EXPECT_EQ( dof[maxx*i], expectedValue) << "Left Edge test failed " << i<<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkRightEdge(GridType &grid, DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue)
{
    int maxx=grid.getDimensions().x();
    int maxy=grid.getDimensions().y();
    int begin=0;
    int end=maxy;
    if(!with_first)
        begin++;
    if(!with_last)
        end--;
    
    for(int i=begin;i<end;i++) 
            EXPECT_EQ( dof[maxx*i+(maxx-1)], expectedValue) << "Right Edge test failed " << i <<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkUpEdge(GridType &grid, DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue)
{
    int maxx=grid.getDimensions().x();
    int maxy=grid.getDimensions().y();
    int begin=0;
    int end=maxx;
    if(!with_first)
        begin++;
    if(!with_last)
        end--;
    
    for(int i=begin;i<end;i++) //posledni je overlap
            EXPECT_EQ( dof[i], expectedValue) << "Up Edge test failed " << i<<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkDownEdge(GridType &grid, DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue)
{
    int maxx=grid.getDimensions().x();
    int maxy=grid.getDimensions().y();
    int begin=0;
    int end=maxx;
    if(!with_first)
        begin++;
    if(!with_last)
        end--;
    
    for(int i=begin;i<end;i++) //posledni je overlap
            EXPECT_EQ( dof[maxx*(maxy-1)+i], expectedValue) << "Down Edge test failed " << i<<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkConner(GridType &grid, DofType &dof, bool up, bool left, typename DofType::RealType expectedValue )
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


/*expect 9 process*/
template<typename DofType,typename GridType>
void check_Boundary_2D(int rank, GridType &grid, DofType &dof, typename DofType::RealType expectedValue)
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

/*expect 9 process
 * Known BUG of Traversars: Process boundary is writing over overlap.
 * it should be true, true, every where, but we dont chcek boundary overalp on boundary
 * so boundary overlap is not checked (it is filled incorectly by boundary condition).
 */
template<typename DofType,typename GridType>
void check_Overlap_2D(int rank, GridType &grid, DofType &dof, typename DofType::RealType expectedValue)
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

/*Expect 9 process
 */
template<typename DofType,typename GridType>
void checkNeighbor_2D(int rank, GridType &grid, DofType &dof)
{
    if(rank==0)//Up Left
    {
        checkRightEdge(grid,dof,true,false,1);
        checkDownEdge(grid,dof,true,false,3);
        checkConner(grid,dof,false,false,4);
        
    }
    
    if(rank==1)//Up Center
    {
        checkLeftEdge(grid,dof,true,false,0);
        checkRightEdge(grid,dof,true,false,2);
        checkConner(grid,dof,false,true,3);
        checkDownEdge(grid,dof,false,false,4);
        checkConner(grid,dof,false,false,5);
    }
    
    if(rank==2)//Up Right
    {
        checkLeftEdge(grid,dof,true,false,1);
        checkConner(grid,dof,false,true,4);
        checkDownEdge(grid,dof,false,true,5);
    }
    
    if(rank==3)//Center Left
    {
        checkUpEdge(grid,dof,true,false,0);
        checkConner(grid,dof,true,false,1);
        checkRightEdge(grid,dof,false,false,4);
        checkDownEdge(grid,dof,true,false,6);
        checkConner(grid,dof,false,false,7);
    }
    
    if(rank==4)//Center Center
    {
        checkConner(grid,dof,true,true,0);
        checkUpEdge(grid,dof,false,false,1);
        checkConner(grid,dof,true,false,2);
        checkLeftEdge(grid,dof,false,false,3);
        checkRightEdge(grid,dof,false,false,5);
        checkConner(grid,dof,false,true,6);
        checkDownEdge(grid,dof,false,false,7);
        checkConner(grid,dof,false,false,8);
    }
    
    if(rank==5)//Center Right
    {
        checkConner(grid,dof,true,true,1);
        checkUpEdge(grid,dof,false,true,2);
        checkLeftEdge(grid,dof,false,false,4);
        checkConner(grid,dof,false,true,7);
        checkDownEdge(grid,dof,false,true,8);
    }
    
    if(rank==6)//Down Left
    {
        checkUpEdge(grid,dof,true,false,3);
        checkConner(grid,dof,true,false,4);
        checkRightEdge(grid,dof,false,true,7);
    }
    
    if(rank==7) //Down Center
    {
        checkConner(grid,dof,true,true,3);
        checkUpEdge(grid,dof,false,false,4);
        checkConner(grid,dof,true,false,5);
        checkLeftEdge(grid,dof,false,true,6);
        checkRightEdge(grid,dof,false,true,8);
    }
    
    if(rank==8) //Down Right
    {
        checkConner(grid,dof,true,true,4);
        checkUpEdge(grid,dof,false,true,5);
        checkLeftEdge(grid,dof,false,true,7);
    }
}


template<typename DofType,typename GridType>
void check_Inner_2D(int rank, GridType grid, DofType dof, typename DofType::RealType expectedValue)
{
    int maxx=grid.getDimensions().x();
    int maxy=grid.getDimensions().y();
    for(int j=1;j<maxy-1;j++)//prvni a posledni jsou buď hranice, nebo overlap
        for(int i=1;i<maxx-1;i++) //buď je vlevo hranice, nebo overlap
            EXPECT_EQ( dof[j*maxx+i], expectedValue) << " "<< j<<" "<<i << " " << maxx << " " << maxy;
}





/*
 * Light check of 2D distributed grid and its synchronization. 
 * expected 9 processors
 */

typedef MpiCommunicator CommunicatorType;
typedef Grid<2,double,Host,int> MeshType;
typedef MeshFunction<MeshType> MeshFunctionType;
typedef Vector<double,Host,int> DofType;
typedef typename MeshType::Cell Cell;
typedef typename MeshType::IndexType IndexType; 
typedef typename MeshType::PointType PointType; 
typedef DistributedMesh<MeshType> DistributedGridType;

class DistributedGirdTest_2D : public ::testing::Test {
 protected:
    
public:

    static DistributedGridType *distrgrid;
    static DofType *dof;

    static SharedPointer<MeshType> gridptr;
    static SharedPointer<MeshFunctionType> meshFunctionptr;

    static MeshFunctionEvaluator< MeshFunctionType, ConstFunction<double,2> > constFunctionEvaluator;
    static SharedPointer< ConstFunction<double,2>, Host > constFunctionPtr;

    static MeshFunctionEvaluator< MeshFunctionType, LinearFunction<double,2> > linearFunctionEvaluator;
    static SharedPointer< LinearFunction<double,2>, Host > linearFunctionPtr;

    static int rank;
    static int nproc;    
     
  // Per-test-case set-up.
  // Called before the first test in this test case.
  // Can be omitted if not needed.
  static void SetUpTestCase() {
      
    int size=10;
    rank=CommunicatorType::GetRank(CommunicatorType::AllGroup);
    nproc=CommunicatorType::GetSize(CommunicatorType::AllGroup);
    
    PointType globalOrigin;
    PointType globalProportions;
    MeshType globalGrid;
    
    globalOrigin.x()=-0.5;
    globalOrigin.y()=-0.5;    
    globalProportions.x()=size;
    globalProportions.y()=size;
        
    globalGrid.setDimensions(size,size);
    globalGrid.setDomain(globalOrigin,globalProportions);
    
    typename DistributedGridType::CoordinatesType overlap( 1 );
    
    typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
    distrgrid=new DistributedGridType();

    distrgrid->setDomainDecomposition( typename DistributedGridType::CoordinatesType( 3, 3 ) );
    distrgrid->template setGlobalGrid<CommunicatorType>( globalGrid );
    distrgrid->setupGrid(*gridptr);    
    SubdomainOverlapsGetter< MeshType, CommunicatorType >::getOverlaps( *gridptr, lowerOverlap, upperOverlap, 1 );
    distrgrid->setOverlaps( lowerOverlap, upperOverlap );
    distrgrid->setupGrid(*gridptr);
    
    
    dof=new DofType(gridptr->template getEntitiesCount< Cell >());
    
    meshFunctionptr->bind(gridptr,*dof);
    
    constFunctionPtr->Number=rank;
  }

  // Per-test-case tear-down.
  // Called after the last test in this test case.
  // Can be omitted if not needed.
  static void TearDownTestCase() {
      delete dof;
      delete distrgrid;

  }

};

DistributedMesh<MeshType> *DistributedGirdTest_2D::distrgrid=NULL;
DofType *DistributedGirdTest_2D::dof=NULL;
SharedPointer<MeshType> DistributedGirdTest_2D::gridptr;
SharedPointer<MeshFunctionType> DistributedGirdTest_2D::meshFunctionptr;
MeshFunctionEvaluator< MeshFunctionType, ConstFunction<double,2> > DistributedGirdTest_2D::constFunctionEvaluator;
SharedPointer< ConstFunction<double,2>, Host > DistributedGirdTest_2D::constFunctionPtr;
MeshFunctionEvaluator< MeshFunctionType, LinearFunction<double,2> > DistributedGirdTest_2D::linearFunctionEvaluator;
SharedPointer< LinearFunction<double,2>, Host > DistributedGirdTest_2D::linearFunctionPtr;
int DistributedGirdTest_2D::rank;
int DistributedGirdTest_2D::nproc;    

TEST_F(DistributedGirdTest_2D, evaluateAllEntities)
{
    //Check Traversars
    //All entities, without overlap
    setDof_2D(*dof,-1);
    constFunctionEvaluator.evaluateAllEntities( meshFunctionptr , constFunctionPtr );
    //Printer<MeshType,DofType>::print_dof(rank,*gridptr,*dof);
    check_Boundary_2D(rank, *gridptr, *dof, rank);
    check_Overlap_2D(rank, *gridptr, *dof, -1);
    check_Inner_2D(rank, *gridptr, *dof, rank);
}

TEST_F(DistributedGirdTest_2D, evaluateBoundaryEntities)
{
    //Boundary entities, without overlap
    setDof_2D(*dof,-1);
    constFunctionEvaluator.evaluateBoundaryEntities( meshFunctionptr , constFunctionPtr );
    //print_dof_2D(rank,*gridptr,dof);
    check_Boundary_2D(rank, *gridptr, *dof, rank);
    check_Overlap_2D(rank, *gridptr, *dof, -1);
    check_Inner_2D(rank, *gridptr, *dof, -1);
}

TEST_F(DistributedGirdTest_2D, evaluateInteriorEntities)
{
    //Inner entities, without overlap
    setDof_2D(*dof,-1);
    constFunctionEvaluator.evaluateInteriorEntities( meshFunctionptr , constFunctionPtr );
    check_Boundary_2D(rank, *gridptr, *dof, -1);
    check_Overlap_2D(rank, *gridptr, *dof, -1);
    check_Inner_2D(rank, *gridptr, *dof, rank);
}    

TEST_F(DistributedGirdTest_2D, LinearFunctionTest)
{
    //fill meshfunction with linear function (physical center of cell corresponds with its coordinates in grid) 
    setDof_2D(*dof,-1);
    linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr, linearFunctionPtr);
    meshFunctionptr->template synchronize<CommunicatorType>();
    
    int count =gridptr->template getEntitiesCount< Cell >();
    for(int i=0;i<count;i++)
    {
            auto entity= gridptr->template getEntity< Cell >(i);
            entity.refresh();
            EXPECT_EQ(meshFunctionptr->getValue(entity), (*linearFunctionPtr)(entity)) << "Linear function doesnt fit recievd data. " << entity.getCoordinates().x() << " "<<entity.getCoordinates().y() << " "<< gridptr->getDimensions().x() <<" "<<gridptr->getDimensions().y();
    }
}

TEST_F(DistributedGirdTest_2D, SynchronizerNeighborTest)
{
    setDof_2D(*dof,-1);
    constFunctionEvaluator.evaluateAllEntities( meshFunctionptr , constFunctionPtr );
    meshFunctionptr->template synchronize<CommunicatorType>();
    checkNeighbor_2D(rank, *gridptr, *dof);
}


#else
TEST(NoMPI, NoTest)
{
    ASSERT_TRUE(true) << ":-(";
}
#endif

#endif


#if (defined(HAVE_GTEST) && defined(HAVE_MPI))
#include <sstream>

  class MinimalistBufferedPrinter : public ::testing::EmptyTestEventListener {
      
  private:
      std::stringstream sout;
      
  public:
      
    // Called before a test starts.
    virtual void OnTestStart(const ::testing::TestInfo& test_info) {
      sout<< test_info.test_case_name() <<"." << test_info.name() << " Start." <<std::endl;
    }

    // Called after a failed assertion or a SUCCEED() invocation.
    virtual void OnTestPartResult(
        const ::testing::TestPartResult& test_part_result) {
      sout << (test_part_result.failed() ? "====Failure=== " : "===Success=== ") 
              << test_part_result.file_name() << " "
              << test_part_result.line_number() <<std::endl
              << test_part_result.summary() <<std::endl;
    }

    // Called after a test ends.
    virtual void OnTestEnd(const ::testing::TestInfo& test_info) 
    {
        int rank=CommunicatorType::GetRank(CommunicatorType::AllGroup);
        sout<< test_info.test_case_name() <<"." << test_info.name() << " End." <<std::endl;
        std::cout << rank << ":" << std::endl << sout.str()<< std::endl;
        sout.str( std::string() );
        sout.clear();
    }
  };
#endif

#include "../../src/UnitTests/GtestMissingError.h"
int main( int argc, char* argv[] )
{
   /*CommunicatorType::Init(argc,argv);
   
    DistributedGridType *distrgrid;
    DofType *dof;

    SharedPointer<MeshType> gridptr;
    SharedPointer<MeshFunctionType> meshFunctionptr;

     MeshFunctionEvaluator< MeshFunctionType, ConstFunction<double,2> > constFunctionEvaluator;
     SharedPointer< ConstFunction<double,2>, Host > constFunctionPtr;

     MeshFunctionEvaluator< MeshFunctionType, LinearFunction<double,2> > linearFunctionEvaluator;
     SharedPointer< LinearFunction<double,2>, Host > linearFunctionPtr;
   
     int rank;
    int nproc;    
   
   
   
    int size=10;
    rank=CommunicatorType::GetRank(CommunicatorType::AllGroup);
    nproc=CommunicatorType::GetSize(CommunicatorType::AllGroup);
    
    PointType globalOrigin;
    PointType globalProportions;
    MeshType globalGrid;
    
    globalOrigin.x()=-0.5;
    globalOrigin.y()=-0.5;    
    globalProportions.x()=size;
    globalProportions.y()=size;
        
    globalGrid.setDimensions(size,size);
    globalGrid.setDomain(globalOrigin,globalProportions);
    
    typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
    distrgrid=new DistributedGridType();
    distrgrid->setDomainDecomposition( typename DistributedGridType::CoordinatesType( 3, 3 ) );
    distrgrid->template setGlobalGrid<CommunicatorType>( globalGrid );
    distrgrid->setupGrid(*gridptr);    
    SubdomainOverlapsGetter< MeshType, CommunicatorType >::getOverlaps( *gridptr, lowerOverlap, upperOverlap, 1 );
    distrgrid->setOverlaps( lowerOverlap, upperOverlap );
    
    dof=new DofType(gridptr->template getEntitiesCount< Cell >());
    
    meshFunctionptr->bind(gridptr,*dof);
    
    constFunctionPtr->Number=rank;
   
   
   
   return 0;*/
   
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );

    #ifdef HAVE_MPI
       ::testing::TestEventListeners& listeners =
          ::testing::UnitTest::GetInstance()->listeners();

       delete listeners.Release(listeners.default_result_printer());
       listeners.Append(new MinimalistBufferedPrinter);

       CommunicatorType::Init(argc,argv);
    #endif
       int result= RUN_ALL_TESTS();

    #ifdef HAVE_MPI
       CommunicatorType::Finalize();
    #endif
       return result;
#else
   
   throw GtestMissingError();
#endif
}


