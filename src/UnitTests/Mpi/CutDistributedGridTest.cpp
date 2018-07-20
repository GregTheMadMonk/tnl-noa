#ifdef HAVE_GTEST  
#include <gtest/gtest.h>

#ifdef HAVE_MPI  

#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>


using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Meshes::DistributedMeshes;
using namespace TNL::Devices;
using namespace TNL::Communicators;


typedef MpiCommunicator CommunicatorType;

template<
        typename MeshType,
        typename CommunicatorType>
void SetUpDistributedGrid(DistributedMesh<MeshType> &distributedGrid, MeshType &globalGrid,int size,typename MeshType::CoordinatesType distribution )
{
    typename MeshType::PointType globalOrigin;
    typename MeshType::PointType globalProportions;
    
    globalOrigin.setValue(-0.5);    
    globalProportions.setValue(size);

    globalGrid.setDimensions(size);
    globalGrid.setDomain(globalOrigin,globalProportions);
    
    typename MeshType::CoordinatesType overlap;
    overlap.setValue(1);
    distributedGrid.setDomainDecomposition( distribution );
    distributedGrid.template setGlobalGrid<CommunicatorType>(globalGrid,overlap);
}

//===============================================2D================================================================

TEST(CutDistributedGirdTest_2D, IsInCut)
{
    typedef Grid<2,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType; 
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<1,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType,CommunicatorType>(distributedGrid,globalGrid, 10, CoordinatesType(3,4));

    CutDistributedGridType cutDistributedGrid;
    bool result=cutDistributedGrid.SetupByCut<CommunicatorType>(
            distributedGrid,
            StaticVector<1,int>(1),
            StaticVector<1,int>(0),
            StaticVector<1,int>(5)
            );

    if(CommunicatorType::GetRank(CommunicatorType::AllGroup)%3==1)
    {
        ASSERT_TRUE(result); 
    }
    else
    {
        ASSERT_FALSE(result);
    }  
}

TEST(CutDistributedGirdTest_2D, GloblaGridDimesion)
{
    typedef Grid<2,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType; 
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<1,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType,CommunicatorType>(distributedGrid, globalGrid, 10, CoordinatesType(3,4));

    CutDistributedGridType cutDistributedGrid;
    if(cutDistributedGrid.SetupByCut<CommunicatorType>(
            distributedGrid,
            StaticVector<1,int>(1),
            StaticVector<1,int>(0),
            StaticVector<1,int>(5)
            ))
    {
        EXPECT_EQ(cutDistributedGrid.getGlobalGrid().getMeshDimension(),1) << "Dimenze globálního gridu neodpovídajá řezu";
        EXPECT_EQ(cutDistributedGrid.getGlobalGrid().getDimensions().x(),10) << "Rozměry globálního gridu neodpovídají"; 
    }
}

TEST(CutDistributedGirdTest_2D, IsDistributed)
{
    typedef Grid<2,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType; 
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<1,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType,CommunicatorType>(distributedGrid,globalGrid, 10, CoordinatesType(3,4));

    CutDistributedGridType cutDistributedGrid;
    if(cutDistributedGrid.SetupByCut<CommunicatorType>(
            distributedGrid,
            StaticVector<1,int>(1),
            StaticVector<1,int>(0),
            StaticVector<1,int>(5)
            ))
    {
        EXPECT_TRUE(cutDistributedGrid.isDistributed()) << "Řez by měl být distribuovaný";
    }
}

TEST(CutDistributedGirdTest_2D, IsNotDistributed)
{
    typedef Grid<2,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType; 
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<1,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType,CommunicatorType>(distributedGrid,globalGrid, 10, CoordinatesType(12,1));

    CutDistributedGridType cutDistributedGrid;
    if(cutDistributedGrid.SetupByCut<CommunicatorType>(
            distributedGrid,
            StaticVector<1,int>(1),
            StaticVector<1,int>(0),
            StaticVector<1,int>(5)
            ))
    {
        EXPECT_FALSE(cutDistributedGrid.isDistributed()) << "Řez by neměl být distribuovaný";
    }
}

//===============================================3D - 1D cut================================================================

TEST(CutDistributedGirdTest_3D, IsInCut_1D)
{
    typedef Grid<3,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType; 
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<1,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType,CommunicatorType>(distributedGrid,globalGrid, 10, CoordinatesType(2,2,3));

    CutDistributedGridType cutDistributedGrid;
    bool result=cutDistributedGrid.SetupByCut<CommunicatorType>(
            distributedGrid,
            StaticVector<1,int>(2),
            StaticVector<2,int>(0,1),
            StaticVector<2,int>(2,2)
            );

    if(CommunicatorType::GetRank(CommunicatorType::AllGroup)%4==0)
    {
        ASSERT_TRUE(result); 
    }
    else
    {
        ASSERT_FALSE(result);
    }  
}

TEST(CutDistributedGirdTest_3D, GloblaGridDimesion_1D)
{
    typedef Grid<3,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType; 
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<1,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType,CommunicatorType>(distributedGrid, globalGrid, 10, CoordinatesType(2,2,3));

    CutDistributedGridType cutDistributedGrid;
    if(cutDistributedGrid.SetupByCut<CommunicatorType>(
            distributedGrid,
            StaticVector<1,int>(2),
            StaticVector<2,int>(0,1),
            StaticVector<2,int>(2,2)
            ))
    {
        EXPECT_EQ(cutDistributedGrid.getGlobalGrid().getMeshDimension(),1) << "Dimenze globálního gridu neodpovídajá řezu";
        EXPECT_EQ(cutDistributedGrid.getGlobalGrid().getDimensions().x(),10) << "Rozměry globálního gridu neodpovídají"; 
    }
}

TEST(CutDistributedGirdTest_3D, IsDistributed_1D)
{
    typedef Grid<3,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType; 
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<1,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType,CommunicatorType>(distributedGrid,globalGrid, 10, CoordinatesType(2,2,3));

    CutDistributedGridType cutDistributedGrid;
    if(cutDistributedGrid.SetupByCut<CommunicatorType>(
            distributedGrid,
            StaticVector<1,int>(2),
            StaticVector<2,int>(0,1),
            StaticVector<2,int>(2,2)
            ))
    {
        EXPECT_TRUE(cutDistributedGrid.isDistributed()) << "Řez by měl být distribuovaný";
    }
}

TEST(CutDistributedGirdTest_3D, IsNotDistributed_1D)
{
    typedef Grid<3,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType; 
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<1,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType,CommunicatorType>(distributedGrid,globalGrid, 30, CoordinatesType(12,1,1));

    CutDistributedGridType cutDistributedGrid;
    if(cutDistributedGrid.SetupByCut<CommunicatorType>(
            distributedGrid,
            StaticVector<1,int>(2),
            StaticVector<2,int>(0,1),
            StaticVector<2,int>(1,1)
            ))
    {
        EXPECT_FALSE(cutDistributedGrid.isDistributed()) << "Řez by neměl být distribuovaný";
    }
}

//===================================3D-2D cut=========================================================================

TEST(CutDistributedGirdTest_3D, IsInCut_2D)
{
    typedef Grid<3,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType; 
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<2,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType,CommunicatorType>(distributedGrid,globalGrid, 10, CoordinatesType(2,2,3));

    CutDistributedGridType cutDistributedGrid;
    bool result=cutDistributedGrid.SetupByCut<CommunicatorType>(
            distributedGrid,
            StaticVector<2,int>(0,1),
            StaticVector<1,int>(2),
            StaticVector<1,int>(5)
            );

    int rank=CommunicatorType::GetRank(CommunicatorType::AllGroup);
    if(rank>3 && rank<8)
    {
        ASSERT_TRUE(result); 
    }
    else
    {
        ASSERT_FALSE(result);
    }  
}

TEST(CutDistributedGirdTest_3D, GloblaGridDimesion_2D)
{
    typedef Grid<3,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType; 
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<2,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType,CommunicatorType>(distributedGrid, globalGrid, 10, CoordinatesType(2,2,3));

    CutDistributedGridType cutDistributedGrid;
    if(cutDistributedGrid.SetupByCut<CommunicatorType>(
            distributedGrid,
            StaticVector<2,int>(0,1),
            StaticVector<1,int>(2),
            StaticVector<1,int>(5)
            ))
    {
        EXPECT_EQ(cutDistributedGrid.getGlobalGrid().getMeshDimension(),2) << "Dimenze globálního gridu neodpovídajá řezu";
        EXPECT_EQ(cutDistributedGrid.getGlobalGrid().getDimensions().x(),10) << "Rozměry globálního gridu neodpovídají"; 
        EXPECT_EQ(cutDistributedGrid.getGlobalGrid().getDimensions().y(),10) << "Rozměry globálního gridu neodpovídají";
    }
}

TEST(CutDistributedGirdTest_3D, IsDistributed_2D)
{
    typedef Grid<3,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType; 
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<2,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType,CommunicatorType>(distributedGrid,globalGrid, 10, CoordinatesType(2,2,3));

    CutDistributedGridType cutDistributedGrid;
    if(cutDistributedGrid.SetupByCut<CommunicatorType>(
            distributedGrid,
            StaticVector<2,int>(0,1),
            StaticVector<1,int>(2),
            StaticVector<1,int>(5)
            ))
    {
        EXPECT_TRUE(cutDistributedGrid.isDistributed()) << "Řez by měl být distribuovaný";
    }
}

TEST(CutDistributedGirdTest_3D, IsNotDistributed_2D)
{
    typedef Grid<3,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType; 
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<2,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType,CommunicatorType>(distributedGrid,globalGrid, 30, CoordinatesType(1,1,12));

    CutDistributedGridType cutDistributedGrid;
    if(cutDistributedGrid.SetupByCut<CommunicatorType>(
            distributedGrid,
            StaticVector<2,int>(0,1),
            StaticVector<1,int>(2),
            StaticVector<1,int>(5)
            ))
    {
        EXPECT_FALSE(cutDistributedGrid.isDistributed()) << "Řez by neměl být distribuovaný";
    }
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

  class MinimalistBuffredPrinter : public ::testing::EmptyTestEventListener {
      
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
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );

    #ifdef HAVE_MPI
       ::testing::TestEventListeners& listeners =
          ::testing::UnitTest::GetInstance()->listeners();

       delete listeners.Release(listeners.default_result_printer());
       listeners.Append(new MinimalistBuffredPrinter);

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

