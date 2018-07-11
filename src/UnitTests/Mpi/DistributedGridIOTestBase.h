/***************************************************************************
                          DistributedGridIO.h  -  description
                             -------------------
    begin                : Nov 1, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/
#ifdef HAVE_GTEST
  
#include <gtest/gtest.h>

#ifdef HAVE_MPI

#include "DistributedGridIOTest.h"

TEST( DistributedGridIO, Save_1D )
{
    TestDistributedGridIO<1,Host>::TestSave();
}

TEST( DistributedGridIO, Save_2D )
{
    TestDistributedGridIO<2,Host>::TestSave();
}

TEST( DistributedGridIO, Save_3D )
{
    TestDistributedGridIO<3,Host>::TestSave();
}

TEST( DistributedGridIO, Load_1D )
{
    TestDistributedGridIO<1,Host>::TestLoad();
}

TEST( DistributedGridIO, Load_2D )
{
    TestDistributedGridIO<2,Host>::TestLoad();
}

TEST( DistributedGridIO, Load_3D )
{
    TestDistributedGridIO<3,Host>::TestLoad();
}

#ifdef HAVE_CUDA
TEST( DistributedGridIO, Save_1D_GPU )
{
    TestDistributedGridIO<1,Cuda>::TestSave();
}

TEST( DistributedGridIO, Save_2D_GPU )
{
    TestDistributedGridIO<2,Cuda>::TestSave();
}

TEST( DistributedGridIO, Save_3D_GPU )
{
    TestDistributedGridIO<3,Cuda>::TestSave();
}

TEST( DistributedGridIO, Load_1D_GPU )
{
    TestDistributedGridIO<1,Cuda>::TestLoad();
}

TEST( DistributedGridIO, Load_2D_GPU )
{
    TestDistributedGridIO<2,Cuda>::TestLoad();
}

TEST( DistributedGridIO, Load_3D_GPU )
{
    TestDistributedGridIO<3,Cuda>::TestLoad();
}

#endif

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

#include "../GtestMissingError.h"
int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );

    #ifdef HAVE_MPI
       ::testing::TestEventListeners& listeners =
          ::testing::UnitTest::GetInstance()->listeners();

       delete listeners.Release(listeners.default_result_printer());
       listeners.Append(new MinimalistBuffredPrinter);

       CommunicatorType::Init(argc,argv );
       CommunicatorType::setRedirection( false );
       CommunicatorType::setupRedirection();
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
